"""
FastAPI reverse proxy for capturing LLM API calls across multiple providers (OpenAI, Gemini, Anthropic, generic).
"""

import time
import json
from typing import AsyncGenerator
from fastapi import FastAPI, Request, Response, Header
from fastapi.responses import StreamingResponse
import httpx

from blackbox.traces.recorder import TraceRecorder
from blackbox.traces.schemas import RawCallCreate
from blackbox.traces.models import init_db

app = FastAPI(title="Black Box Proxy", description="Universal AI Agent Flight Recorder Proxy")

# Default upstream bases (configurable via headers/env)
DEFAULT_UPSTREAMS = {
    "openai": "https://api.openai.com",
    "google": "https://generativelanguage.googleapis.com",
    "anthropic": "https://api.anthropic.com",
}

RUN_CALL_COUNTERS = {}


@app.on_event("startup")
async def on_startup():
    await init_db()


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Black Box Proxy"}


async def forward_and_record(
    request: Request,
    provider: str,
    target_base: str,
    path: str
) -> Response:
    start_time = time.time()
    run_id = request.headers.get("x-blackbox-run-id", "default_run")
    
    # Increment call index for this run
    RUN_CALL_COUNTERS[run_id] = RUN_CALL_COUNTERS.get(run_id, 0) + 1
    call_index = RUN_CALL_COUNTERS[run_id]

    headers = dict(request.headers)
    # Strip proxy-specific headers before forwarding
    headers.pop("host", None)
    headers.pop("x-blackbox-run-id", None)

    body_bytes = await request.body()
    try:
        body_json = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
    except Exception:
        body_json = {"raw_bytes": body_bytes.decode("utf-8", errors="ignore")}

    target_url = f"{target_base.rstrip('/')}/{path.lstrip('/')}"
    is_streaming = body_json.get("stream", False)

    async with httpx.AsyncClient(timeout=120.0) as client:
        if not is_streaming:
            upstream_resp = await client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body_bytes,
                params=request.query_params
            )
            duration_ms = (time.time() - start_time) * 1000

            resp_headers = dict(upstream_resp.headers)
            resp_body_text = upstream_resp.text

            # Record call asynchronously
            call_create = RawCallCreate(
                run_id=run_id,
                call_index=call_index,
                provider=provider,
                endpoint=path,
                request_headers=headers,
                request_body=body_json,
                response_status=upstream_resp.status_code,
                response_headers=resp_headers,
                response_body_text=resp_body_text,
                is_streaming=False,
                duration_ms=duration_ms
            )
            await TraceRecorder.record_call(call_create)

            return Response(
                content=upstream_resp.content,
                status_code=upstream_resp.status_code,
                headers=resp_headers
            )
        else:
            # Streaming response proxying
            req = client.build_request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body_bytes,
                params=request.query_params
            )
            upstream_resp = await client.send(req, stream=True)
            accumulated_chunks = []

            async def stream_generator() -> AsyncGenerator[bytes, None]:
                nonlocal accumulated_chunks
                try:
                    async for chunk in upstream_resp.aiter_bytes():
                        accumulated_chunks.append(chunk.decode("utf-8", errors="ignore"))
                        yield chunk
                finally:
                    duration_ms = (time.time() - start_time) * 1000
                    full_resp_text = "".join(accumulated_chunks)
                    call_create = RawCallCreate(
                        run_id=run_id,
                        call_index=call_index,
                        provider=provider,
                        endpoint=path,
                        request_headers=headers,
                        request_body=body_json,
                        response_status=upstream_resp.status_code,
                        response_headers=dict(upstream_resp.headers),
                        response_body_text=full_resp_text,
                        is_streaming=True,
                        duration_ms=duration_ms
                    )
                    await TraceRecorder.record_call(call_create)

            return StreamingResponse(
                stream_generator(),
                status_code=upstream_resp.status_code,
                headers=dict(upstream_resp.headers)
            )


# Catch-all endpoint for OpenAI-compatible endpoints (/v1/chat/completions, /v1/embeddings, etc.)
@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_openai(request: Request, path: str):
    target_base = request.headers.get("x-target-base", DEFAULT_UPSTREAMS["openai"])
    return await forward_and_record(request, "openai", target_base, f"v1/{path}")


# Google Gemini endpoint proxy
@app.api_route("/google/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_google(request: Request, path: str):
    target_base = request.headers.get("x-target-base", DEFAULT_UPSTREAMS["google"])
    return await forward_and_record(request, "google", target_base, path)


# Anthropic endpoint proxy
@app.api_route("/anthropic/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_anthropic(request: Request, path: str):
    target_base = request.headers.get("x-target-base", DEFAULT_UPSTREAMS["anthropic"])
    return await forward_and_record(request, "anthropic", target_base, path)
