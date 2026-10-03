"""
Black Box Universal Proxy Server
=================================
A FastAPI reverse proxy that intercepts LLM API calls from any AI agent,
records every request/response pair to SQLite, and forwards the traffic
transparently to the real upstream provider.

Supported provider routes
--------------------------
/v1/...           → OpenAI / Groq / Ollama / LiteLLM / any OpenAI-compatible API
/google/...       → Google Gemini (generativelanguage.googleapis.com)
/anthropic/...    → Anthropic Claude (api.anthropic.com)
/proxy/...        → Generic catch-all; set X-Target-Base header to override upstream

Custom headers understood by the proxy (stripped before forwarding)
--------------------------------------------------------------------
X-BlackBox-Run-ID    : Associates this call with a named agent run (required for grouping)
X-BlackBox-Agent     : Human-readable agent name (optional, stored for filtering)
X-Target-Base        : Override the upstream base URL for this request
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from blackbox.config import get_settings
from blackbox.traces.models import init_db, close_db
from blackbox.traces.recorder import TraceRecorder
from blackbox.traces.schemas import AgentRunOut, RawCallCreate, RawCallOut, RunStatusUpdate

# ── Logging ────────────────────────────────────────────────────────────────────
logger = logging.getLogger("blackbox.proxy")

# ── Settings ───────────────────────────────────────────────────────────────────
settings = get_settings()

# ── Custom headers consumed by the proxy ──────────────────────────────────────
HEADER_RUN_ID = "x-blackbox-run-id"
HEADER_AGENT = "x-blackbox-agent"
HEADER_TARGET_BASE = "x-target-base"

# Headers the proxy must remove before forwarding to upstream
PROXY_HEADERS = {HEADER_RUN_ID, HEADER_AGENT, HEADER_TARGET_BASE, "host"}


# ── In-process call-index counters (resets on server restart) ─────────────────
_run_call_counters: Dict[str, int] = {}
_counter_lock = asyncio.Lock()


async def _next_call_index(run_id: str) -> int:
    async with _counter_lock:
        _run_call_counters[run_id] = _run_call_counters.get(run_id, 0) + 1
        return _run_call_counters[run_id]


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=settings.log_level.upper())
    logger.info("Black Box Proxy starting — initialising database …")
    await init_db()
    logger.info(
        "Proxy ready on %s:%d", settings.blackbox_proxy_host, settings.blackbox_proxy_port
    )
    yield
    logger.info("Black Box Proxy shutting down …")
    await close_db()


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Black Box Proxy",
    description="Universal AI Agent Flight Recorder — intercepts, records, and replays LLM calls.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Core forwarding logic ──────────────────────────────────────────────────────

async def _forward_and_record(
    request: Request,
    provider: str,
    upstream_base: str,
    upstream_path: str,
) -> Response:
    """
    Forwards an incoming request to `upstream_base/upstream_path`, records
    the full round-trip in SQLite, and returns the upstream response to the caller.

    Handles both regular (buffered) and streaming responses.
    """
    start_time = time.perf_counter()

    # ── Extract proxy-specific headers ────────────────────────────────────────
    run_id: str = request.headers.get(HEADER_RUN_ID) or str(uuid.uuid4())
    agent_name: str = request.headers.get(HEADER_AGENT, "unknown_agent")

    # ── Build forwarding headers (strip proxy-specific ones) ──────────────────
    forward_headers: Dict[str, str] = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in PROXY_HEADERS
    }

    # ── Read request body ─────────────────────────────────────────────────────
    body_bytes: bytes = await request.body()
    try:
        import json
        body_json: dict = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
    except Exception:
        body_json = {"_raw": body_bytes.decode("utf-8", errors="replace")}

    is_streaming: bool = bool(body_json.get("stream", False))
    call_index: int = await _next_call_index(run_id)

    # Ensure the run exists in DB
    await TraceRecorder.get_or_create_run(run_id, agent_name=agent_name)

    target_url = f"{upstream_base.rstrip('/')}/{upstream_path.lstrip('/')}"
    if request.query_params:
        target_url += f"?{request.query_params}"

    logger.info(
        "[%s] call #%d → %s %s (streaming=%s)",
        run_id, call_index, request.method, target_url, is_streaming,
    )

    # ── Non-streaming ──────────────────────────────────────────────────────────
    if not is_streaming:
        try:
            async with httpx.AsyncClient(timeout=settings.upstream_timeout_seconds) as client:
                upstream_resp = await client.request(
                    method=request.method,
                    url=target_url,
                    headers=forward_headers,
                    content=body_bytes,
                )
        except httpx.HTTPError as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            err_msg = str(exc)
            logger.warning("[%s] upstream error: %s", run_id, err_msg)

            await TraceRecorder.record_call(
                RawCallCreate(
                    run_id=run_id,
                    call_index=call_index,
                    provider=provider,
                    endpoint=upstream_path,
                    request_headers=dict(forward_headers),
                    request_body=body_json,
                    response_status=0,
                    response_headers={},
                    response_body_text="",
                    is_streaming=False,
                    duration_ms=duration_ms,
                    error_message=err_msg,
                )
            )
            return JSONResponse(
                {"error": "upstream_error", "detail": err_msg}, status_code=502
            )

        duration_ms = (time.perf_counter() - start_time) * 1000

        await TraceRecorder.record_call(
            RawCallCreate(
                run_id=run_id,
                call_index=call_index,
                provider=provider,
                endpoint=upstream_path,
                request_headers=dict(forward_headers),
                request_body=body_json,
                response_status=upstream_resp.status_code,
                response_headers=dict(upstream_resp.headers),
                response_body_text=upstream_resp.text,
                is_streaming=False,
                duration_ms=duration_ms,
            )
        )

        logger.info(
            "[%s] call #%d ← %d (%.1f ms)",
            run_id, call_index, upstream_resp.status_code, duration_ms,
        )

        # Rebuild response — strip hop-by-hop headers that cause issues
        _excluded = {"transfer-encoding", "connection", "keep-alive", "content-encoding"}
        safe_headers = {
            k: v
            for k, v in upstream_resp.headers.items()
            if k.lower() not in _excluded
        }
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            headers=safe_headers,
        )

    # ── Streaming ──────────────────────────────────────────────────────────────
    chunks: list[str] = []

    async def _stream_and_record() -> AsyncGenerator[bytes, None]:
        nonlocal chunks
        error_msg: Optional[str] = None

        try:
            async with httpx.AsyncClient(timeout=settings.upstream_timeout_seconds) as client:
                async with client.stream(
                    method=request.method,
                    url=target_url,
                    headers=forward_headers,
                    content=body_bytes,
                ) as upstream_resp:
                    status_code = upstream_resp.status_code
                    resp_headers = dict(upstream_resp.headers)

                    async for chunk in upstream_resp.aiter_bytes():
                        chunks.append(chunk.decode("utf-8", errors="replace"))
                        yield chunk

        except httpx.HTTPError as exc:
            error_msg = str(exc)
            logger.warning("[%s] streaming upstream error: %s", run_id, error_msg)
            status_code = 0
            resp_headers = {}

        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            full_body = "".join(chunks)

            await TraceRecorder.record_call(
                RawCallCreate(
                    run_id=run_id,
                    call_index=call_index,
                    provider=provider,
                    endpoint=upstream_path,
                    request_headers=dict(forward_headers),
                    request_body=body_json,
                    response_status=status_code,
                    response_headers=resp_headers,
                    response_body_text=full_body,
                    is_streaming=True,
                    duration_ms=duration_ms,
                    error_message=error_msg,
                )
            )
            logger.info(
                "[%s] call #%d ← %d streaming done (%.1f ms, %d bytes)",
                run_id, call_index, status_code, duration_ms, len(full_body),
            )

    return StreamingResponse(
        _stream_and_record(),
        media_type="text/event-stream",
    )


# ── Provider Routes ────────────────────────────────────────────────────────────

@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_openai_compatible(request: Request, path: str) -> Response:
    """
    Handles OpenAI-compatible endpoints used by:
    OpenAI, Groq, Ollama, LiteLLM, Mistral, DeepSeek, Perplexity, etc.
    """
    upstream = request.headers.get(HEADER_TARGET_BASE, settings.openai_base_url)
    return await _forward_and_record(request, "openai", upstream, f"v1/{path}")


@app.api_route("/google/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_google(request: Request, path: str) -> Response:
    """
    Handles Google Generative AI (Gemini) endpoints.
    e.g. /google/v1beta/models/gemini-2.5-flash:generateContent
    """
    upstream = request.headers.get(HEADER_TARGET_BASE, settings.google_base_url)
    return await _forward_and_record(request, "google", upstream, path)


@app.api_route("/anthropic/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_anthropic(request: Request, path: str) -> Response:
    """
    Handles Anthropic Claude endpoints.
    e.g. /anthropic/v1/messages
    """
    upstream = request.headers.get(HEADER_TARGET_BASE, settings.anthropic_base_url)
    return await _forward_and_record(request, "anthropic", upstream, path)


@app.api_route("/proxy/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_generic(request: Request, path: str) -> Response:
    """
    Generic catch-all proxy. Requires X-Target-Base header to specify upstream.
    """
    upstream = request.headers.get(HEADER_TARGET_BASE, settings.openai_base_url)
    return await _forward_and_record(request, "generic", upstream, path)


# ── Management & Observation API ──────────────────────────────────────────────

@app.get("/health", tags=["meta"])
async def health_check() -> dict:
    """Liveness probe."""
    return {
        "status": "ok",
        "service": "blackbox-proxy",
        "version": "1.0.0",
    }


@app.get("/api/runs", response_model=list[AgentRunOut], tags=["traces"])
async def list_runs(limit: int = 50, offset: int = 0):
    """List all recorded agent runs, newest first."""
    runs = await TraceRecorder.list_runs(limit=limit, offset=offset)
    return runs


@app.get("/api/runs/{run_id}", tags=["traces"])
async def get_run(run_id: str):
    """Get a single run and all its raw calls."""
    run = await TraceRecorder.get_run(run_id)
    if run is None:
        return JSONResponse({"error": "not_found"}, status_code=404)
    calls = await TraceRecorder.get_run_calls(run_id)
    return {
        "run": AgentRunOut.model_validate(run).model_dump(),
        "calls": [RawCallOut.model_validate(c).model_dump() for c in calls],
    }


@app.patch("/api/runs/{run_id}/status", tags=["traces"])
async def update_run_status(run_id: str, body: RunStatusUpdate):
    """Manually update the status of a run (SUCCESS / FAILURE / RUNNING)."""
    await TraceRecorder.update_run_status(run_id, body.status)
    return {"run_id": run_id, "status": body.status}


@app.get("/api/stats", tags=["meta"])
async def stats():
    """Quick summary statistics."""
    total = await TraceRecorder.count_runs()
    return {"total_runs": total}
