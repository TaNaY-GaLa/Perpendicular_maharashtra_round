"""
Black Box Universal Proxy Server
=================================
A FastAPI reverse proxy that intercepts LLM API calls from any AI agent,
records every request/response pair to SQLite, supports checkpointed replay,
and reconstructs structured agent execution trees.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Optional, Any, List

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from blackbox.config import get_settings
from blackbox.traces.models import init_db, close_db
from blackbox.traces.recorder import TraceRecorder
from blackbox.traces.reconstructor import StepReconstructor, AgentStep
from blackbox.proxy.replay import ReplayEngine, ReplaySession
from blackbox.traces.schemas import AgentRunOut, RawCallCreate, RawCallOut, RunStatusUpdate

# ── Logging ────────────────────────────────────────────────────────────────────
logger = logging.getLogger("blackbox.proxy")

# ── Settings ───────────────────────────────────────────────────────────────────
settings = get_settings()

# ── Custom headers consumed by the proxy ──────────────────────────────────────
HEADER_RUN_ID = "x-blackbox-run-id"
HEADER_AGENT = "x-blackbox-agent"
HEADER_TARGET_BASE = "x-target-base"
HEADER_REPLAY_SESSION = "x-blackbox-replay-session"

# Headers the proxy must remove before forwarding to upstream
PROXY_HEADERS = {HEADER_RUN_ID, HEADER_AGENT, HEADER_TARGET_BASE, HEADER_REPLAY_SESSION, "host"}


# ── In-process call-index counters ────────────────────────────────────────────
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
    title="Black Box Proxy & Trace Engine",
    description="Universal AI Agent Flight Recorder & Replay Engine",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Core forwarding & replay logic ─────────────────────────────────────────────

async def _forward_and_record(
    request: Request,
    provider: str,
    upstream_base: str,
    upstream_path: str,
) -> Response:
    """
    Forwards incoming request to `upstream_base/upstream_path` OR serves cached
    response if this call belongs to an active Checkpointed Replay session.
    """
    start_time = time.perf_counter()

    # ── Extract proxy-specific headers ────────────────────────────────────────
    run_id: str = request.headers.get(HEADER_RUN_ID) or str(uuid.uuid4())
    agent_name: str = request.headers.get(HEADER_AGENT, "unknown_agent")
    replay_session_id: Optional[str] = request.headers.get(HEADER_REPLAY_SESSION)

    # ── Build forwarding headers (strip proxy-specific ones) ──────────────────
    forward_headers: Dict[str, str] = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in PROXY_HEADERS
    }

    # ── Read request body ─────────────────────────────────────────────────────
    body_bytes: bytes = await request.body()
    try:
        body_json: dict = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
    except Exception:
        body_json = {"_raw": body_bytes.decode("utf-8", errors="replace")}

    call_index: int = await _next_call_index(run_id)

    # Ensure run exists in DB
    await TraceRecorder.get_or_create_run(run_id, agent_name=agent_name)

    # ── CHECKPOINTED REPLAY CHECK ─────────────────────────────────────────────
    if replay_session_id:
        session = ReplayEngine.get_session(replay_session_id)
        if session:
            # Check if this call should be served from cache
            cached_call = await ReplayEngine.match_cached_call(session, call_index)
            if cached_call:
                logger.info(
                    "⚡ [REPLAY CACHE HIT] session=%s call #%d served from cache (%s)",
                    replay_session_id, call_index, cached_call.id
                )
                duration_ms = (time.perf_counter() - start_time) * 1000
                # Record this replay call into DB
                await TraceRecorder.record_call(
                    RawCallCreate(
                        run_id=run_id,
                        call_index=call_index,
                        provider=cached_call.provider,
                        endpoint=cached_call.endpoint,
                        request_headers=dict(forward_headers),
                        request_body=body_json,
                        response_status=cached_call.response_status,
                        response_headers=json.loads(cached_call.response_headers_json or "{}"),
                        response_body_text=cached_call.response_body_text,
                        is_streaming=cached_call.is_streaming,
                        duration_ms=0.5,  # instantaneous replay
                    )
                )
                return Response(
                    content=cached_call.response_body_text.encode("utf-8"),
                    status_code=cached_call.response_status,
                    media_type="application/json",
                )

            # If call_index == checkpoint_step, apply override if specified
            body_json = ReplayEngine.apply_override(session, call_index, body_json)
            body_bytes = json.dumps(body_json).encode("utf-8")

    # ── LIVE UPSTREAM EXECUTION ────────────────────────────────────────────────
    is_streaming: bool = bool(body_json.get("stream", False))
    target_url = f"{upstream_base.rstrip('/')}/{upstream_path.lstrip('/')}"
    if request.query_params:
        target_url += f"?{request.query_params}"

    logger.info(
        "[%s] call #%d → %s %s (streaming=%s)",
        run_id, call_index, request.method, target_url, is_streaming,
    )

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

        _excluded = {"transfer-encoding", "connection", "keep-alive", "content-encoding"}
        safe_headers = {
            k: v for k, v in upstream_resp.headers.items() if k.lower() not in _excluded
        }
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            headers=safe_headers,
        )

    # Streaming handling
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

    return StreamingResponse(_stream_and_record(), media_type="text/event-stream")


# ── Provider Routes ────────────────────────────────────────────────────────────

@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_openai_compatible(request: Request, path: str) -> Response:
    upstream = request.headers.get(HEADER_TARGET_BASE, settings.openai_base_url)
    return await _forward_and_record(request, "openai", upstream, f"v1/{path}")


@app.api_route("/google/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_google(request: Request, path: str) -> Response:
    upstream = request.headers.get(HEADER_TARGET_BASE, settings.google_base_url)
    return await _forward_and_record(request, "google", upstream, path)


@app.api_route("/anthropic/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_anthropic(request: Request, path: str) -> Response:
    upstream = request.headers.get(HEADER_TARGET_BASE, settings.anthropic_base_url)
    return await _forward_and_record(request, "anthropic", upstream, path)


@app.api_route("/proxy/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_generic(request: Request, path: str) -> Response:
    upstream = request.headers.get(HEADER_TARGET_BASE, settings.openai_base_url)
    return await _forward_and_record(request, "generic", upstream, path)


# ── Step Reconstruction & Replay APIs ─────────────────────────────────────────

@app.get("/api/runs/{run_id}/steps", response_model=List[AgentStep], tags=["steps"])
async def get_run_steps(run_id: str):
    """
    Reconstructs the high-level agent execution steps (USER_QUERY, TOOL_CALL, TOOL_OBSERVATION, FINAL_RESPONSE)
    from raw calls recorded for a run.
    """
    calls = await TraceRecorder.get_run_calls(run_id)
    if not calls:
        return JSONResponse({"error": "Run not found or has no calls"}, status_code=404)
    steps = StepReconstructor.reconstruct_from_raw_calls(calls)
    return steps


class CreateReplayRequest(BaseModel):
    checkpoint_step: int = Field(..., ge=1, description="Step to replay from (prior steps served from cache)")
    override_body: Optional[Dict[str, Any]] = Field(default=None, description="Optional override payload for step K")


@app.post("/api/runs/{run_id}/replay", tags=["replay"])
async def init_replay_session(run_id: str, req: CreateReplayRequest):
    """
    Initializes a Checkpointed Replay Session for an existing run.
    Returns session_id and headers to pass in subsequent agent calls.
    """
    run = await TraceRecorder.get_run(run_id)
    if not run:
        return JSONResponse({"error": "Run not found"}, status_code=404)

    session_id = f"replay_{uuid.uuid4().hex[:8]}"
    session = ReplayEngine.register_replay_session(
        session_id=session_id,
        original_run_id=run_id,
        checkpoint_step=req.checkpoint_step,
        override_body=req.override_body,
    )
    return {
        "replay_session_id": session_id,
        "original_run_id": run_id,
        "checkpoint_step": req.checkpoint_step,
        "headers_to_use": {
            HEADER_RUN_ID: f"{run_id}_fork_{session_id}",
            HEADER_REPLAY_SESSION: session_id,
        },
    }


# ── Management APIs ───────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
async def health_check() -> dict:
    return {"status": "ok", "service": "blackbox-proxy", "version": "1.1.0"}


@app.get("/api/runs", response_model=list[AgentRunOut], tags=["traces"])
async def list_runs(limit: int = 50, offset: int = 0):
    return await TraceRecorder.list_runs(limit=limit, offset=offset)


@app.get("/api/runs/{run_id}", tags=["traces"])
async def get_run(run_id: str):
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
    await TraceRecorder.update_run_status(run_id, body.status)
    return {"run_id": run_id, "status": body.status}


@app.get("/api/stats", tags=["meta"])
async def stats():
    total = await TraceRecorder.count_runs()
    return {"total_runs": total}
