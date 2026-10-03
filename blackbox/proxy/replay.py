"""
Replay Session Manager and Cache Hit Evaluator.
Allows executions to be replayed deterministically from intermediate states:
- If current step < checkpoint_step: Serve cached response from DB (0ms latency, $0 cost).
- If current step == checkpoint_step: Apply user override / modification.
- If current step > checkpoint_step: Forward live to the LLM upstream.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from blackbox.traces.recorder import TraceRecorder
from blackbox.traces.models import RawCall


class ReplaySession(BaseModel):
    session_id: str
    original_run_id: str
    checkpoint_step: int = Field(..., ge=1, description="Step from which execution diverges live.")
    override_body: Optional[Dict[str, Any]] = Field(default=None, description="Optional override payload for step K.")
    created_at: str


class ReplayEngine:
    # In-memory registry of active replay sessions
    _active_sessions: Dict[str, ReplaySession] = {}

    @classmethod
    def register_replay_session(
        cls,
        session_id: str,
        original_run_id: str,
        checkpoint_step: int,
        override_body: Optional[Dict[str, Any]] = None,
    ) -> ReplaySession:
        from datetime import datetime
        session = ReplaySession(
            session_id=session_id,
            original_run_id=original_run_id,
            checkpoint_step=checkpoint_step,
            override_body=override_body,
            created_at=datetime.utcnow().isoformat(),
        )
        cls._active_sessions[session_id] = session
        return session

    @classmethod
    def get_session(cls, session_id: str) -> Optional[ReplaySession]:
        return cls._active_sessions.get(session_id)

    @classmethod
    async def match_cached_call(cls, session: ReplaySession, call_index: int) -> Optional[RawCall]:
        """
        If call_index < session.checkpoint_step, fetch and return the identical
        RawCall recorded in the original run.
        """
        if call_index < session.checkpoint_step:
            return await TraceRecorder.get_call_by_index(session.original_run_id, call_index)
        return None

    @classmethod
    def apply_override(cls, session: ReplaySession, call_index: int, request_body: Dict[str, Any]) -> Dict[str, Any]:
        """
        If call_index == session.checkpoint_step and override_body exists,
        patch the incoming request.
        """
        if call_index == session.checkpoint_step and session.override_body:
            # Shallow / deep merge override
            merged = request_body.copy()
            merged.update(session.override_body)
            return merged
        return request_body
