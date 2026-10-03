"""
TraceRecorder — async persistence engine for Black Box.

Responsibilities
----------------
- Create / retrieve AgentRun rows
- Persist RawCall rows with computed hashes and sanitised headers
- Update run status (SUCCESS / FAILURE) on completion
- Provide query helpers for downstream modules (replay, diagnosis)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import func, select, update

from blackbox.traces.models import AgentRun, AsyncSessionLocal, RawCall
from blackbox.traces.schemas import RawCallCreate


class TraceRecorder:
    # ── Run management ─────────────────────────────────────────────────────────

    @staticmethod
    async def get_or_create_run(
        run_id: str,
        agent_name: str = "unknown_agent",
        tags: Optional[List[str]] = None,
    ) -> AgentRun:
        """
        Returns the existing AgentRun for run_id, or creates a new one.
        Safe to call multiple times for the same run_id (idempotent).
        """
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(AgentRun).where(AgentRun.id == run_id)
            )
            run = result.scalar_one_or_none()

            if run is None:
                run = AgentRun(
                    id=run_id,
                    agent_name=agent_name,
                    status="RUNNING",
                    started_at=datetime.utcnow(),
                    tags=json.dumps(tags) if tags else None,
                )
                session.add(run)
                await session.commit()
                await session.refresh(run)

            return run

    @staticmethod
    async def update_run_status(run_id: str, status: str) -> None:
        """
        Sets the status of an AgentRun to SUCCESS or FAILURE
        and records the completion timestamp.
        """
        async with AsyncSessionLocal() as session:
            await session.execute(
                update(AgentRun)
                .where(AgentRun.id == run_id)
                .values(status=status, completed_at=datetime.utcnow())
            )
            await session.commit()

    @staticmethod
    async def list_runs(limit: int = 50, offset: int = 0) -> List[AgentRun]:
        """Returns a paginated list of all agent runs, newest first."""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(AgentRun)
                .order_by(AgentRun.started_at.desc())
                .limit(limit)
                .offset(offset)
            )
            return list(result.scalars().all())

    @staticmethod
    async def get_run(run_id: str) -> Optional[AgentRun]:
        """Fetch a single AgentRun by ID."""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(AgentRun).where(AgentRun.id == run_id)
            )
            return result.scalar_one_or_none()

    # ── Call recording ─────────────────────────────────────────────────────────

    @staticmethod
    async def record_call(call_data: RawCallCreate) -> RawCall:
        """
        Persists a single intercepted LLM round-trip to the database.

        Steps:
        1. Ensure the parent AgentRun exists.
        2. Compute SHA-256 request hash.
        3. Sanitise authentication headers before storage.
        4. Write RawCall row and increment the run's total_calls counter.
        """
        # Ensure parent run exists (creates it silently if not)
        await TraceRecorder.get_or_create_run(call_data.run_id)

        async with AsyncSessionLocal() as session:
            raw_call = RawCall(
                id=str(uuid.uuid4()),
                run_id=call_data.run_id,
                call_index=call_data.call_index,
                provider=call_data.provider,
                endpoint=call_data.endpoint,
                request_headers_json=json.dumps(call_data.sanitised_headers()),
                request_body_json=json.dumps(call_data.request_body),
                request_hash=call_data.request_hash(),
                response_status=call_data.response_status,
                response_headers_json=json.dumps(call_data.response_headers),
                response_body_text=call_data.response_body_text,
                is_streaming=call_data.is_streaming,
                duration_ms=call_data.duration_ms,
                timestamp=datetime.utcnow(),
                error_message=call_data.error_message,
            )
            session.add(raw_call)

            # Atomic increment of total_calls
            await session.execute(
                update(AgentRun)
                .where(AgentRun.id == call_data.run_id)
                .values(total_calls=AgentRun.total_calls + 1)
            )

            await session.commit()
            await session.refresh(raw_call)
            return raw_call

    # ── Query helpers ──────────────────────────────────────────────────────────

    @staticmethod
    async def get_run_calls(run_id: str) -> List[RawCall]:
        """Returns all RawCalls for a run, ordered by call_index ascending."""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(RawCall)
                .where(RawCall.run_id == run_id)
                .order_by(RawCall.call_index)
            )
            return list(result.scalars().all())

    @staticmethod
    async def get_call_by_index(run_id: str, call_index: int) -> Optional[RawCall]:
        """Fetch a specific call by its index position within a run."""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(RawCall).where(
                    RawCall.run_id == run_id,
                    RawCall.call_index == call_index,
                )
            )
            return result.scalar_one_or_none()

    @staticmethod
    async def find_by_hash(request_hash: str) -> Optional[RawCall]:
        """
        Lookup a RawCall by its request hash.
        Used by the Replay Engine to serve cached responses.
        Returns the most recent match.
        """
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(RawCall)
                .where(RawCall.request_hash == request_hash)
                .order_by(RawCall.timestamp.desc())
                .limit(1)
            )
            return result.scalar_one_or_none()

    @staticmethod
    async def count_runs() -> int:
        """Total number of agent runs stored."""
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(func.count()).select_from(AgentRun))
            return result.scalar_one()
