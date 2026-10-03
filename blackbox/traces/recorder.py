"""
Trace recording and persistence logic using SQLAlchemy async sessions.
"""

import json
import uuid
from datetime import datetime
from typing import Optional, List
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from blackbox.traces.models import AsyncSessionLocal, AgentRun, RawCall
from blackbox.traces.schemas import RawCallCreate


class TraceRecorder:
    @staticmethod
    async def get_or_create_run(run_id: str, agent_name: str = "default_agent") -> AgentRun:
        async with AsyncSessionLocal() as session:
            stmt = select(AgentRun).where(AgentRun.id == run_id)
            result = await session.execute(stmt)
            run = result.scalar_one_or_none()
            if not run:
                run = AgentRun(
                    id=run_id,
                    agent_name=agent_name,
                    status="RUNNING",
                    started_at=datetime.utcnow()
                )
                session.add(run)
                await session.commit()
                await session.refresh(run)
            return run

    @staticmethod
    async def record_call(call_data: RawCallCreate) -> RawCall:
        await TraceRecorder.get_or_create_run(call_data.run_id)
        async with AsyncSessionLocal() as session:
            call_id = str(uuid.uuid4())
            request_hash = call_data.compute_hash()
            raw_call = RawCall(
                id=call_id,
                run_id=call_data.run_id,
                call_index=call_data.call_index,
                provider=call_data.provider,
                endpoint=call_data.endpoint,
                request_headers_json=json.dumps(call_data.request_headers),
                request_body_json=json.dumps(call_data.request_body),
                request_hash=request_hash,
                response_status=call_data.response_status,
                response_headers_json=json.dumps(call_data.response_headers),
                response_body_text=call_data.response_body_text,
                is_streaming=call_data.is_streaming,
                duration_ms=call_data.duration_ms,
                timestamp=datetime.utcnow()
            )
            session.add(raw_call)
            
            # Increment run total calls
            stmt = select(AgentRun).where(AgentRun.id == call_data.run_id)
            result = await session.execute(stmt)
            run = result.scalar_one_or_none()
            if run:
                run.total_calls += 1
            
            await session.commit()
            await session.refresh(raw_call)
            return raw_call

    @staticmethod
    async def update_run_status(run_id: str, status: str):
        async with AsyncSessionLocal() as session:
            stmt = select(AgentRun).where(AgentRun.id == run_id)
            result = await session.execute(stmt)
            run = result.scalar_one_or_none()
            if run:
                run.status = status
                run.completed_at = datetime.utcnow()
                await session.commit()

    @staticmethod
    async def get_run_calls(run_id: str) -> List[RawCall]:
        async with AsyncSessionLocal() as session:
            stmt = select(RawCall).where(RawCall.run_id == run_id).order_by(RawCall.call_index)
            result = await session.execute(stmt)
            return list(result.scalars().all())
