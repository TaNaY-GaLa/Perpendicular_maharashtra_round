"""
Replay engine supporting cached step replaying up to step K and live execution with modifications beyond step K.
"""

from typing import Dict, Any, Optional
import json
from sqlalchemy.future import select
from blackbox.traces.models import AsyncSessionLocal, RawCall


class ReplayEngine:
    def __init__(self, target_run_id: str, checkpoint_step_k: int, override_payload: Optional[Dict[str, Any]] = None):
        self.target_run_id = target_run_id
        self.checkpoint_step_k = checkpoint_step_k
        self.override_payload = override_payload

    async def get_cached_response(self, call_index: int) -> Optional[RawCall]:
        """
        If call_index <= checkpoint_step_k, return cached RawCall from DB.
        """
        if call_index <= self.checkpoint_step_k:
            async with AsyncSessionLocal() as session:
                stmt = select(RawCall).where(
                    RawCall.run_id == self.target_run_id,
                    RawCall.call_index == call_index
                )
                result = await session.execute(stmt)
                return result.scalar_one_or_none()
        return None

    def apply_override(self, request_body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies modified prompt/tool args override at the checkpoint step if provided.
        """
        if self.override_payload:
            request_body.update(self.override_payload)
        return request_body
