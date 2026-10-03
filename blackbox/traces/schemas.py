"""
Pydantic schemas for Black Box proxy requests and trace representations.
"""

import hashlib
import json
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, ConfigDict


class RawCallCreate(BaseModel):
    run_id: str
    call_index: int
    provider: str
    endpoint: str
    request_headers: Dict[str, str]
    request_body: Dict[str, Any]
    response_status: int
    response_headers: Dict[str, str]
    response_body_text: str
    is_streaming: bool = False
    duration_ms: float = 0.0

    def compute_hash(self) -> str:
        serialized = json.dumps(self.request_body, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class RawCallResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    run_id: str
    call_index: int
    provider: str
    endpoint: str
    request_hash: str
    response_status: int
    response_body_text: str
    is_streaming: bool
    duration_ms: float
