"""
Pydantic v2 schemas for Black Box trace data.

Keeps a strict contract between the proxy layer and the storage layer.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Helpers ────────────────────────────────────────────────────────────────────

def _canonical_json(obj: Dict[str, Any]) -> str:
    """
    Produces deterministic JSON for hashing.
    Sorts keys recursively and strips whitespace.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_request_hash(request_body: Dict[str, Any]) -> str:
    """SHA-256 fingerprint of a canonicalised request body."""
    return hashlib.sha256(_canonical_json(request_body).encode("utf-8")).hexdigest()


# ── Input Schemas (proxy → storage) ───────────────────────────────────────────

class RawCallCreate(BaseModel):
    """
    Data contract for recording a single intercepted LLM call.
    The proxy creates one of these per round-trip and passes it to TraceRecorder.
    """

    run_id: str = Field(..., description="Agent run identifier (from X-BlackBox-Run-ID header).")
    call_index: int = Field(..., ge=1, description="1-based sequential call position within the run.")
    provider: str = Field(..., description="LLM provider slug: openai | google | anthropic | generic.")
    endpoint: str = Field(..., description="API endpoint path, e.g. v1/chat/completions.")

    request_headers: Dict[str, str] = Field(default_factory=dict)
    request_body: Dict[str, Any] = Field(default_factory=dict)

    response_status: int = Field(..., description="HTTP status code returned by upstream.")
    response_headers: Dict[str, str] = Field(default_factory=dict)
    response_body_text: str = Field(default="", description="Full response body as text.")

    is_streaming: bool = Field(default=False)
    duration_ms: float = Field(default=0.0, ge=0.0)
    error_message: Optional[str] = Field(default=None)

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        allowed = {"openai", "google", "anthropic", "generic"}
        v = v.lower().strip()
        if v not in allowed:
            v = "generic"
        return v

    def request_hash(self) -> str:
        """SHA-256 hash of the canonical request body for cache lookup."""
        return compute_request_hash(self.request_body)

    def sanitised_headers(self) -> Dict[str, str]:
        """
        Returns headers safe for storage — strips sensitive auth values
        so we store the key name but not the bearer token.
        """
        sensitive = {"authorization", "x-api-key", "api-key", "x-goog-api-key"}
        return {
            k: ("[REDACTED]" if k.lower() in sensitive else v)
            for k, v in self.request_headers.items()
        }


# ── Output Schemas (storage → API consumers) ──────────────────────────────────

class RawCallOut(BaseModel):
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
    timestamp: datetime
    error_message: Optional[str]


class AgentRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    agent_name: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime]
    total_calls: int
    tags: Optional[str]


class AgentRunDetail(AgentRunOut):
    raw_calls: List[RawCallOut] = Field(default_factory=list)


# ── Control Schemas ────────────────────────────────────────────────────────────

class RunStatusUpdate(BaseModel):
    status: str = Field(..., pattern="^(SUCCESS|FAILURE|RUNNING)$")


class CreateRunRequest(BaseModel):
    run_id: Optional[str] = Field(default=None, description="If omitted, a UUID will be generated.")
    agent_name: str = Field(default="default_agent")
    tags: Optional[List[str]] = Field(default=None)
