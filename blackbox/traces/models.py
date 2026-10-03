"""
SQLAlchemy async ORM models and database engine initialisation for Black Box trace storage.

Tables
------
agent_runs  — one row per agent execution session
raw_calls   — one row per intercepted LLM API call within a run
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    event,
)
from sqlalchemy.ext.asyncio import (
    AsyncAttrs,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from blackbox.config import get_settings

settings = get_settings()

# ── Engine & Session Factory ───────────────────────────────────────────────────
engine = create_async_engine(
    settings.database_url,
    echo=False,
    connect_args={"check_same_thread": False},  # required for SQLite
)

AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


# ── Declarative Base ───────────────────────────────────────────────────────────
class Base(AsyncAttrs, DeclarativeBase):
    pass


# ── Models ─────────────────────────────────────────────────────────────────────

class AgentRun(Base):
    """
    Represents one full agent execution session.

    Lifecycle: RUNNING → SUCCESS | FAILURE
    """

    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    agent_name: Mapped[str] = mapped_column(String(128), default="unknown_agent", nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), default="RUNNING", nullable=False
    )  # RUNNING | SUCCESS | FAILURE
    started_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    total_calls: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    tags: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # JSON array of string tags e.g. ["math", "tool_use"]
    metadata_json: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # arbitrary k/v metadata

    # ── Relationships ──────────────────────────────────────────────────────────
    raw_calls: Mapped[List["RawCall"]] = relationship(
        "RawCall",
        back_populates="run",
        cascade="all, delete-orphan",
        lazy="select",
        order_by="RawCall.call_index",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<AgentRun id={self.id!r} status={self.status!r} calls={self.total_calls}>"


class RawCall(Base):
    """
    One intercepted LLM API round-trip (request + response).

    The request_hash is a SHA-256 of the canonical request body JSON
    and is used by the Replay Engine to match cached responses.
    """

    __tablename__ = "raw_calls"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    call_index: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-based sequential

    # ── Provider & routing ─────────────────────────────────────────────────────
    provider: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # openai | google | anthropic | generic
    endpoint: Mapped[str] = mapped_column(String(256), nullable=False)

    # ── Raw Request ────────────────────────────────────────────────────────────
    request_headers_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    request_body_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # ── Raw Response ───────────────────────────────────────────────────────────
    response_status: Mapped[int] = mapped_column(Integer, nullable=False)
    response_headers_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    response_body_text: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # ── Metadata ───────────────────────────────────────────────────────────────
    is_streaming: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    duration_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # ── Relationships ──────────────────────────────────────────────────────────
    run: Mapped["AgentRun"] = relationship("AgentRun", back_populates="raw_calls")

    # ── Convenience helpers ────────────────────────────────────────────────────
    @property
    def request_body(self) -> dict:
        try:
            return json.loads(self.request_body_json)
        except Exception:
            return {}

    @property
    def response_body(self) -> dict:
        try:
            return json.loads(self.response_body_text)
        except Exception:
            return {}

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<RawCall run={self.run_id!r} idx={self.call_index} "
            f"provider={self.provider!r} status={self.response_status}>"
        )


# ── DB Initialisation ──────────────────────────────────────────────────────────

async def init_db() -> None:
    """Create all tables if they do not already exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Dispose of the async engine connection pool."""
    await engine.dispose()
