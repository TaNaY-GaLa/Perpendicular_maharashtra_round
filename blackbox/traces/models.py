"""
Database initialization and SQLAlchemy ORM models for Black Box trace storage.
"""

import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import String, DateTime, Text, Integer, Float, ForeignKey, Boolean
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

DATABASE_URL = "sqlite+aiosqlite:///blackbox_traces.db"

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    agent_name: Mapped[str] = mapped_column(String, default="default_agent")
    status: Mapped[str] = mapped_column(String, default="RUNNING")  # RUNNING, SUCCESS, FAILURE
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    total_calls: Mapped[int] = mapped_column(Integer, default=0)
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    raw_calls: Mapped[List["RawCall"]] = relationship("RawCall", back_populates="run", cascade="all, delete-orphan")


class RawCall(Base):
    __tablename__ = "raw_calls"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("agent_runs.id"))
    call_index: Mapped[int] = mapped_column(Integer)
    provider: Mapped[str] = mapped_column(String)  # openai, google, anthropic, generic
    endpoint: Mapped[str] = mapped_column(String)
    request_headers_json: Mapped[str] = mapped_column(Text)
    request_body_json: Mapped[str] = mapped_column(Text)
    request_hash: Mapped[str] = mapped_column(String, index=True)
    response_status: Mapped[int] = mapped_column(Integer)
    response_headers_json: Mapped[str] = mapped_column(Text)
    response_body_text: Mapped[str] = mapped_column(Text)
    is_streaming: Mapped[bool] = mapped_column(Boolean, default=False)
    duration_ms: Mapped[float] = mapped_column(Float, default=0.0)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    run: Mapped["AgentRun"] = relationship("AgentRun", back_populates="raw_calls")


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
