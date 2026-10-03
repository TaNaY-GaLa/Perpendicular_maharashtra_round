"""
Phase 1 Test Suite — Black Box Proxy Recorder

Tests
-----
1.  Health check endpoint returns 200 ok
2.  SHA-256 request hash is deterministic and provider-agnostic
3.  RawCallCreate sanitises authentication headers before storage
4.  TraceRecorder.get_or_create_run is idempotent
5.  TraceRecorder.record_call writes correct data and increments call count
6.  TraceRecorder.get_call_by_index returns the right call
7.  TraceRecorder.find_by_hash returns the most recent matching call
8.  TraceRecorder.update_run_status sets completed_at correctly
9.  Management API /api/runs returns the stored run
10. Management API /api/runs/{id} returns run + calls detail
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from blackbox.traces.models import init_db
from blackbox.traces.recorder import TraceRecorder
from blackbox.traces.schemas import RawCallCreate, compute_request_hash

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def event_loop():
    """Single event loop shared across all session-scoped async fixtures."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
async def setup_database():
    """Initialise the SQLite schema once before any test runs."""
    await init_db()


@pytest.fixture
def http_client():
    """Synchronous TestClient for FastAPI endpoint tests."""
    from blackbox.proxy.server import app
    with TestClient(app, raise_server_exceptions=True) as client:
        yield client


@pytest.fixture
def unique_run_id() -> str:
    return f"test_run_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def sample_call_data(unique_run_id: str) -> RawCallCreate:
    return RawCallCreate(
        run_id=unique_run_id,
        call_index=1,
        provider="openai",
        endpoint="v1/chat/completions",
        request_headers={
            "authorization": "Bearer sk-supersecret",
            "content-type": "application/json",
        },
        request_body={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "What is 2 + 2?"}],
        },
        response_status=200,
        response_headers={"content-type": "application/json"},
        response_body_text=json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": "4"}}]}
        ),
        is_streaming=False,
        duration_ms=142.7,
    )


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_ok(self, http_client):
        resp = http_client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["service"] == "blackbox-proxy"


class TestRequestHashing:
    def test_hash_is_deterministic(self):
        body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}
        h1 = compute_request_hash(body)
        h2 = compute_request_hash(body)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_hash_differs_for_different_bodies(self):
        b1 = {"messages": [{"content": "hello"}]}
        b2 = {"messages": [{"content": "goodbye"}]}
        assert compute_request_hash(b1) != compute_request_hash(b2)

    def test_hash_ignores_key_ordering(self):
        b1 = {"a": 1, "b": 2}
        b2 = {"b": 2, "a": 1}
        assert compute_request_hash(b1) == compute_request_hash(b2)

    def test_call_create_hash_method(self, sample_call_data):
        expected = compute_request_hash(sample_call_data.request_body)
        assert sample_call_data.request_hash() == expected


class TestHeaderSanitisation:
    def test_auth_header_is_redacted(self, sample_call_data):
        safe = sample_call_data.sanitised_headers()
        assert safe.get("authorization") == "[REDACTED]"

    def test_non_auth_headers_are_preserved(self, sample_call_data):
        safe = sample_call_data.sanitised_headers()
        assert safe.get("content-type") == "application/json"

    def test_all_sensitive_keys_redacted(self):
        call = RawCallCreate(
            run_id="r1",
            call_index=1,
            provider="anthropic",
            endpoint="anthropic/v1/messages",
            request_headers={
                "x-api-key": "secret",
                "api-key": "secret",
                "x-goog-api-key": "secret",
                "user-agent": "python-httpx",
            },
            request_body={},
            response_status=200,
            response_headers={},
            response_body_text="",
        )
        safe = call.sanitised_headers()
        assert safe["x-api-key"] == "[REDACTED]"
        assert safe["api-key"] == "[REDACTED]"
        assert safe["x-goog-api-key"] == "[REDACTED]"
        assert safe["user-agent"] == "python-httpx"


class TestTraceRecorder:
    async def test_get_or_create_run_creates_new(self, unique_run_id):
        run = await TraceRecorder.get_or_create_run(unique_run_id, agent_name="test_agent")
        assert run.id == unique_run_id
        assert run.agent_name == "test_agent"
        assert run.status == "RUNNING"

    async def test_get_or_create_run_is_idempotent(self, unique_run_id):
        run1 = await TraceRecorder.get_or_create_run(unique_run_id)
        run2 = await TraceRecorder.get_or_create_run(unique_run_id)
        assert run1.id == run2.id
        # Idempotent: calling twice shouldn't create duplicate rows
        run = await TraceRecorder.get_run(unique_run_id)
        assert run is not None

    async def test_record_call_persists_data(self, sample_call_data):
        call = await TraceRecorder.record_call(sample_call_data)
        assert call.run_id == sample_call_data.run_id
        assert call.call_index == 1
        assert call.provider == "openai"
        assert call.response_status == 200
        assert call.duration_ms == pytest.approx(142.7, rel=0.01)
        # Auth header must be redacted in storage
        stored_headers = json.loads(call.request_headers_json)
        assert stored_headers.get("authorization") == "[REDACTED]"

    async def test_record_call_increments_total_calls(self, sample_call_data):
        run_id = sample_call_data.run_id
        run_before = await TraceRecorder.get_run(run_id)
        calls_before = run_before.total_calls if run_before else 0
        await TraceRecorder.record_call(sample_call_data)
        run_after = await TraceRecorder.get_run(run_id)
        assert run_after.total_calls == calls_before + 1

    async def test_get_call_by_index(self, sample_call_data):
        await TraceRecorder.record_call(sample_call_data)
        call = await TraceRecorder.get_call_by_index(sample_call_data.run_id, 1)
        assert call is not None
        assert call.call_index == 1

    async def test_find_by_hash_returns_match(self, sample_call_data):
        await TraceRecorder.record_call(sample_call_data)
        expected_hash = sample_call_data.request_hash()
        found = await TraceRecorder.find_by_hash(expected_hash)
        assert found is not None
        assert found.request_hash == expected_hash

    async def test_update_run_status(self, unique_run_id):
        await TraceRecorder.get_or_create_run(unique_run_id)
        await TraceRecorder.update_run_status(unique_run_id, "SUCCESS")
        run = await TraceRecorder.get_run(unique_run_id)
        assert run.status == "SUCCESS"
        assert run.completed_at is not None

    async def test_get_run_calls_ordered(self, unique_run_id):
        """Multiple calls should come back sorted by call_index."""
        for idx in [3, 1, 2]:
            data = RawCallCreate(
                run_id=unique_run_id,
                call_index=idx,
                provider="openai",
                endpoint="v1/chat/completions",
                request_headers={},
                request_body={"idx": idx},
                response_status=200,
                response_headers={},
                response_body_text="ok",
            )
            await TraceRecorder.record_call(data)
        calls = await TraceRecorder.get_run_calls(unique_run_id)
        indices = [c.call_index for c in calls]
        assert indices == sorted(indices)


class TestManagementAPI:
    async def test_list_runs(self, http_client, unique_run_id):
        await TraceRecorder.get_or_create_run(unique_run_id, agent_name="api_test_agent")
        resp = http_client.get("/api/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        ids = [r["id"] for r in data]
        assert unique_run_id in ids

    async def test_get_run_detail(self, http_client, sample_call_data):
        await TraceRecorder.record_call(sample_call_data)
        resp = http_client.get(f"/api/runs/{sample_call_data.run_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["run"]["id"] == sample_call_data.run_id
        assert len(body["calls"]) >= 1

    async def test_get_run_not_found(self, http_client):
        resp = http_client.get("/api/runs/nonexistent_run_xyz")
        assert resp.status_code == 404

    async def test_update_status_endpoint(self, http_client, unique_run_id):
        await TraceRecorder.get_or_create_run(unique_run_id)
        resp = http_client.patch(
            f"/api/runs/{unique_run_id}/status",
            json={"status": "FAILURE"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "FAILURE"

    def test_stats_endpoint(self, http_client):
        resp = http_client.get("/api/stats")
        assert resp.status_code == 200
        assert "total_runs" in resp.json()
