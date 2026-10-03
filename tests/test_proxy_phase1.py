"""
Unit and integration tests for Phase 1 Black Box Proxy Server & Database Persistence.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from blackbox.proxy.server import app
from blackbox.traces.recorder import TraceRecorder
from blackbox.traces.models import init_db


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


def test_health_check():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_trace_recorder_persistence():
    await init_db()
    run_id = "test_run_123"
    await TraceRecorder.get_or_create_run(run_id, agent_name="unit_test_agent")

    from blackbox.traces.schemas import RawCallCreate
    call_data = RawCallCreate(
        run_id=run_id,
        call_index=1,
        provider="openai",
        endpoint="v1/chat/completions",
        request_headers={"authorization": "Bearer fake_key"},
        request_body={"model": "gpt-4o", "messages": [{"role": "user", "content": "test"}]},
        response_status=200,
        response_headers={"content-type": "application/json"},
        response_body_text='{"choices": [{"message": {"content": "hello"}}]}',
        is_streaming=False,
        duration_ms=150.5
    )

    call = await TraceRecorder.record_call(call_data)
    assert call.run_id == run_id
    assert call.call_index == 1
    assert call.provider == "openai"

    calls = await TraceRecorder.get_run_calls(run_id)
    assert len(calls) == 1
    assert calls[0].request_hash == call_data.compute_hash()
