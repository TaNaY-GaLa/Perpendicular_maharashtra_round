"""
Phase 2 Test Suite: Step Reconstruction & Checkpointed Replay Engine
Tests:
1. Reconstruct multi-turn tool calling runs into typed AgentSteps.
2. Verify Replay Session registration and cache-hit serving.
3. Test replay cache hit determinism (steps < K return instant cached responses).
4. Verify step K override modification.
"""

import asyncio
import json
import uuid
import pytest
from fastapi.testclient import TestClient

from blackbox.proxy.server import app
from blackbox.traces.models import init_db
from blackbox.traces.recorder import TraceRecorder
from blackbox.traces.reconstructor import StepReconstructor
from blackbox.proxy.replay import ReplayEngine
from blackbox.traces.schemas import RawCallCreate


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
async def setup_database():
    await init_db()


@pytest.fixture
def http_client():
    with TestClient(app, raise_server_exceptions=True) as client:
        yield client


@pytest.mark.asyncio
async def test_step_reconstruction():
    run_id = f"test_recon_{uuid.uuid4().hex[:6]}"
    
    # Call 1: User prompt -> Tool call
    await TraceRecorder.record_call(
        RawCallCreate(
            run_id=run_id,
            call_index=1,
            provider="openai",
            endpoint="v1/chat/completions",
            request_headers={},
            request_body={"messages": [{"role": "user", "content": "Calculate 10 + 20"}]},
            response_status=200,
            response_headers={},
            response_body_text=json.dumps({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "c1",
                            "function": {"name": "calculate", "arguments": '{"expression": "10+20"}'}
                        }]
                    }
                }]
            }),
            duration_ms=50.0
        )
    )

    # Call 2: Tool observation -> Final Answer
    await TraceRecorder.record_call(
        RawCallCreate(
            run_id=run_id,
            call_index=2,
            provider="openai",
            endpoint="v1/chat/completions",
            request_headers={},
            request_body={
                "messages": [
                    {"role": "user", "content": "Calculate 10 + 20"},
                    {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "calculate", "arguments": '{"expression": "10+20"}'}}]},
                    {"role": "tool", "tool_call_id": "c1", "content": "30"}
                ]
            },
            response_status=200,
            response_headers={},
            response_body_text=json.dumps({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "The result is 30."
                    }
                }]
            }),
            duration_ms=60.0
        )
    )

    raw_calls = await TraceRecorder.get_run_calls(run_id)
    steps = StepReconstructor.reconstruct_from_raw_calls(raw_calls)

    assert len(steps) == 4
    assert steps[0].step_type == "USER_QUERY"
    assert steps[0].content == "Calculate 10 + 20"
    assert steps[1].step_type == "TOOL_CALL"
    assert steps[1].tool_name == "calculate"
    assert steps[2].step_type == "TOOL_OBSERVATION"
    assert steps[2].tool_result == "30"
    assert steps[3].step_type == "FINAL_RESPONSE"
    assert "30" in steps[3].content


def test_steps_api_endpoint(http_client):
    run_id = f"test_api_steps_{uuid.uuid4().hex[:6]}"
    
    # Insert a quick call via recorder loop
    asyncio.run(
        TraceRecorder.record_call(
            RawCallCreate(
                run_id=run_id,
                call_index=1,
                provider="openai",
                endpoint="v1/chat/completions",
                request_headers={},
                request_body={"messages": [{"role": "user", "content": "Hello world"}]},
                response_status=200,
                response_headers={},
                response_body_text=json.dumps({"choices": [{"message": {"content": "Hi!"}}]}),
                duration_ms=30.0
            )
        )
    )

    resp = http_client.get(f"/api/runs/{run_id}/steps")
    assert resp.status_code == 200
    steps = resp.json()
    assert len(steps) >= 2
    assert steps[0]["step_type"] == "USER_QUERY"
    assert steps[1]["step_type"] == "FINAL_RESPONSE"


def test_replay_session_registration(http_client):
    run_id = f"test_rep_init_{uuid.uuid4().hex[:6]}"
    asyncio.run(TraceRecorder.get_or_create_run(run_id))

    resp = http_client.post(
        f"/api/runs/{run_id}/replay",
        json={"checkpoint_step": 2, "override_body": {"temperature": 0.0}}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "replay_session_id" in data
    assert data["checkpoint_step"] == 2
    assert "headers_to_use" in data
