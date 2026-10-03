"""
Unit tests for Phase 2 Step Reconstruction & Checkpointed Replay Engine.
"""

import pytest
import asyncio
from blackbox.traces.reconstructor import StepReconstructor
from blackbox.proxy.replay import ReplayEngine
from blackbox.traces.recorder import TraceRecorder
from blackbox.traces.schemas import RawCallCreate
from blackbox.traces.models import init_db


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.mark.asyncio
async def test_step_reconstruction_and_replay():
    await init_db()
    run_id = "replay_test_run"
    
    # 1. Record call 1 (tool call choice)
    c1 = RawCallCreate(
        run_id=run_id,
        call_index=1,
        provider="openai",
        endpoint="v1/chat/completions",
        request_headers={},
        request_body={"messages": [{"role": "user", "content": "What is 2+2?"}]},
        response_status=200,
        response_headers={},
        response_body_text=json_str({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {"name": "calculator", "arguments": '{"expr": "2+2"}'}
                    }]
                }
            }]
        })
    )
    await TraceRecorder.record_call(c1)

    # 2. Record call 2 (final answer)
    c2 = RawCallCreate(
        run_id=run_id,
        call_index=2,
        provider="openai",
        endpoint="v1/chat/completions",
        request_headers={},
        request_body={
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "tool_calls": [{"id": "call_1", "function": {"name": "calculator", "arguments": '{"expr": "2+2"}'}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "4"}
            ]
        },
        response_status=200,
        response_headers={},
        response_body_text=json_str({
            "choices": [{"message": {"role": "assistant", "content": "The result is 4."}}]
        })
    )
    await TraceRecorder.record_call(c2)

    # Test reconstructor
    raw_calls = await TraceRecorder.get_run_calls(run_id)
    steps = StepReconstructor.reconstruct_from_raw_calls(raw_calls)
    
    assert len(steps) >= 2
    assert any(s.step_type == "TOOL_CALL" and s.tool_name == "calculator" for s in steps)
    assert any(s.step_type == "TOOL_OBSERVATION" and s.content == "4" for s in steps)

    # Test replay engine caching
    replay = ReplayEngine(target_run_id=run_id, checkpoint_step_k=1)
    cached = await replay.get_cached_response(call_index=1)
    assert cached is not None
    assert cached.call_index == 1

    uncached = await replay.get_cached_response(call_index=2)
    assert uncached is None


def json_str(obj):
    import json
    return json.dumps(obj)
