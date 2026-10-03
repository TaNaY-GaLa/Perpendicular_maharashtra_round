"""
Dataset generator script for producing labeled synthetic traces with ground-truth fault markers.
"""

import asyncio
import json
import uuid
import random
from blackbox.traces.recorder import TraceRecorder
from blackbox.traces.schemas import RawCallCreate
from blackbox.faults.injector import FaultInjector
from blackbox.traces.models import init_db


async def generate_synthetic_dataset(num_runs: int = 20):
    await init_db()
    print(f"Generating {num_runs} synthetic agent runs (successful and failed)...")

    agents = ["math_agent", "sql_agent", "search_agent"]
    fault_types = FaultInjector.FAULT_TYPES

    for i in range(num_runs):
        run_id = f"syn_run_{uuid.uuid4().hex[:8]}"
        agent = random.choice(agents)
        is_failure = random.choice([True, False])
        fault_type = random.choice(fault_types) if is_failure else None
        target_fault_step = random.randint(1, 3) if is_failure else None

        await TraceRecorder.get_or_create_run(run_id, agent_name=agent)

        # Step 1: Initial query
        c1 = RawCallCreate(
            run_id=run_id,
            call_index=1,
            provider="openai",
            endpoint="v1/chat/completions",
            request_headers={},
            request_body={"messages": [{"role": "user", "content": f"Execute task for {agent}"}]},
            response_status=200,
            response_headers={},
            response_body_text=json.dumps({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call_1",
                            "function": {"name": "query_tool", "arguments": '{"param": "val"}'}
                        }]
                    }
                }]
            })
        )
        await TraceRecorder.record_call(c1)

        # Step 2: Tool observation (Inject fault if target_fault_step == 2)
        tool_content = "Valid query result: 100 rows"
        if is_failure and target_fault_step == 2:
            msgs, meta = FaultInjector.inject_fault("TOOL_OBSERVATION", [{"content": tool_content}], 0, fault_type)
            tool_content = msgs[0]["content"]

        c2 = RawCallCreate(
            run_id=run_id,
            call_index=2,
            provider="openai",
            endpoint="v1/chat/completions",
            request_headers={},
            request_body={
                "messages": [
                    {"role": "user", "content": f"Execute task for {agent}"},
                    {"role": "assistant", "tool_calls": [{"id": "call_1", "function": {"name": "query_tool", "arguments": '{"param": "val"}'}}]},
                    {"role": "tool", "tool_call_id": "call_1", "content": tool_content}
                ]
            },
            response_status=500 if is_failure else 200,
            response_headers={},
            response_body_text=json.dumps({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "Execution failed due to invalid data format." if is_failure else "Task completed successfully with 100 rows."
                    }
                }]
            })
        )
        await TraceRecorder.record_call(c2)

        status = "FAILURE" if is_failure else "SUCCESS"
        await TraceRecorder.update_run_status(run_id, status)

    print(f"Successfully generated {num_runs} synthetic runs in database.")


if __name__ == "__main__":
    asyncio.run(generate_synthetic_dataset(25))
