"""
Benchmark Dataset Generator & Labeled Trace Synthesizer.
Generates 100+ balanced, labeled traces across 3 distinct agents (Math, SQL, Study Assistant):
- 50% SUCCESS runs (normal, unperturbed agent executions)
- 50% FAILURE runs (fault injected at random intermediate step K with known ground-truth)

Exports dataset to:
1. `blackbox_traces.db` (SQLite database with full RawCalls and AgentRuns)
2. `experiments/benchmark_dataset.jsonl` (Standard ML training & evaluation format)
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import uuid
from typing import Any, Dict, List

from blackbox.faults.injector import FaultInjector
from blackbox.traces.models import init_db
from blackbox.traces.recorder import TraceRecorder
from blackbox.traces.reconstructor import StepReconstructor
from blackbox.traces.schemas import RawCallCreate

OUTPUT_DATASET_FILE = "experiments/benchmark_dataset.jsonl"


async def generate_synthetic_benchmark(total_runs: int = 100):
    await init_db()
    os.makedirs("experiments", exist_ok=True)

    print("\n" + "=" * 70)
    print(f"  [+] Black Box Phase 3: Benchmark Dataset Generator")
    print(f"  Target: Generating {total_runs} balanced agent runs (Success vs Labeled Failures)")
    print("=" * 70 + "\n")

    agents = [
        {"name": "math_agent", "tools": ["calculate"]},
        {"name": "sql_agent", "tools": ["run_sql_query"]},
        {"name": "student_study_agent", "tools": ["search_study_notes", "generate_flashcards", "generate_practice_quiz", "create_study_schedule"]},
    ]

    fault_types = FaultInjector.FAULT_TYPES

    dataset_records: List[Dict[str, Any]] = []
    success_count = 0
    failure_count = 0

    for i in range(total_runs):
        run_id = f"bench_run_{uuid.uuid4().hex[:8]}"
        agent = random.choice(agents)
        agent_name = agent["name"]

        # 50% success, 50% failure
        is_failure = (i % 2 == 1)
        fault_type = random.choice(fault_types) if is_failure else None
        target_fault_step = random.randint(2, 3) if is_failure else None

        # Create AgentRun entry
        await TraceRecorder.get_or_create_run(run_id, agent_name=agent_name)

        # ── Turn 1: User prompt -> Assistant selects tool ──────────────────────
        tool_name = random.choice(agent["tools"])
        tool_call_id = f"call_{uuid.uuid4().hex[:6]}"
        initial_user_prompt = f"Perform multi-step task on {agent_name} with topic {tool_name}"

        c1 = RawCallCreate(
            run_id=run_id,
            call_index=1,
            provider="openai",
            endpoint="v1/chat/completions",
            request_headers={"content-type": "application/json"},
            request_body={
                "model": "gemini-2.5-flash",
                "messages": [{"role": "user", "content": initial_user_prompt}],
            },
            response_status=200,
            response_headers={"content-type": "application/json"},
            response_body_text=json.dumps({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tool_call_id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": json.dumps({"param": "val_1"})},
                        }]
                    }
                }]
            }),
            duration_ms=round(random.uniform(120.0, 350.0), 2),
        )
        await TraceRecorder.record_call(c1)

        # ── Turn 2: Tool observation -> Next Step ─────────────────────────────
        normal_observation = f"Successfully executed {tool_name}. Return data: 42 records processed."
        observed_content = normal_observation

        # Inject Fault at Step 2 for all failure runs
        injection_meta = None
        if is_failure:
            target_fault_step = 2
            msgs = [{"content": normal_observation}]
            mod_msgs, res = FaultInjector.inject_into_messages(msgs, 0, fault_type)
            observed_content = mod_msgs[0]["content"]
            injection_meta = res

        c2 = RawCallCreate(
            run_id=run_id,
            call_index=2,
            provider="openai",
            endpoint="v1/chat/completions",
            request_headers={"content-type": "application/json"},
            request_body={
                "model": "gemini-2.5-flash",
                "messages": [
                    {"role": "user", "content": initial_user_prompt},
                    {
                        "role": "assistant",
                        "tool_calls": [{"id": tool_call_id, "function": {"name": tool_name, "arguments": '{"param": "val_1"}'}}],
                    },
                    {"role": "tool", "tool_call_id": tool_call_id, "content": observed_content},
                ],
            },
            response_status=200,
            response_headers={"content-type": "application/json"},
            response_body_text=json.dumps({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": (
                            "Failure encountered: Unable to complete reasoning due to invalid intermediate data."
                            if (is_failure and target_fault_step == 2)
                            else f"Observation verified. Synthesizing final response for {agent_name}."
                        ),
                    }
                }]
            }),
            duration_ms=round(random.uniform(90.0, 280.0), 2),
        )
        await TraceRecorder.record_call(c2)

        # Update run status in DB
        status = "FAILURE" if is_failure else "SUCCESS"
        await TraceRecorder.update_run_status(run_id, status)

        if is_failure:
            failure_count += 1
        else:
            success_count += 1

        # Reconstruct high-level steps for dataset serialization
        raw_calls = await TraceRecorder.get_run_calls(run_id)
        steps = StepReconstructor.reconstruct_from_raw_calls(raw_calls)

        dataset_records.append({
            "run_id": run_id,
            "agent_name": agent_name,
            "status": status,
            "total_steps": len(steps),
            "ground_truth_failure": is_failure,
            "ground_truth_culprit_step": target_fault_step if is_failure else None,
            "fault_type": fault_type if is_failure else None,
            "fault_description": injection_meta.description if injection_meta else None,
            "steps": [s.model_dump() for s in steps],
        })

        if (i + 1) % 25 == 0 or (i + 1) == total_runs:
            print(f"  [>] Progress: {i + 1}/{total_runs} runs generated ({success_count} success, {failure_count} failures)")

    # Export to JSONL file
    with open(OUTPUT_DATASET_FILE, "w", encoding="utf-8") as f:
        for rec in dataset_records:
            f.write(json.dumps(rec) + "\n")

    print("\n" + "=" * 70)
    print(f"  [OK] Benchmark Dataset Successfully Generated!")
    print(f"  --> File Exported: {OUTPUT_DATASET_FILE}")
    print(f"  --> Summary: {total_runs} total runs ({success_count} SUCCESS, {failure_count} FAILURE)")
    print(f"  --> All failures labeled with exact ground-truth culprit step indices!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(generate_synthetic_benchmark(100))
