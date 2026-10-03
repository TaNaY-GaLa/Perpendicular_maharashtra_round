"""
Phase 3 Test Suite: Fault Injection Engine & Benchmark Dataset Integrity
Tests:
1. Verify all 5 fault injection mutation strategies.
2. Verify ground-truth culprit step labeling.
3. Test benchmark dataset JSONL schema validity and balance.
"""

import os
import json
import pytest
from blackbox.faults.injector import FaultInjector


def test_fault_injector_types_completeness():
    assert len(FaultInjector.FAULT_TYPES) == 5
    expected = [
        "TOOL_OUTPUT_CORRUPTION",
        "CONTEXT_DROPPING",
        "HALLUCINATION_TRIGGER",
        "TOOL_ERROR_MASKING",
        "ACTION_SWAPPING",
    ]
    for exp in expected:
        assert exp in FaultInjector.FAULT_TYPES


def test_corrupt_tool_output():
    text = "Valid SQL result: 25 rows found."
    corrupted, desc = FaultInjector.corrupt_tool_output(text)
    assert corrupted != text
    assert len(desc) > 0


def test_drop_context():
    long_text = "This is a very long text containing critical exam rules and formulas."
    dropped, desc = FaultInjector.drop_context(long_text)
    assert "[CONTEXT_TRUNCATED" in dropped or dropped == "[EMPTY_CONTEXT_DROP]"


def test_trigger_hallucination():
    text = "User account balance: $500"
    hallucinated, desc = FaultInjector.trigger_hallucination(text)
    assert text in hallucinated
    assert len(hallucinated) > len(text)


def test_mask_tool_error():
    masked, desc = FaultInjector.mask_tool_error("Error: Timeout")
    parsed = json.loads(masked)
    assert parsed["status"] == "SUCCESS"
    assert parsed["data"] is None


def test_swap_action_arguments():
    args = {"expression": "2 + 2"}
    swapped, desc = FaultInjector.swap_action_arguments(args)
    assert swapped["expression"] != "2 + 2"


def test_inject_into_messages_step_labeling():
    msgs = [
        {"role": "user", "content": "Calculate 10 * 10"},
        {"role": "tool", "content": "100"},
    ]
    # Inject at step index 1 (second message, 1-based step = 2)
    new_msgs, res = FaultInjector.inject_into_messages(msgs, 1, "TOOL_OUTPUT_CORRUPTION")
    assert res.fault_injected is True
    assert res.target_step_index == 2
    assert res.fault_type == "TOOL_OUTPUT_CORRUPTION"
    assert new_msgs[1]["content"] != "100"


def test_benchmark_dataset_jsonl_validity():
    dataset_path = "experiments/benchmark_dataset.jsonl"
    assert os.path.exists(dataset_path), "Dataset file experiments/benchmark_dataset.jsonl must exist"

    with open(dataset_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) >= 50, f"Expected at least 50 traces, found {len(lines)}"

    records = [json.loads(line) for line in lines]
    successes = [r for r in records if r["status"] == "SUCCESS"]
    failures = [r for r in records if r["status"] == "FAILURE"]

    assert len(successes) > 0
    assert len(failures) > 0

    # Ensure every failure record has ground truth culprit step
    for fail in failures:
        assert fail["ground_truth_failure"] is True
        assert fail["ground_truth_culprit_step"] is not None
        assert fail["fault_type"] in FaultInjector.FAULT_TYPES
        assert fail["fault_description"] is not None
        assert len(fail["steps"]) > 0
