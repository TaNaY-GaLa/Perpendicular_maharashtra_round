"""
Unit tests for Phase 3 Fault Injection & Synthetic Data Generation.
"""

import pytest
import asyncio
from blackbox.faults.injector import FaultInjector


def test_fault_injection_strategies():
    messages = [
        {"role": "user", "content": "Query DB"},
        {"role": "tool", "content": "User balance is 500"}
    ]

    for fault in FaultInjector.FAULT_TYPES:
        mod_msgs, meta = FaultInjector.inject_fault("TOOL_OBSERVATION", messages, 1, fault)
        assert meta["fault_injected"] is True
        assert meta["fault_type"] == fault
        assert mod_msgs[1]["content"] != "User balance is 500"
