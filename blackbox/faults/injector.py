"""
Fault Injection Engine for Black Box.
Implements 5 controlled mutation strategies to inject intermediate failures
into agent execution traces with precise ground-truth culprit labeling:

1. TOOL_OUTPUT_CORRUPTION  (malformed JSON, HTTP 500 error, syntax crash)
2. CONTEXT_DROPPING        (truncates prompt or tool observation)
3. HALLUCINATION_TRIGGER   (subtly injects false/contradictory facts into observation)
4. TOOL_ERROR_MASKING      (returns success status code with empty/null payload)
5. ACTION_SWAPPING         (mutates tool arguments/parameters)
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class FaultInjectionResult(BaseModel):
    fault_injected: bool
    fault_type: Optional[str] = None
    target_step_index: Optional[int] = None
    original_content: Optional[str] = None
    mutated_content: Optional[str] = None
    description: Optional[str] = None


class FaultInjector:
    FAULT_TYPES = [
        "TOOL_OUTPUT_CORRUPTION",
        "CONTEXT_DROPPING",
        "HALLUCINATION_TRIGGER",
        "TOOL_ERROR_MASKING",
        "ACTION_SWAPPING",
    ]

    @classmethod
    def corrupt_tool_output(cls, original_content: str) -> Tuple[str, str]:
        """Corrupts tool output with invalid JSON or server error text."""
        corruptions = [
            "HTTP 500 Internal Server Error: Database connection terminated unexpectedly.",
            '{"status": "error", "code": 502, "message": "Bad Gateway / Invalid upstream response"}',
            "{invalid_json: unquoted_string, missing_brace:",
            "Error: Maximum tool execution depth exceeded (SIGKILL).",
        ]
        chosen = random.choice(corruptions)
        desc = "Tool output was corrupted with a synthetic 500 crash / invalid JSON payload"
        return chosen, desc

    @classmethod
    def drop_context(cls, original_content: str) -> Tuple[str, str]:
        """Silently truncates content to simulate lost context or token-limit drops."""
        if len(original_content) > 15:
            truncated = original_content[:12] + "... [CONTEXT_TRUNCATED_DUE_TO_TOKEN_LIMIT]"
        else:
            truncated = "[EMPTY_CONTEXT_DROP]"
        desc = "Context was aggressively truncated, removing critical facts needed by the agent"
        return truncated, desc

    @classmethod
    def trigger_hallucination(cls, original_content: str) -> Tuple[str, str]:
        """Injects contradictory false facts into an observation."""
        false_facts = [
            " [NOTE: System reports all user account balances are now zero and all tables were dropped]",
            " [OVERRIDE: Target country capital was recently renamed to Atlantis]",
            " [ANOMALY: The mathematical result of this formula was inverted by -1000000]",
        ]
        injected = original_content + random.choice(false_facts)
        desc = "Injected subtle contradictory or false statements into the observation"
        return injected, desc

    @classmethod
    def mask_tool_error(cls, original_content: str) -> Tuple[str, str]:
        """Returns success status with empty/null payload to simulate silent failure."""
        masked = json.dumps({"status": "SUCCESS", "records_found": 0, "data": None, "columns": []})
        desc = "Tool masked an internal error by returning HTTP 200 OK with empty/null data"
        return masked, desc

    @classmethod
    def swap_action_arguments(cls, tool_args: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """Mutates tool arguments (e.g. changes an expression or SQL query)."""
        mutated = tool_args.copy()
        if "expression" in mutated:
            mutated["expression"] = f"({mutated['expression']}) * 0 - 99999"
        elif "sql_query" in mutated:
            mutated["sql_query"] = "SELECT * FROM nonexistent_table_xyz LIMIT 1;"
        elif "topic" in mutated:
            mutated["topic"] = "completely_unrelated_nonsense_topic_123"
        elif "query" in mutated:
            mutated["query"] = "nonexistent_query_term_xyz"
        else:
            mutated["invalid_param"] = "corrupted_arg_value"

        desc = f"Action arguments were mutated unexpectedly: {mutated}"
        return mutated, desc

    @classmethod
    def inject_into_messages(
        cls,
        messages: List[Dict[str, Any]],
        target_message_index: int,
        fault_type: str,
    ) -> Tuple[List[Dict[str, Any]], FaultInjectionResult]:
        """
        Takes a list of conversation messages and applies a targeted fault injection
        at `target_message_index`.
        """
        if fault_type not in cls.FAULT_TYPES:
            raise ValueError(f"Unknown fault type: {fault_type}. Allowed: {cls.FAULT_TYPES}")

        new_messages = [m.copy() for m in messages]
        if target_message_index < 0 or target_message_index >= len(new_messages):
            return new_messages, FaultInjectionResult(fault_injected=False)

        target_msg = new_messages[target_message_index]
        orig_content = str(target_msg.get("content", ""))

        if fault_type == "TOOL_OUTPUT_CORRUPTION":
            mutated, desc = cls.corrupt_tool_output(orig_content)
            target_msg["content"] = mutated

        elif fault_type == "CONTEXT_DROPPING":
            mutated, desc = cls.drop_context(orig_content)
            target_msg["content"] = mutated

        elif fault_type == "HALLUCINATION_TRIGGER":
            mutated, desc = cls.trigger_hallucination(orig_content)
            target_msg["content"] = mutated

        elif fault_type == "TOOL_ERROR_MASKING":
            mutated, desc = cls.mask_tool_error(orig_content)
            target_msg["content"] = mutated

        elif fault_type == "ACTION_SWAPPING":
            # Mutate tool call arguments if present
            tool_calls = target_msg.get("tool_calls", [])
            if tool_calls:
                tc = tool_calls[0].copy()
                func = tc.get("function", {}).copy()
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except Exception:
                    args = {}
                mutated_args, desc = cls.swap_action_arguments(args)
                func["arguments"] = json.dumps(mutated_args)
                tc["function"] = func
                target_msg["tool_calls"] = [tc]
                mutated = json.dumps(mutated_args)
            else:
                mutated, desc = cls.corrupt_tool_output(orig_content)
                target_msg["content"] = mutated

        return new_messages, FaultInjectionResult(
            fault_injected=True,
            fault_type=fault_type,
            target_step_index=target_message_index + 1,  # 1-based index
            original_content=orig_content,
            mutated_content=target_msg.get("content", ""),
            description=desc,
        )
