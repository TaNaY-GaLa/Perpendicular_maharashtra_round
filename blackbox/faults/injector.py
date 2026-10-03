"""
Fault Injection Engine for Black Box.
Implements 5 controlled mutation strategies to inject intermediate failures into agent execution runs.
"""

import json
import random
from typing import Dict, Any, Tuple


class FaultInjector:
    FAULT_TYPES = [
        "TOOL_OUTPUT_CORRUPTION",
        "CONTEXT_DROPPING",
        "HALLUCINATION_TRIGGER",
        "TOOL_ERROR_MASKING",
        "ACTION_SWAPPING"
    ]

    @staticmethod
    def inject_fault(
        step_type: str,
        messages: list,
        target_step_index: int,
        fault_type: str
    ) -> Tuple[list, Dict[str, Any]]:
        """
        Injects a specific fault into the message payload at target_step_index.
        Returns modified messages list and metadata dict.
        """
        modified_messages = [m.copy() for m in messages]
        metadata = {
            "fault_injected": True,
            "target_step_index": target_step_index,
            "fault_type": fault_type
        }

        if target_step_index >= len(modified_messages):
            return modified_messages, {"fault_injected": False}

        target_msg = modified_messages[target_step_index]

        if fault_type == "TOOL_OUTPUT_CORRUPTION":
            # Corrupt tool output to junk JSON or malformed content
            target_msg["content"] = "ERROR 500: Internal Database Failure {junk_data: null}"
        
        elif fault_type == "CONTEXT_DROPPING":
            # Truncate content or remove system instructions
            target_msg["content"] = target_msg.get("content", "")[:10] + "... [TRUNCATED]"

        elif fault_type == "HALLUCINATION_TRIGGER":
            # Inject false facts into previous observations
            target_msg["content"] = target_msg.get("content", "") + " [NOTE: User account balance is -$99,999]"

        elif fault_type == "TOOL_ERROR_MASKING":
            # Return success with empty/null payload
            target_msg["content"] = json.dumps({"status": "success", "data": None})

        elif fault_type == "ACTION_SWAPPING":
            # Modify arguments or function calls
            target_msg["content"] = "Swapped action: Executed delete_table instead of read_table."

        return modified_messages, metadata
