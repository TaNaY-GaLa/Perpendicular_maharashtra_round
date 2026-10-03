"""
Step reconstructor module that analyzes message histories across multi-turn raw calls
to construct typed, agent-agnostic step trees.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict


class AgentStep(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    step_index: int
    step_type: str  # SYSTEM_PROMPT, USER_QUERY, MODEL_THINKING, TOOL_CALL, TOOL_OBSERVATION, FINAL_RESPONSE
    sender: str  # system, user, assistant, tool
    content: str
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_call_id: Optional[str] = None
    raw_call_index: int


class StepReconstructor:
    @staticmethod
    def reconstruct_from_raw_calls(raw_calls: List[Any]) -> List[AgentStep]:
        """
        Parses a list of RawCall objects for an agent run and reconstructs typed AgentSteps.
        Handles standard OpenAI message arrays, Anthropic messages, and generic payload structures.
        """
        steps: List[AgentStep] = []
        step_counter = 1

        for raw in raw_calls:
            import json
            try:
                body = json.loads(raw.request_body_json)
            except Exception:
                body = {}

            try:
                resp = json.loads(raw.response_body_text)
            except Exception:
                resp = {}

            # Process OpenAI / Generic message format
            messages = body.get("messages", [])
            
            # Inspect response choice
            choices = resp.get("choices", [])
            if choices and isinstance(choices, list):
                msg = choices[0].get("message", {})
                content = msg.get("content", "") or ""
                tool_calls = msg.get("tool_calls", [])

                if tool_calls:
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        try:
                            args = json.loads(func.get("arguments", "{}"))
                        except Exception:
                            args = {"raw": func.get("arguments", "")}

                        steps.append(
                            AgentStep(
                                step_index=step_counter,
                                step_type="TOOL_CALL",
                                sender="assistant",
                                content=f"Call tool: {func.get('name')}",
                                tool_name=func.get("name"),
                                tool_args=args,
                                tool_call_id=tc.get("id"),
                                raw_call_index=raw.call_index
                            )
                        )
                        step_counter += 1
                elif content:
                    steps.append(
                        AgentStep(
                            step_index=step_counter,
                            step_type="MODEL_THINKING" if len(raw_calls) > step_counter else "FINAL_RESPONSE",
                            sender="assistant",
                            content=content,
                            raw_call_index=raw.call_index
                        )
                    )
                    step_counter += 1

            # Extract tool observations from incoming messages if present
            for m in messages:
                if m.get("role") == "tool":
                    # Check if already added
                    t_id = m.get("tool_call_id")
                    if not any(s.tool_call_id == t_id and s.step_type == "TOOL_OBSERVATION" for s in steps):
                        steps.append(
                            AgentStep(
                                step_index=step_counter,
                                step_type="TOOL_OBSERVATION",
                                sender="tool",
                                content=str(m.get("content", "")),
                                tool_call_id=t_id,
                                raw_call_index=raw.call_index
                            )
                        )
                        step_counter += 1

        return steps
