"""
Agent Step Data Models and Reconstructor Engine.
Parses multi-turn raw calls (OpenAI, Gemini REST, Anthropic) into typed, human-readable agent steps:
- USER_QUERY
- MODEL_THINKING
- TOOL_CALL
- TOOL_OBSERVATION
- FINAL_RESPONSE
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class AgentStep(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    step_index: int = Field(..., description="1-based step counter in the run timeline.")
    step_type: str = Field(..., description="USER_QUERY, MODEL_THINKING, TOOL_CALL, TOOL_OBSERVATION, FINAL_RESPONSE")
    sender: str = Field(..., description="user | assistant | tool | system")
    content: str = Field(default="")
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_call_id: Optional[str] = None
    tool_result: Optional[str] = None
    raw_call_index: int = Field(..., description="Corresponding RawCall index in SQLite DB")
    duration_ms: float = 0.0


class StepReconstructor:
    @staticmethod
    def reconstruct_from_raw_calls(raw_calls: List[Any]) -> List[AgentStep]:
        """
        Takes a list of RawCall database entities for a run and reconstructs
        the sequential AgentStep execution tree.
        Supports:
        1. OpenAI Chat Completions schema (used by OpenAI, Groq, Ollama, and Gemini OpenAI endpoints)
        2. Google Gemini native generateContent schema (contents -> parts -> functionCall/functionResponse)
        """
        steps: List[AgentStep] = []
        step_counter = 1

        for raw in raw_calls:
            # Parse request body
            try:
                req_body = json.loads(raw.request_body_json) if raw.request_body_json else {}
            except Exception:
                req_body = {}

            # Parse response body
            try:
                resp_body = json.loads(raw.response_body_text) if raw.response_body_text else {}
            except Exception:
                resp_body = {}

            duration = getattr(raw, "duration_ms", 0.0)
            call_idx = getattr(raw, "call_index", 1)
            provider = getattr(raw, "provider", "openai")

            # --- PARSE INCOMING USER/TOOL MESSAGES (From Request) ---
            messages = req_body.get("messages", [])
            for m in messages:
                role = m.get("role", "")
                
                # Check for User Query at start
                if role == "user" and step_counter == 1:
                    content_str = str(m.get("content", ""))
                    if not any(s.step_type == "USER_QUERY" for s in steps):
                        steps.append(
                            AgentStep(
                                step_index=step_counter,
                                step_type="USER_QUERY",
                                sender="user",
                                content=content_str,
                                raw_call_index=call_idx,
                            )
                        )
                        step_counter += 1

                # Check for Tool Observation
                elif role == "tool":
                    call_id = m.get("tool_call_id")
                    content_str = str(m.get("content", ""))
                    if not any(s.tool_call_id == call_id and s.step_type == "TOOL_OBSERVATION" for s in steps):
                        steps.append(
                            AgentStep(
                                step_index=step_counter,
                                step_type="TOOL_OBSERVATION",
                                sender="tool",
                                content=content_str,
                                tool_call_id=call_id,
                                tool_result=content_str,
                                raw_call_index=call_idx,
                            )
                        )
                        step_counter += 1

            # --- PARSE MODEL OUTPUT (From Response) ---
            # 1. OpenAI-style choices
            choices = resp_body.get("choices", [])
            if choices and isinstance(choices, list):
                msg = choices[0].get("message", {})
                content = msg.get("content", "") or ""
                tool_calls = msg.get("tool_calls", [])

                if tool_calls:
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        fname = func.get("name", "unknown_tool")
                        try:
                            fargs = json.loads(func.get("arguments", "{}"))
                        except Exception:
                            fargs = {"raw": func.get("arguments", "")}

                        steps.append(
                            AgentStep(
                                step_index=step_counter,
                                step_type="TOOL_CALL",
                                sender="assistant",
                                content=f"Invoked tool `{fname}`",
                                tool_name=fname,
                                tool_args=fargs,
                                tool_call_id=tc.get("id"),
                                raw_call_index=call_idx,
                                duration_ms=duration,
                            )
                        )
                        step_counter += 1

                elif content:
                    # Final response or intermediate thinking
                    is_last_call = (raw == raw_calls[-1])
                    steps.append(
                        AgentStep(
                            step_index=step_counter,
                            step_type="FINAL_RESPONSE" if is_last_call else "MODEL_THINKING",
                            sender="assistant",
                            content=content,
                            raw_call_index=call_idx,
                            duration_ms=duration,
                        )
                    )
                    step_counter += 1

            # 2. Native Google Gemini Candidates format
            candidates = resp_body.get("candidates", [])
            if candidates and isinstance(candidates, list):
                cand = candidates[0]
                content_obj = cand.get("content", {})
                parts = content_obj.get("parts", [])
                
                for part in parts:
                    if "functionCall" in part:
                        fc = part["functionCall"]
                        fname = fc.get("name", "unknown_tool")
                        fargs = fc.get("args", {})
                        steps.append(
                            AgentStep(
                                step_index=step_counter,
                                step_type="TOOL_CALL",
                                sender="assistant",
                                content=f"Invoked tool `{fname}`",
                                tool_name=fname,
                                tool_args=fargs,
                                tool_call_id=f"gemini_call_{step_counter}",
                                raw_call_index=call_idx,
                                duration_ms=duration,
                            )
                        )
                        step_counter += 1
                    elif "text" in part:
                        text_str = part["text"]
                        is_last_call = (raw == raw_calls[-1])
                        steps.append(
                            AgentStep(
                                step_index=step_counter,
                                step_type="FINAL_RESPONSE" if is_last_call else "MODEL_THINKING",
                                sender="assistant",
                                content=text_str,
                                raw_call_index=call_idx,
                                duration_ms=duration,
                            )
                        )
                        step_counter += 1

        return steps
