"""
Benchmark Agent 1: Math Calculator Agent
Executes multi-turn mathematical workflows via tools, routed through Black Box.
"""

import asyncio
import json
import uuid
from typing import Dict, Any, List
from agents.common import call_gemini_via_blackbox


# Tool implementation
def calculate(expression: str) -> str:
    """Safely evaluates basic arithmetic expressions."""
    try:
        # Restricted safe eval for numbers and basic operators
        allowed = set("0123456789+-*/(). %")
        if not all(c in allowed for c in expression):
            return "Error: Unsupported characters in mathematical expression"
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Performs mathematical arithmetic calculation on an expression string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate, e.g. '(45 * 2) + 10'",
                    }
                },
                "required": ["expression"],
            },
        },
    }
]


async def run_math_agent(query: str, run_id: str = None, replay_session_id: str = None):
    run_id = run_id or f"math_run_{uuid.uuid4().hex[:8]}"
    print(f"\n🚀 Running Math Agent (Run ID: {run_id})")
    print(f"❓ Query: {query}\n")

    messages = [
        {"role": "system", "content": "You are a precise mathematical assistant. Always use the calculate tool for calculations."},
        {"role": "user", "content": query},
    ]

    max_turns = 5
    for turn in range(max_turns):
        print(f"--- Turn {turn + 1} ---")
        response = await call_gemini_via_blackbox(
            messages=messages,
            run_id=run_id,
            agent_name="math_agent",
            tools=TOOLS,
            replay_session_id=replay_session_id,
        )

        choice = response["choices"][0]
        message = choice["message"]
        tool_calls = message.get("tool_calls")

        # Append assistant turn to history
        messages.append(message)

        if tool_calls:
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                args = json.loads(tc["function"]["arguments"])
                print(f"🔧 Tool Call: {func_name}({args})")

                if func_name == "calculate":
                    result = calculate(args.get("expression", ""))
                    print(f"💡 Tool Result: {result}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })
        else:
            final_text = message.get("content", "")
            print(f"\n🏁 Final Agent Answer:\n{final_text}\n")
            break

    print(f"✓ Run complete. Inspect steps at: http://127.0.0.1:8000/api/runs/{run_id}/steps\n")


if __name__ == "__main__":
    test_query = "What is 154 * 28, and what is that result divided by 4?"
    asyncio.run(run_math_agent(test_query))
