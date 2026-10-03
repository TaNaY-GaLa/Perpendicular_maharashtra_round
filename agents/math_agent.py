"""
Benchmark Agent 1: Math Calculator Agent (Interactive)
Specialty: Tool-use reasoning with multi-turn calculator tool integration.
"""

import asyncio
import json
import uuid
import sys
from typing import Dict, Any, List
from agents.common import call_gemini_via_blackbox


def calculate(expression: str) -> str:
    """Safely evaluates basic arithmetic expressions."""
    try:
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


async def run_math_agent():
    print("\n" + "=" * 65)
    print("  🧮 Math Calculator Agent (Interactive Mode)")
    print("  Specialty: Decides when and how to call external calculation tools")
    print("=" * 65)

    user_query = input("\nEnter your math question (or type 'exit' to quit): ").strip()
    if not user_query or user_query.lower() == "exit":
        return

    run_id = f"math_run_{uuid.uuid4().hex[:8]}"
    print(f"\n🚀 Execution Started | Run ID: {run_id}")
    print(f"❓ Prompt: {user_query}\n")

    messages = [
        {
            "role": "system",
            "content": "You are a precise mathematical assistant. When any arithmetic calculation is needed, you MUST invoke the `calculate` tool to compute it accurately.",
        },
        {"role": "user", "content": user_query},
    ]

    max_turns = 6
    for turn in range(max_turns):
        print(f"🔄 Turn {turn + 1}: Contacting Gemini via Black Box...")
        response = await call_gemini_via_blackbox(
            messages=messages,
            run_id=run_id,
            agent_name="math_agent",
            tools=TOOLS,
        )

        choice = response["choices"][0]
        message = choice["message"]
        tool_calls = message.get("tool_calls")
        messages.append(message)

        if tool_calls:
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except Exception:
                    args = {"expression": tc["function"]["arguments"]}

                print(f"  ⚡ Agent Action: Decided to invoke tool `{func_name}`")
                print(f"  📥 Tool Arguments: {args}")

                if func_name == "calculate":
                    result = calculate(args.get("expression", ""))
                    print(f"  📤 Tool Output: {result}\n")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })
        else:
            final_text = message.get("content", "")
            print("\n" + "─" * 65)
            print("🏁 Final Agent Output:")
            print("─" * 65)
            print(f"{final_text}\n")
            print("─" * 65)
            break

    print(f"✓ Recorded in Black Box.")
    print(f"🔍 Inspect structured steps: http://127.0.0.1:8000/api/runs/{run_id}/steps\n")


if __name__ == "__main__":
    asyncio.run(run_math_agent())
