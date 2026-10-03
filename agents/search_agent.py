"""
Benchmark Agent 3: Search & Research Agent
Retrieves knowledge articles and synthesizes answers via Black Box.
"""

import asyncio
import json
import uuid
from typing import Dict, Any, List
from agents.common import call_gemini_via_blackbox

# Mock Knowledge Base Documents
KNOWLEDGE_BASE = {
    "black_box": "In aviation, a black box is an electronic recording device placed in an aircraft for the purpose of facilitating the investigation of aviation accidents and incidents.",
    "gemini": "Gemini is a multimodal AI model family created by Google DeepMind, capable of processing text, audio, images, and video.",
    "fastapi": "FastAPI is a modern, high-performance web framework for building APIs with Python based on standard type hints.",
}


def search_knowledge_base(topic: str) -> str:
    """Searches the knowledge base for a specific keyword or topic."""
    topic_clean = topic.lower().replace(" ", "_")
    for key, doc in KNOWLEDGE_BASE.items():
        if key in topic_clean or topic_clean in key:
            return json.dumps({"topic": key, "summary": doc})
    return json.dumps({"error": f"No documents found matching topic '{topic}'"})


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Searches internal knowledge articles for relevant information by topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Keyword or topic to search for, e.g. 'black_box', 'gemini', 'fastapi'",
                    }
                },
                "required": ["topic"],
            },
        },
    }
]


async def run_search_agent(query: str, run_id: str = None, replay_session_id: str = None):
    run_id = run_id or f"search_run_{uuid.uuid4().hex[:8]}"
    print(f"\n🚀 Running Search Agent (Run ID: {run_id})")
    print(f"❓ User Inquiry: {query}\n")

    messages = [
        {"role": "system", "content": "You are a research analyst. Use search_knowledge_base to look up relevant topic info."},
        {"role": "user", "content": query},
    ]

    max_turns = 5
    for turn in range(max_turns):
        print(f"--- Turn {turn + 1} ---")
        response = await call_gemini_via_blackbox(
            messages=messages,
            run_id=run_id,
            agent_name="search_agent",
            tools=TOOLS,
            replay_session_id=replay_session_id,
        )

        choice = response["choices"][0]
        message = choice["message"]
        tool_calls = message.get("tool_calls")
        messages.append(message)

        if tool_calls:
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                args = json.loads(tc["function"]["arguments"])
                print(f"🔧 Tool Call: {func_name}({args})")

                if func_name == "search_knowledge_base":
                    result = search_knowledge_base(args.get("topic", ""))
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
    test_query = "What is a black box in aviation?"
    asyncio.run(run_search_agent(test_query))
