"""
Benchmark Agent 3: Search & Research Agent (Interactive)
Specialty: Multi-source research agent with knowledge retrieval, document indexing, and fact synthesis.
"""

import asyncio
import json
import uuid
from typing import Dict, Any, List
from agents.common import call_gemini_via_blackbox

# Knowledge Base of documents
KNOWLEDGE_BASE = {
    "black box": "In aviation, a flight recorder (commonly known as a black box) is an electronic recording device placed in an aircraft to record flight parameters and cockpit audio, facilitating incident investigation.",
    "gemini": "Gemini is a multimodal AI model family created by Google DeepMind, capable of processing and reasoning across text, code, audio, and images.",
    "fastapi": "FastAPI is a modern, high-performance web framework for building REST APIs with Python 3.8+ based on standard Python type hints.",
    "langgraph": "LangGraph is a library developed by LangChain for building stateful, multi-actor applications with LLMs, using cyclical graphs.",
    "sqlite": "SQLite is a C-language library that implements a small, fast, self-contained, high-reliability, full-featured SQL database engine.",
}


def search_knowledge_base(query: str) -> str:
    """Searches knowledge base documents for keywords and topics."""
    q = query.lower()
    matches = {}
    for topic, text in KNOWLEDGE_BASE.items():
        if topic in q or any(word in text.lower() for word in q.split() if len(word) > 3):
            matches[topic] = text

    if matches:
        return json.dumps({"matched_articles": len(matches), "articles": matches})
    return json.dumps({"matched_articles": 0, "message": f"No knowledge base documents found matching '{query}'"})


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Searches the internal knowledge base for relevant articles on technologies, concepts, and systems.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Topic or keyword string to search, e.g. 'black box', 'fastapi', 'gemini'",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


async def run_search_agent():
    print("\n" + "=" * 65)
    print("  🔎 Search & Research Agent (Interactive Mode)")
    print("  Specialty: Queries internal document store & synthesizes research summaries")
    print("  Available Topics: `black box`, `gemini`, `fastapi`, `langgraph`, `sqlite`")
    print("=" * 65)

    user_query = input("\nEnter your research question (or 'exit'): ").strip()
    if not user_query or user_query.lower() == "exit":
        return

    run_id = f"search_run_{uuid.uuid4().hex[:8]}"
    print(f"\n🚀 Execution Started | Run ID: {run_id}")
    print(f"❓ Prompt: {user_query}\n")

    messages = [
        {
            "role": "system",
            "content": "You are a research analyst assistant. Use the `search_knowledge_base` tool to look up facts from the internal document repository.",
        },
        {"role": "user", "content": user_query},
    ]

    max_turns = 6
    for turn in range(max_turns):
        print(f"🔄 Turn {turn + 1}: Contacting Gemini via Black Box...")
        response = await call_gemini_via_blackbox(
            messages=messages,
            run_id=run_id,
            agent_name="search_agent",
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
                    args = {"query": tc["function"]["arguments"]}

                print(f"  ⚡ Agent Action: Decided to search knowledge base with `{func_name}`")
                print(f"  📥 Search Query: {args.get('query')}")

                if func_name == "search_knowledge_base":
                    result = search_knowledge_base(args.get("query", ""))
                    print(f"  📤 Knowledge Docs Retrieved: {result}\n")
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
    asyncio.run(run_search_agent())
