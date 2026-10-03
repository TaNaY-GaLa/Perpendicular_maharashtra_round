"""
Benchmark Agent 2: SQL Database Query Agent
Inspects SQLite schema, executes queries, and returns formatted data via Black Box.
"""

import asyncio
import json
import sqlite3
import uuid
from typing import Dict, Any, List
from agents.common import call_gemini_via_blackbox

# In-memory demo SQLite database
db_conn = sqlite3.connect(":memory:")
cursor = db_conn.cursor()
cursor.execute("CREATE TABLE products (id INTEGER, name TEXT, category TEXT, price REAL, stock INTEGER)")
cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?)", [
    (1, "Laptop Pro", "Electronics", 1299.99, 15),
    (2, "Wireless Mouse", "Electronics", 29.99, 85),
    (3, "Standing Desk", "Furniture", 450.00, 10),
    (4, "Ergonomic Chair", "Furniture", 299.50, 22),
    (5, "USB-C Cable", "Electronics", 12.50, 140),
])
db_conn.commit()


def run_sql_query(query: str) -> str:
    """Executes a SQL query against the in-memory products database."""
    try:
        cur = db_conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        col_names = [desc[0] for desc in cur.description] if cur.description else []
        return json.dumps({"columns": col_names, "rows": rows})
    except Exception as e:
        return json.dumps({"error": str(e)})


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_sql_query",
            "description": "Executes a SQL query on the `products` table (schema: id, name, category, price, stock).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL SELECT query to execute, e.g. 'SELECT name, price FROM products WHERE category = \"Furniture\"'",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


async def run_sql_agent(query: str, run_id: str = None, replay_session_id: str = None):
    run_id = run_id or f"sql_run_{uuid.uuid4().hex[:8]}"
    print(f"\n🚀 Running SQL Agent (Run ID: {run_id})")
    print(f"❓ User Request: {query}\n")

    messages = [
        {"role": "system", "content": "You are a database analyst assistant. Use the run_sql_query tool to query the products table."},
        {"role": "user", "content": query},
    ]

    max_turns = 5
    for turn in range(max_turns):
        print(f"--- Turn {turn + 1} ---")
        response = await call_gemini_via_blackbox(
            messages=messages,
            run_id=run_id,
            agent_name="sql_agent",
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

                if func_name == "run_sql_query":
                    result = run_sql_query(args.get("query", ""))
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
    test_query = "What is the most expensive furniture item, and how many are currently in stock?"
    asyncio.run(run_sql_agent(test_query))
