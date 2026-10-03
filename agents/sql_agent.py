"""
Benchmark Agent 2: SQL Database Agent (Interactive)
Specialty: Converts natural language questions into executable SQL queries, runs them against an actual SQLite database, and interprets results.
"""

import asyncio
import json
import sqlite3
import uuid
from typing import Dict, Any, List
from agents.common import call_gemini_via_blackbox

# In-memory product database
db_conn = sqlite3.connect(":memory:")
cursor = db_conn.cursor()
cursor.execute("CREATE TABLE products (id INTEGER, name TEXT, category TEXT, price REAL, stock INTEGER)")
cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?)", [
    (1, "Laptop Pro", "Electronics", 1299.99, 15),
    (2, "Wireless Mouse", "Electronics", 29.99, 85),
    (3, "Standing Desk", "Furniture", 450.00, 10),
    (4, "Ergonomic Chair", "Furniture", 299.50, 22),
    (5, "USB-C Cable", "Electronics", 12.50, 140),
    (6, "4K Monitor", "Electronics", 399.00, 18),
    (7, "Noise Cancelling Headphones", "Audio", 199.99, 45),
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
            "description": "Executes a SQL query on table `products` (id, name, category, price, stock).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL SELECT statement to execute against products table.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


async def run_sql_agent():
    print("\n" + "=" * 65)
    print("  🗄️ SQL Database Agent (Interactive Mode)")
    print("  Specialty: Translates natural language to SQL queries and executes them live")
    print("  Database Table: `products` (id, name, category, price, stock)")
    print("=" * 65)

    user_query = input("\nAsk a question about products/inventory (or 'exit'): ").strip()
    if not user_query or user_query.lower() == "exit":
        return

    run_id = f"sql_run_{uuid.uuid4().hex[:8]}"
    print(f"\n🚀 Execution Started | Run ID: {run_id}")
    print(f"❓ Prompt: {user_query}\n")

    messages = [
        {
            "role": "system",
            "content": "You are a database analyst. The database has one table: `products` with columns (id, name, category, price, stock). You MUST use the `run_sql_query` tool to execute queries.",
        },
        {"role": "user", "content": user_query},
    ]

    max_turns = 6
    for turn in range(max_turns):
        print(f"🔄 Turn {turn + 1}: Contacting Gemini via Black Box...")
        response = await call_gemini_via_blackbox(
            messages=messages,
            run_id=run_id,
            agent_name="sql_agent",
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

                print(f"  ⚡ Agent Action: Decided to query database using `{func_name}`")
                print(f"  📥 SQL Query Generated: {args.get('query')}")

                if func_name == "run_sql_query":
                    result = run_sql_query(args.get("query", ""))
                    print(f"  📤 DB Rows Returned: {result}\n")
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
    asyncio.run(run_sql_agent())
