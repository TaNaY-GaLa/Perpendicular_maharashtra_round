"""
Benchmark Agent 4: Autonomous Financial & Operations Analyst Agent
A multi-turn, multi-tool autonomous agent that runs entirely using the standard OpenAI client.
Notice: There is ZERO Black Box code or mentions in this file.
The agent uses whatever OPENAI_BASE_URL is set in the environment.
"""

import json
import os
import sqlite3
import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Initialize standard OpenAI client
# It automatically reads OPENAI_BASE_URL (points to proxy) and API key from environment
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY") or "default_key"
client = OpenAI(
    api_key=api_key,
    base_url=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
)

# ── IN-MEMORY OPERATIONS DATABASE ─────────────────────────────────────────────
db = sqlite3.connect(":memory:")
cur = db.cursor()
cur.execute("CREATE TABLE transactions (id INTEGER, date TEXT, department TEXT, amount REAL, description TEXT)")
cur.executemany("INSERT INTO transactions VALUES (?, ?, ?, ?, ?)", [
    (101, "2026-09-01", "Engineering", 14500.00, "Cloud Cluster Billing"),
    (102, "2026-09-05", "Marketing", 6200.00, "Q3 Campaign Ads"),
    (103, "2026-09-12", "Operations", 3100.00, "Office Maintenance"),
    (104, "2026-09-18", "Engineering", 8900.00, "GPU Server Lease"),
    (105, "2026-09-24", "Sales", 4400.00, "Client Dinners & Travel"),
    (106, "2026-09-28", "Marketing", 9800.00, "Conference Booth Registration"),
])
db.commit()


# ── TOOL DEFINITIONS (4 Specialized Tools) ────────────────────────────────────

def query_database(sql_query: str) -> str:
    """Tool 1: Executes SQL query on company financial transactions."""
    try:
        c = db.cursor()
        c.execute(sql_query)
        rows = c.fetchall()
        cols = [d[0] for d in c.description] if c.description else []
        return json.dumps({"columns": cols, "data": rows})
    except Exception as e:
        return json.dumps({"error": str(e)})


def currency_convert(amount: float, from_currency: str, to_currency: str) -> str:
    """Tool 2: Converts amounts across global currencies."""
    rates_to_usd = {"USD": 1.0, "EUR": 1.08, "GBP": 1.29, "INR": 0.012, "JPY": 0.0068}
    try:
        from_rate = rates_to_usd.get(from_currency.upper(), 1.0)
        to_rate = rates_to_usd.get(to_currency.upper(), 1.0)
        usd_val = amount * from_rate
        converted = usd_val / to_rate
        return json.dumps({
            "original": f"{amount} {from_currency}",
            "converted": round(converted, 2),
            "target_currency": to_currency
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def risk_assessment(department: str, total_spent: float) -> str:
    """Tool 3: Analyzes budget burn rate and assigns risk rating."""
    budget_caps = {"Engineering": 25000.0, "Marketing": 15000.0, "Operations": 5000.0, "Sales": 6000.0}
    cap = budget_caps.get(department, 10000.0)
    ratio = total_spent / cap
    if ratio > 0.9:
        rating = "CRITICAL: Near or exceeding budget limit"
    elif ratio > 0.7:
        rating = "MODERATE: High expenditure pace"
    else:
        rating = "HEALTHY: Well within budget allowance"
    return json.dumps({"department": department, "budget_limit": cap, "spent": total_spent, "utilization": f"{round(ratio*100, 1)}%", "risk_rating": rating})


def log_audit_record(summary: str, risk_flag: str) -> str:
    """Tool 4: Commits an official audit summary to compliance registry."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return json.dumps({"status": "AUDIT_RECORDED", "timestamp": timestamp, "summary": summary, "risk_flag": risk_flag})


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "Runs SQL on `transactions` table (id, date, department, amount, description).",
            "parameters": {
                "type": "object",
                "properties": {"sql_query": {"type": "string", "description": "SQL query to execute"}},
                "required": ["sql_query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "currency_convert",
            "description": "Converts currency between USD, EUR, GBP, INR, JPY.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "from_currency": {"type": "string"},
                    "to_currency": {"type": "string"}
                },
                "required": ["amount", "from_currency", "to_currency"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "risk_assessment",
            "description": "Evaluates departmental budget risk rating based on expenditure totals.",
            "parameters": {
                "type": "object",
                "properties": {
                    "department": {"type": "string"},
                    "total_spent": {"type": "number"}
                },
                "required": ["department", "total_spent"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "log_audit_record",
            "description": "Writes official compliance audit entry.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "risk_flag": {"type": "string"}
                },
                "required": ["summary", "risk_flag"]
            }
        }
    }
]


# ── EXECUTION LOOP ────────────────────────────────────────────────────────────

def run_complex_agent():
    print("\n" + "=" * 70)
    print("  💼 Autonomous Financial & Operations Analyst Agent")
    print("  Tools Available: SQL Database, Currency Converter, Risk Assessor, Audit Logger")
    print("=" * 70)

    prompt = input("\nEnter financial audit task (or press Enter for default): ").strip()
    if not prompt:
        prompt = (
            "1. Calculate the total expenditure for the 'Engineering' department.\n"
            "2. Convert that total from USD to EUR.\n"
            "3. Run a risk assessment on the Engineering spend.\n"
            "4. Log an audit record with the final evaluation."
        )

    print(f"\n🚀 Agent starting task:\n{prompt}\n")

    messages = [
        {"role": "system", "content": "You are a professional financial auditor. Use available tools step-by-step to complete the audit."},
        {"role": "user", "content": prompt}
    ]

    max_steps = 10
    for step in range(max_steps):
        print(f"--- Turn {step + 1} ---")
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=messages,
            tools=TOOL_DEFINITIONS,
        )

        choice = response.choices[0]
        message = choice.message
        messages.append(message)

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                fn_name = tc.function.name
                args = json.loads(tc.function.arguments)
                print(f"  🔧 Executing tool: `{fn_name}`")
                print(f"  📥 Arguments: {args}")

                if fn_name == "query_database":
                    out = query_database(args.get("sql_query", ""))
                elif fn_name == "currency_convert":
                    out = currency_convert(float(args.get("amount", 0)), args.get("from_currency", "USD"), args.get("to_currency", "EUR"))
                elif fn_name == "risk_assessment":
                    out = risk_assessment(args.get("department", ""), float(args.get("total_spent", 0)))
                elif fn_name == "log_audit_record":
                    out = log_audit_record(args.get("summary", ""), args.get("risk_flag", "NORMAL"))
                else:
                    out = json.dumps({"error": f"Unknown tool {fn_name}"})

                print(f"  📤 Tool Result: {out}\n")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": out
                })
        else:
            print("\n" + "=" * 70)
            print("🏁 Final Audit Report:")
            print("=" * 70)
            print(f"{message.content}\n")
            break


if __name__ == "__main__":
    run_complex_agent()
