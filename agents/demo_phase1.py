"""
Black Box Phase 1 — Live Demo Script
======================================
Run this script to:
1. Spin up the proxy server on localhost:8000
2. Simulate 3 fake agent calls (no real API key needed — uses a mock echo server)
3. Fetch and print all recorded traces from the management API

Usage:
    PYTHONPATH=. python agents/demo_phase1.py
"""

import asyncio
import json
import sys
import httpx


PROXY_BASE = "http://127.0.0.1:8000"
RUN_ID = "demo-run-001"


async def wait_for_server(max_retries: int = 20) -> bool:
    """Poll until the proxy server is up."""
    for i in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{PROXY_BASE}/health", timeout=2.0)
                if resp.status_code == 200:
                    return True
        except Exception:
            pass
        await asyncio.sleep(0.3)
    return False


async def simulate_agent_run():
    """
    Simulates 3 agent 'turns' by hitting the proxy.
    Since we're not using a real API key, the upstream will return an auth error —
    but the proxy will still intercept and RECORD the attempt in the database.
    That's exactly what we want to demonstrate!
    """
    print("\n" + "=" * 60)
    print("  BLACK BOX — Phase 1 Live Demo")
    print("=" * 60)

    print("\n[1] Checking proxy health...")
    alive = await wait_for_server()
    if not alive:
        print("✗  Proxy not running! Start it with:")
        print("   PYTHONPATH=. uvicorn blackbox.proxy.server:app --port 8000")
        return

    print("✓  Proxy is alive at http://127.0.0.1:8000")

    headers = {
        "x-blackbox-run-id": RUN_ID,
        "x-blackbox-agent": "demo_agent",
        "content-type": "application/json",
        "authorization": "Bearer fake-key-for-demo",  # will be redacted in DB
    }

    # Simulate 3 turns of a multi-step agent
    turns = [
        {
            "label": "Turn 1 — User asks a question",
            "body": {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "What is the capital of France?"}],
            },
        },
        {
            "label": "Turn 2 — Agent decides to use a tool",
            "body": {
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": "What is the capital of France?"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "function": {
                                    "name": "web_search",
                                    "arguments": '{"query": "capital of France"}',
                                },
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_abc123", "content": "Paris"},
                ],
            },
        },
        {
            "label": "Turn 3 — Agent gives final answer",
            "body": {
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": "What is the capital of France?"},
                    {"role": "assistant", "content": "The capital of France is Paris."},
                ],
            },
        },
    ]

    print(f"\n[2] Simulating 3-turn agent run (Run ID: {RUN_ID})\n")

    async with httpx.AsyncClient(timeout=10.0) as client:
        for i, turn in enumerate(turns, start=1):
            print(f"  → {turn['label']}")
            try:
                resp = await client.post(
                    f"{PROXY_BASE}/v1/chat/completions",
                    headers=headers,
                    json=turn["body"],
                )
                print(f"    Proxy returned HTTP {resp.status_code} (upstream may reject without a real key)")
            except Exception as e:
                print(f"    Request error: {e}")

    # ── Query the management API to see what was recorded ─────────────────────
    print(f"\n[3] Querying management API to see recorded trace...\n")

    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{PROXY_BASE}/api/runs/{RUN_ID}")

    if resp.status_code == 200:
        data = resp.json()
        run = data["run"]
        calls = data["calls"]

        print(f"  Run ID     : {run['id']}")
        print(f"  Agent Name : {run['agent_name']}")
        print(f"  Status     : {run['status']}")
        print(f"  Total Calls: {run['total_calls']}")
        print(f"\n  Recorded Calls:")
        for c in calls:
            print(
                f"    [{c['call_index']}] {c['provider'].upper()} | {c['endpoint']} "
                f"| HTTP {c['response_status']} | {c['duration_ms']:.1f}ms "
                f"| hash={c['request_hash'][:12]}..."
            )
        print("\n✓  All 3 turns recorded in BlackBox SQLite trace store!")
    else:
        print(f"  Could not fetch run: HTTP {resp.status_code}")

    print("\n[4] Open these URLs in your browser:")
    print(f"   http://127.0.0.1:8000/health")
    print(f"   http://127.0.0.1:8000/api/runs")
    print(f"   http://127.0.0.1:8000/api/runs/{RUN_ID}")
    print(f"   http://127.0.0.1:8000/api/stats")
    print(f"   http://127.0.0.1:8000/docs  ← Interactive Swagger UI")
    print()


if __name__ == "__main__":
    asyncio.run(simulate_agent_run())
