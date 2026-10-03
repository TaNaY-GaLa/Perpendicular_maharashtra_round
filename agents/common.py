"""
Interactive Benchmark Agent Helper.
Prompts for Google Gemini API key if not present in env, sets up proxy routing,
and provides utility functions for tool execution and multi-turn loops.
"""

import os
import getpass
import json
import httpx
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()

BLACKBOX_PROXY_URL = os.getenv("BLACKBOX_PROXY_URL", "http://127.0.0.1:8000")


def ensure_gemini_api_key() -> str:
    """
    Checks environment for GOOGLE_API_KEY / GEMINI_API_KEY.
    If missing, prompts user directly in the terminal to enter it.
    """
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\n" + "=" * 65)
        print("  🔑 Black Box Agent: Google Gemini API Key Setup")
        print("=" * 65)
        print("No API key detected in environment or .env file.")
        api_key = getpass.getpass("Please paste your Google Gemini API Key: ").strip()
        if not api_key:
            print("❌ No API key provided. Exiting.")
            exit(1)
        os.environ["GOOGLE_API_KEY"] = api_key
        print("✓ API Key set for this session.\n")
    return api_key


async def call_gemini_via_blackbox(
    messages: List[Dict[str, Any]],
    run_id: str,
    agent_name: str,
    tools: Optional[List[Dict[str, Any]]] = None,
    replay_session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calls Google Gemini models via Black Box Proxy OpenAI-compatible endpoint.
    Google's official OpenAI-compatible endpoint is:
    https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
    """
    api_key = ensure_gemini_api_key()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "x-blackbox-run-id": run_id,
        "x-blackbox-agent": agent_name,
        "x-target-base": "https://generativelanguage.googleapis.com/v1beta/openai",
    }
    if replay_session_id:
        headers["x-blackbox-replay-session"] = replay_session_id

    payload: Dict[str, Any] = {
        "model": "gemini-2.5-flash",
        "messages": messages,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{BLACKBOX_PROXY_URL}/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()
