"""
Black Box SDK auto-interception utility.
Importing this single module automatically configures LLM clients (OpenAI, LangChain, etc.)
to route traffic transparently through the local Black Box Flight Recorder proxy.
No other code changes are needed in the agent.
"""

import os
import sys

# Set proxy environment variables automatically for standard clients
PROXY_HOST = os.getenv("BLACKBOX_PROXY_HOST", "127.0.0.1")
PROXY_PORT = os.getenv("BLACKBOX_PROXY_PORT", "8000")
DEFAULT_PROXY_URL = f"http://{PROXY_HOST}:{PROXY_PORT}/v1"

# Automatically export standard environment variables that all LLM SDKs check
if not os.getenv("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = DEFAULT_PROXY_URL


def patch_agent(agent_name: str = "custom_agent", run_id: str = None):
    """
    Optional helper to assign custom agent name and run ID headers if desired.
    If not called, default run IDs are generated automatically by the proxy.
    """
    if agent_name:
        os.environ["BLACKBOX_CURRENT_AGENT"] = agent_name
    if run_id:
        os.environ["BLACKBOX_CURRENT_RUN_ID"] = run_id
