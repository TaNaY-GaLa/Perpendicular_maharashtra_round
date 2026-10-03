"""
Black Box SDK auto-interception utility.
Importing this single module automatically intercepts and routes LLM clients:
- OpenAI SDK & compatible libraries
- Google GenAI (google-genai) native SDK
- Anthropic Claude native SDK
- LangChain / LlamaIndex / AutoGen / CrewAI

Transparently intercepts outbound LLM HTTP requests without requiring any code modifications to the agent logic!
"""

import os
import sys

PROXY_HOST = os.getenv("BLACKBOX_PROXY_HOST", "127.0.0.1")
PROXY_PORT = os.getenv("BLACKBOX_PROXY_PORT", "8000")
BASE_PROXY_URL = f"http://{PROXY_HOST}:{PROXY_PORT}"

# 1. Standard environment variables for OpenAI-compatible SDKs
os.environ["OPENAI_BASE_URL"] = f"{BASE_PROXY_URL}/v1"
os.environ["ANTHROPIC_BASE_URL"] = f"{BASE_PROXY_URL}/anthropic"
os.environ["GEMINI_BASE_URL"] = f"{BASE_PROXY_URL}/google"

# 2. Patch Google GenAI SDK (google.genai) automatically if present
try:
    import google.genai as genai_module
    _orig_client_init = genai_module.Client.__init__

    def _patched_client_init(self, *args, **kwargs):
        # Inject Black Box proxy into http_options without user having to code it
        http_options = kwargs.get("http_options", {}) or {}
        if isinstance(http_options, dict) and "base_url" not in http_options:
            http_options["base_url"] = f"{BASE_PROXY_URL}/google"
            http_options["api_version"] = "v1beta"
            kwargs["http_options"] = http_options
        _orig_client_init(self, *args, **kwargs)

    genai_module.Client.__init__ = _patched_client_init
except ImportError:
    pass

# 3. Patch Anthropic SDK automatically if present
try:
    import anthropic as anthropic_module
    _orig_anthropic_init = anthropic_module.Anthropic.__init__

    def _patched_anthropic_init(self, *args, **kwargs):
        if "base_url" not in kwargs or not kwargs["base_url"]:
            kwargs["base_url"] = f"{BASE_PROXY_URL}/anthropic"
        _orig_anthropic_init(self, *args, **kwargs)

    anthropic_module.Anthropic.__init__ = _patched_anthropic_init
except ImportError:
    pass


def patch_agent(agent_name: str = "custom_agent", run_id: str = None):
    """Optional helper to assign custom metadata tags to the current agent run."""
    if agent_name:
        os.environ["BLACKBOX_CURRENT_AGENT"] = agent_name
    if run_id:
        os.environ["BLACKBOX_CURRENT_RUN_ID"] = run_id
