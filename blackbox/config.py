"""
Centralised settings for Black Box loaded from environment variables / .env file.
All other modules import from here — never hardcode values elsewhere.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Proxy ──────────────────────────────────────────────────────────────────
    blackbox_proxy_host: str = "0.0.0.0"
    blackbox_proxy_port: int = 8000

    # ── Upstream Provider Base URLs ────────────────────────────────────────────
    openai_base_url: str = "https://api.openai.com"
    google_base_url: str = "https://generativelanguage.googleapis.com"
    anthropic_base_url: str = "https://api.anthropic.com"

    # ── API Keys (forwarded transparently; never logged) ───────────────────────
    openai_api_key: str = ""
    google_api_key: str = ""
    anthropic_api_key: str = ""

    # ── Database ───────────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///blackbox_traces.db"

    # ── Logging ────────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── HTTP Client ────────────────────────────────────────────────────────────
    upstream_timeout_seconds: float = 120.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Returns a cached singleton Settings instance."""
    return Settings()
