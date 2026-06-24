"""Configuration reference & precedence (§2.3, §12.1).

Config precedence: explicit kwargs > YAML spec > env vars > defaults.
This module centralizes env-var names and a small precedence helper.
"""

from __future__ import annotations

import os
from typing import Any

# Environment variable names (§12.1)
ENV_DB_URL = "AGENTKIT_DB_URL"
ENV_LOG_LEVEL = "AGENTKIT_LOG_LEVEL"
ENV_MAX_CONCURRENT = "AGENTKIT_MAX_CONCURRENT"
ENV_OBS = "AGENTKIT_OBS"
ENV_OBS_SAMPLE_RATE = "OBS_SAMPLE_RATE"
ENV_MEMORY_BACKEND = "MEMORY_BACKEND"

DEFAULTS = {
    ENV_DB_URL: "sqlite:///./agentkit.db",
    ENV_LOG_LEVEL: "INFO",
    ENV_MAX_CONCURRENT: "64",
    ENV_OBS: "langfuse",
    ENV_OBS_SAMPLE_RATE: "1.0",
    ENV_MEMORY_BACKEND: "sqlite_vec",
}


def env(name: str, default: str | None = None) -> str | None:
    """Read an env var, falling back to a known default then the supplied one."""
    return os.environ.get(name, DEFAULTS.get(name, default))


def pick(*candidates: Any) -> Any:
    """Return the first non-None candidate (used for kwargs > spec > env > default)."""
    for c in candidates:
        if c is not None:
            return c
    return None


def db_url() -> str:
    return env(ENV_DB_URL)  # type: ignore[return-value]


def max_concurrent() -> int:
    return int(env(ENV_MAX_CONCURRENT))  # type: ignore[arg-type]
