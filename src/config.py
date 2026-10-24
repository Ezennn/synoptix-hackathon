"""Config + secrets loader (PLAN item 5).

Loads a local ``.env`` file into ``os.environ`` so API keys and run settings
live in ONE place and never get hard-coded or committed (``.env`` is gitignored;
``.env.example`` is the committed template).

Zero-dependency on purpose: a tiny parser, not python-dotenv, so the engine has
no new install. Values already in the real environment WIN over the file, so a
CI secret or an inline ``REGWATCH_LLM_MODE=live python ...`` always overrides it.

Call ``load_env()`` once at process start (the gateway and the run scripts do).
Idempotent — calling it twice is harmless.

Recognised variables (see .env.example for the authoritative list):
    REGWATCH_LLM_MODE     stub | live            (default stub)
    REGWATCH_MODEL        answer model id        (default gpt-4o-mini)
    REGWATCH_CACHE_DIR    gateway cache dir       (default data/cache)
    OPENAI_API_KEY        LLM provider key (live mode; LiteLLM reads it)
    ZEP_API_KEY           Zep/Graphiti backend (Agent B)
    LETTA_API_KEY         Letta backend (Agent B)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV = ROOT / ".env"

_loaded = False


def load_env(path: Optional[Path] = None, override: bool = False) -> bool:
    """Load KEY=VALUE lines from ``.env`` into os.environ.

    - Lines that are blank or start with '#' are ignored.
    - Surrounding quotes on the value are stripped.
    - An existing env var is NOT overwritten unless ``override=True`` — so the
      real environment and inline overrides win over the file.
    Returns True if a file was read, False if none was found.
    """
    global _loaded
    env_path = Path(path) if path else DEFAULT_ENV
    if not env_path.exists():
        _loaded = True
        return False

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
    _loaded = True
    return True


def ensure_loaded() -> None:
    """Load .env once per process if it hasn't been loaded yet."""
    if not _loaded:
        load_env()
