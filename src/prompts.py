"""The one system prompt every adapter answers with.

PLAN.md and the frozen contract both require it: "The LLM model and prompt
template are identical across all adapters. An adapter decides WHAT context to
assemble, never WHICH model answers." Holding the text in one place is what
actually enforces that — the three adapters had drifted to three different
prompts, and one of them had no recency instruction at all, which confounded
the first live run.

The recency hint is a toggle so the two arms of the experiment are
reproducible, and so the run log can record which arm produced a row:

    REGWATCH_RECENCY_HINT=1   (default) tell the model to prefer recent facts
    REGWATCH_RECENCY_HINT=0             say nothing about conflicts

Read at call time, so setting it before the run is enough.
"""
from __future__ import annotations

import os

_BASE = (
    "You are a compliance and portfolio advisor. Answer the user's question "
    "using ONLY the facts provided. "
)
_RECENCY_HINT = "If facts conflict, prefer the most recent. "
_TAIL = "Be concise — answer with the current value only."


def recency_hint_enabled() -> bool:
    return os.environ.get("REGWATCH_RECENCY_HINT", "1") != "0"


def system_prompt() -> str:
    return _BASE + (_RECENCY_HINT if recency_hint_enabled() else "") + _TAIL
