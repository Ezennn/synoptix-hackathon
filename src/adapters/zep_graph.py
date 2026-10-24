"""Zep / Graphiti temporal knowledge graph. Agent B.

SPIKE GATE (hour 0, 30 min). This adapter is only worth finishing if the spike
answers yes to the question that matters:

    Can `ask()` return the EXACT facts/edges Zep surfaced into the prompt?

If it cannot, say so here and this backend is scoreable on axes 1/4/5 only.
Record the answer in the docstring below before writing the rest.

SPIKE RESULT: <not run yet>

Showcase backend: Zep's graph carries validity intervals, so a contradicting
statement should invalidate the old edge rather than sit beside it. That is the
contrast with litellm_trim in the demo's five columns.
"""
from __future__ import annotations

import os
import time
from typing import List

from .base import TurnResult


class ZepGraphBackend:
    name = "zep_graph"

    def __init__(self) -> None:
        from zep_cloud.client import Zep  # imported lazily: see src/registry.py

        api_key = os.environ.get("ZEP_API_KEY")
        if not api_key:
            raise RuntimeError("ZEP_API_KEY not set")
        self._client = Zep(api_key=api_key)

    def reset(self, session_id: str) -> None:
        raise NotImplementedError("spike first: confirm thread/user delete semantics")

    def load(self, session_id: str, documents: List[str]) -> None:
        raise NotImplementedError("spike first: ingest documents into the graph")

    def update(self, session_id: str, statement: str) -> None:
        raise NotImplementedError("spike first: add the contradicting statement")

    def ask(self, session_id: str, question: str) -> TurnResult:
        # Must return the surfaced facts/edges verbatim as retrieval_context,
        # and route the answer call through A's gateway (src/llm.py) so tokens
        # and latency are measured the same way as every other backend.
        started = time.perf_counter()
        del started
        raise NotImplementedError("spike first: graph search -> context -> gateway answer")


def build() -> ZepGraphBackend:
    return ZepGraphBackend()
