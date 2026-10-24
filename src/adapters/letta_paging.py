"""Letta / MemGPT self-editing core memory. Agent B.

SPIKE GATE (hour 0, 30 min). Two things to settle before committing time:

  1. Can the core-memory block be read back as the exact injected context?
     Letta edits its own memory, so context capture may be lossy — if it is,
     say so here and score this backend on axes 1/4/5 only.
  2. Letta needs Python 3.11+. This machine is on 3.9.6, so it cannot run here
     without a newer interpreter.

SPIKE RESULT: <not run yet>

Letta is the one backend with genuine run-to-run variance, so the sweep runs it
3x per probe and reports the spread rather than a single number.
"""
from __future__ import annotations

import os
import time
from typing import List

from .base import TurnResult


class LettaPagingBackend:
    name = "letta_paging"

    def __init__(self) -> None:
        from letta_client import Letta  # imported lazily: see src/registry.py

        self._client = Letta(token=os.environ.get("LETTA_API_KEY"))

    def reset(self, session_id: str) -> None:
        raise NotImplementedError("spike first: agent lifecycle per session")

    def load(self, session_id: str, documents: List[str]) -> None:
        raise NotImplementedError("spike first: ingest into archival/core memory")

    def update(self, session_id: str, statement: str) -> None:
        raise NotImplementedError("spike first: send the contradicting statement")

    def ask(self, session_id: str, question: str) -> TurnResult:
        started = time.perf_counter()
        del started
        raise NotImplementedError("spike first: read core-memory block as context")


def build() -> LettaPagingBackend:
    return LettaPagingBackend()
