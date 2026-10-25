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

TUNED DESIGN — consequence-aware self-editing
---------------------------------------------
Letta's memory is a set of self-edited core-memory blocks: on each new fact the
LLM decides whether to append, overwrite, or reconcile its notes. By DEFAULT it
is conservative (it just records the new fact). The tuning is a single change to
the memory-editing instruction (``MEMORY_EDIT_DIRECTIVE`` below): when a new
statement has CONSEQUENCES for an existing block, rewrite the block to reflect
the resolved current state. Then "client acme subB divested" arriving while a
block says "crypto exposure held via subB" causes Letta to rewrite that block to
"no crypto exposure (subB divested)" — inference baked in at WRITE time.

The directive is GENERIC (reconcile consequences of any supersession), naming no
scenario value, so it stays on the honest side of the tuning line. Trade-off:
more aggressive self-editing = more inference BUT more drift risk, which is why
the sweep runs Letta 3x and reports the spread. Apply ``MEMORY_EDIT_DIRECTIVE``
in ``update`` (as the agent's memory-edit system guidance) once the spike
confirms the lifecycle and context read-back.
"""
from __future__ import annotations

import os
import time
from typing import List

from .base import TurnResult

#: Consequence-aware self-editing directive (generic; no scenario value named).
#: Supplied to Letta's memory-edit step in ``update`` once the spike lands.
MEMORY_EDIT_DIRECTIVE = (
    "When a new statement supersedes or has consequences for an existing memory "
    "block, REWRITE that block to reflect the resolved current state (e.g. if an "
    "entity that held a position is divested/closed, record that the exposure via "
    "it has ended). Reconcile consequences; do not merely append the new fact."
)


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
        # TUNED: supply MEMORY_EDIT_DIRECTIVE as the memory-edit guidance so the
        # agent reconciles consequences of this statement into its blocks
        # (consequence-aware self-editing). See module docstring.
        raise NotImplementedError(
            "spike first: send the contradicting statement with MEMORY_EDIT_DIRECTIVE"
        )

    def ask(self, session_id: str, question: str) -> TurnResult:
        started = time.perf_counter()
        del started
        raise NotImplementedError("spike first: read core-memory block as context")


def build() -> LettaPagingBackend:
    return LettaPagingBackend()
