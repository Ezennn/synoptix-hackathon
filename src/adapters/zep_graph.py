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

TUNED DESIGN (Rung-equivalent for a graph backend) — multi-hop depth = 2
------------------------------------------------------------------------
The transitive probe ("does the client still have crypto exposure?" after the
holding entity is divested) needs the invalidating fact, which sits ONE entity
hop away from the queried entity:

    crypto_exposure --held_via--> subsidiary_B --ownership--> divested

A depth-1 (direct-neighbours-only) graph search returns facts attached to
"crypto exposure" and MISSES "subsidiary B divested". The tuning is therefore a
single retrieval parameter: expand the graph search to DEPTH 2 so the search
pulls in the divestment fact one hop out, and let both facts land in
retrieval_context for the answer model to connect. Depth is matched to the
longest dependency chain in the probe set (one link => depth 2); deeper than the
chain only adds noise and cost. This is a GENERIC retrieval setting, not a
scenario-specific edge, so it stays on the honest side of the tuning line.

Set via ``GRAPH_SEARCH_DEPTH`` below; the ``ask`` implementation (once the spike
confirms edge-level context capture) must pass this depth to Zep's graph search
and keep the connected facts the reranker returns.
"""
from __future__ import annotations

import os
import time
from typing import List

from .base import TurnResult

#: Multi-hop retrieval depth. 2 = pull in facts one entity-hop away from the
#: queried entity, which is what the transitive/divestment probe needs. Tune to
#: match the longest dependency chain; generic (no scenario value baked in).
GRAPH_SEARCH_DEPTH = 2


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
        #
        # TUNED: run the graph search at depth GRAPH_SEARCH_DEPTH (=2) so facts
        # one entity-hop from the queried entity (e.g. "subsidiary B divested"
        # reached via the held_via edge) are pulled into retrieval_context,
        # enabling the transitive-invalidation answer. See module docstring.
        started = time.perf_counter()
        del started
        raise NotImplementedError(
            "spike first: graph search (depth=%d) -> context -> gateway answer"
            % GRAPH_SEARCH_DEPTH
        )


def build() -> ZepGraphBackend:
    return ZepGraphBackend()
