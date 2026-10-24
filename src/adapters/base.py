"""THE CONTRACT — frozen interface every memory backend meets at.

This module is the single point where Agent A's engine and Agent B's
service/external/UI halves meet. Nothing here changes without a ``CONTRACT:``
PR that both agents review, because a change to ``TurnResult`` or the
``MemoryBackend`` signature breaks both halves at once.

Design rules this file encodes (see PLAN.md "The golden rule"):

1. Every backend is driven through the SAME four-method lifecycle
   (``reset`` -> ``load`` -> ``update`` -> ``ask``), so the only thing that
   varies between runs is which memory layer assembled the context.
2. Every probe returns a ``TurnResult`` carrying the EXACT text the memory
   layer injected this turn (``retrieval_context``). The evaluator's
   precision / faithfulness / recall metrics score that text directly — it is
   how we test the memory tool and not the LLM.
3. ``tokens_in`` / ``tokens_out`` / ``latency_ms`` are measured by the shared
   gateway (``src/llm.py``), never estimated by an adapter, so cost and speed
   are comparable across backends.
4. The LLM model and prompt template are identical across all adapters. An
   adapter decides WHAT context to assemble, never WHICH model answers.

Python target: 3.9. ``from __future__ import annotations`` + ``typing`` generics
keep this importable on 3.9 (PEP 604 ``X | None`` and bare ``list[str]`` at
runtime are 3.10+); adapters must follow the same rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

try:  # Protocol is in typing on 3.8+, runtime_checkable lets isinstance work
    from typing import Protocol, runtime_checkable
except ImportError:  # pragma: no cover — 3.9 always has these
    from typing_extensions import Protocol, runtime_checkable  # type: ignore


@dataclass
class TurnResult:
    """The scored result of one probe (``ask``) against a backend.

    Every field is required by at least one metric axis, so an adapter must
    populate all of them honestly rather than leaving a default:

    - ``answer``             -> axis 1 (state consistency, via the DAG metric)
                                and axis 3 (faithfulness).
    - ``retrieval_context``  -> axis 2 (context precision) and axis 3
                                (faithfulness + recall). This MUST be the real
                                text chunks the memory layer fed the model this
                                turn. If a backend cannot expose them, return
                                ``[]`` AND say so in the adapter docstring — that
                                backend is then only scorable on axes 1/4/5.
    - ``tokens_in`` / ``tokens_out`` -> axis 5 (cost). From the gateway.
    - ``latency_ms``         -> axis 4 (retrieval latency). Wall-clock for the
                                whole load+retrieve+generate of this turn.
    """

    answer: str
    retrieval_context: List[str] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        """Convenience for the cost axis; adapters need not set this."""
        return self.tokens_in + self.tokens_out


@runtime_checkable
class MemoryBackend(Protocol):
    """A memory/context layer under test.

    Implement all four methods plus ``name``. The harness drives one scenario
    per backend as:

        backend.reset(session_id)
        backend.load(session_id, phase1_documents)   # Phase 1 baseline ingest
        backend.update(session_id, mutation)         # Phase 2 fact mutation
        result = backend.ask(session_id, probe)      # Phase 3/4 probe -> scored

    ``load`` and ``update`` may be called multiple times to simulate several
    sessions over time (January docs, then an April mutation, etc.). ``ask`` is
    the only method that returns a scored ``TurnResult``.
    """

    #: Short, stable label used as the backend's column in results and the UI
    #: (e.g. "sanity", "litellm-trim", "langgraph", "zep", "letta").
    name: str

    def reset(self, session_id: str) -> None:
        """Clear ALL state for ``session_id`` so a run starts clean.

        Must leave no carry-over from a prior scenario: the next ``load`` is
        the backend's first sight of this session.
        """
        ...

    def load(self, session_id: str, documents: List[str]) -> None:
        """Ingest baseline context (Phase 1).

        ``documents`` are the policy pages / historical statements / meeting
        notes for one session. Called once per simulated session, in
        chronological order, before the mutation and probes for that timeline.
        """
        ...

    def update(self, session_id: str, statement: str) -> None:
        """Apply a fact mutation (Phase 2).

        ``statement`` is a single client/policy change that supersedes an
        earlier fact (e.g. "we are now strictly conservative and have divested
        from crypto"). The backend decides how to reconcile it — invalidate,
        append, or overwrite. That decision is exactly what the state-
        consistency metric measures.
        """
        ...

    def ask(self, session_id: str, question: str) -> TurnResult:
        """Answer a probe and return a scored ``TurnResult`` (Phase 3/4).

        The backend assembles whatever context it decides is relevant, calls
        the LLM ONLY through ``src/llm.py``, and returns the answer together
        with the real ``retrieval_context`` and the gateway's token/latency
        readings. No other method returns a result.
        """
        ...
