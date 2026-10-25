"""Summary-memory backend — recent statements verbatim, older ones summarised.

This REPLACES the earlier pure-trim strategy (keep newest-that-fit, drop the
oldest outright). The old behaviour forgot the past completely; this one instead
COMPRESSES the past: it keeps the most recent ``keep_recent`` statements in full
and folds everything older into a single running summary chunk produced by the
LLM. That is the classic "recent-verbatim + older-summarised" memory strategy.

Expected behaviour vs the other columns:
- Cheaper than ``sanity`` (it never re-sends the full old history, only a short
  summary of it) but more expensive than the old pure-trim (it spends a few
  tokens summarising during load/update).
- Keeps SOME signal about the past (unlike pure-trim, which lost it), so it
  should do better on old-fact probes than a hard trimmer would — BUT a summary
  is lossy: the exact wording a probe needs can get blurred away, so it should
  still visibly lose to the graph/reducer on precise transitive invalidation.
  That lossy-compression trade-off is what this column demonstrates.

name: kept as "litellm-trim" so the registry entry, the scoreboard column and
the UI do not change — only the STRATEGY behind the column changed.

retrieval_context: exactly what is injected this turn — the summary chunk (when
one exists) followed by the recent verbatim statements. Fully exposed, so the
order-free metrics (contextual recall, contextual relevancy) score it honestly.

The summary is built through the shared gateway (``src/llm.py``) so its tokens
and latency are measured the same way as every other call; nothing is estimated
locally. Summarisation is cached implicitly by the gateway's content cache.
"""
from __future__ import annotations

from typing import Dict, List

from src.adapters.base import TurnResult
from src.llm import LLMGateway, get_gateway
from src.prompts import system_prompt

_SUMMARISE_SYSTEM = (
    "You compress a compliance/portfolio client's history into a faithful, "
    "dense summary. Preserve every concrete fact, value, date and any "
    "supersession ('X replaced Y', 'divested', 'no longer'). Do NOT invent, "
    "resolve or drop conflicts — record them. Output plain bullet points."
)


class LiteLLMTrimBackend:
    """Recent-verbatim + older-summarised memory.

    ``keep_recent`` statements stay in full; everything older is summarised into
    one chunk. The summary is recomputed lazily whenever the older set changes.
    """

    name = "litellm-trim"

    def __init__(self, gateway: LLMGateway = None, keep_recent: int = 4) -> None:
        self.gw = gateway or get_gateway()
        self.keep_recent = keep_recent
        self._store: Dict[str, List[str]] = {}
        # Cache of the last summary per session, keyed by how many older
        # statements it covered, so we only re-summarise when the old set grows.
        self._summary: Dict[str, str] = {}
        self._summary_covers: Dict[str, int] = {}

    def reset(self, session_id: str) -> None:
        self._store[session_id] = []
        self._summary.pop(session_id, None)
        self._summary_covers.pop(session_id, None)

    def load(self, session_id: str, documents: List[str]) -> None:
        self._store.setdefault(session_id, []).extend(documents)

    def update(self, session_id: str, statement: str) -> None:
        self._store.setdefault(session_id, []).append(statement)

    def _summarise_older(self, session_id: str, older: List[str]) -> str:
        """Return a cached/fresh summary of the ``older`` statements.

        Re-summarises only when the number of older statements changed since the
        last summary for this session (the gateway also content-caches, so a
        repeated call is free either way).
        """
        if not older:
            return ""
        if self._summary_covers.get(session_id) == len(older) and session_id in self._summary:
            return self._summary[session_id]
        body = "\n".join("- " + s for s in older)
        resp = self.gw.complete(
            [
                {"role": "system", "content": _SUMMARISE_SYSTEM},
                {"role": "user", "content": "EARLIER FACTS TO SUMMARISE:\n" + body},
            ]
        )
        self._summary[session_id] = resp.text.strip()
        self._summary_covers[session_id] = len(older)
        return self._summary[session_id]

    def _assemble(self, session_id: str) -> List[str]:
        """Build the injected context: [summary chunk?] + recent verbatim."""
        statements = list(self._store.get(session_id, []))
        if len(statements) <= self.keep_recent:
            return statements
        older = statements[: -self.keep_recent]
        recent = statements[-self.keep_recent :]
        summary = self._summarise_older(session_id, older)
        context: List[str] = []
        if summary:
            context.append("SUMMARY OF EARLIER FACTS:\n" + summary)
        context.extend(recent)
        return context

    def ask(self, session_id: str, question: str) -> TurnResult:
        context = self._assemble(session_id)
        prompt = "CONTEXT:\n" + "\n".join("- " + c for c in context) + \
                 "\n\nQUESTION: " + question
        resp = self.gw.complete(
            [{"role": "system", "content": system_prompt()}, {"role": "user", "content": prompt}]
        )
        return TurnResult(
            answer=resp.text,
            retrieval_context=context,
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
            latency_ms=resp.latency_ms,
        )


def build(gateway: LLMGateway = None) -> LiteLLMTrimBackend:
    """Zero-arg factory for the registry; uses the shared default gateway when
    called as ``build()``. Optional ``gateway`` for injection."""
    return LiteLLMTrimBackend(gateway)
