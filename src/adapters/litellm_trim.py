"""LiteLLM-trim backend — hard message truncation to a token budget.

Keeps only the most recent statements that fit within ``max_context_tokens``,
dropping the oldest outright (no summarisation, no graph). Expected behaviour:
blazing fast and cheapest, HIGH precision on short windows, but catastrophic
forgetting on multi-hop / transitive probes whose supporting fact scrolled out
of the window. This is the "trimming loses the past" column.

retrieval_context: exactly the statements kept in-window — fully exposed.

Note: this uses a simple char/4 token estimate for the budget rather than
LiteLLM's trimmer, so the engine stays importable without litellm installed.
The gateway still reports real token counts for the cost axis.
"""
from __future__ import annotations

from typing import Dict, List

from src.adapters.base import TurnResult
from src.llm import LLMGateway, get_gateway

_SYSTEM = (
    "You are a compliance and portfolio advisor. Answer using ONLY the facts "
    "provided. Prefer the most recent. Answer with the current value only."
)


def _est_tokens(text: str) -> int:
    return max(1, round(len(text) / 4))


class LiteLLMTrimBackend:
    name = "litellm-trim"

    def __init__(self, gateway: LLMGateway = None, max_context_tokens: int = 120) -> None:
        self.gw = gateway or get_gateway()
        self.max_context_tokens = max_context_tokens
        self._store: Dict[str, List[str]] = {}

    def reset(self, session_id: str) -> None:
        self._store[session_id] = []

    def load(self, session_id: str, documents: List[str]) -> None:
        self._store.setdefault(session_id, []).extend(documents)

    def update(self, session_id: str, statement: str) -> None:
        self._store.setdefault(session_id, []).append(statement)

    def _trim(self, statements: List[str]) -> List[str]:
        """Keep newest-first until the token budget is spent, then restore order."""
        kept_rev: List[str] = []
        budget = self.max_context_tokens
        for s in reversed(statements):
            cost = _est_tokens(s)
            if cost > budget:
                break
            kept_rev.append(s)
            budget -= cost
        return list(reversed(kept_rev))

    def ask(self, session_id: str, question: str) -> TurnResult:
        kept = self._trim(list(self._store.get(session_id, [])))
        prompt = "RECENT FACTS:\n" + "\n".join("- " + c for c in kept) + \
                 "\n\nQUESTION: " + question
        resp = self.gw.complete(
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}]
        )
        return TurnResult(
            answer=resp.text,
            retrieval_context=kept,
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
            latency_ms=resp.latency_ms,
        )
