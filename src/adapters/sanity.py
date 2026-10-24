"""Sanity backend — the control / accuracy ceiling.

Stuffs the ENTIRE accumulated history (every statement and every Phase-4
distractor) into the prompt with no memory management at all. It should score
near-perfectly on state consistency when the model reasons well, and WORST on
context precision (it injects all the noise) and cost (it sends everything).
It is the baseline every other backend is measured against.

retrieval_context: the full list of statements it fed the model — fully
exposed, so this backend is scorable on all five axes.
"""
from __future__ import annotations

from typing import Dict, List

from src.adapters.base import TurnResult
from src.llm import LLMGateway, get_gateway

_SYSTEM = (
    "You are a compliance and portfolio advisor. Answer the user's question "
    "using ONLY the facts provided. If facts conflict, prefer the most recent. "
    "Be concise — answer with the current value only."
)


class SanityBackend:
    name = "sanity"

    def __init__(self, gateway: LLMGateway = None) -> None:
        self.gw = gateway or get_gateway()
        self._store: Dict[str, List[str]] = {}

    def reset(self, session_id: str) -> None:
        self._store[session_id] = []

    def load(self, session_id: str, documents: List[str]) -> None:
        self._store.setdefault(session_id, []).extend(documents)

    def update(self, session_id: str, statement: str) -> None:
        self._store.setdefault(session_id, []).append(statement)

    def ask(self, session_id: str, question: str) -> TurnResult:
        context = list(self._store.get(session_id, []))
        prompt = "FACTS (chronological):\n" + "\n".join(
            "- " + c for c in context
        ) + "\n\nQUESTION: " + question
        resp = self.gw.complete(
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}]
        )
        return TurnResult(
            answer=resp.text,
            retrieval_context=context,
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
            latency_ms=resp.latency_ms,
        )
