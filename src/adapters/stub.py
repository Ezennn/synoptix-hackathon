"""Test-fixture backends. Agent B.

Two deterministic backends with no LLM call and no network, so the service,
the evaluator and the UI can be exercised end to end before A's engine or the
hosted memory tools exist.

`stub_current` always surfaces the live fact; `stub_stale` always surfaces the
superseded one. Together they are the evaluator's own unit test: a correct
pipeline must score the first 1.0 and the second 0.0 on consistency.
"""
from __future__ import annotations

import time
from typing import Dict, List

from .base import TurnResult


class StubBackend:
    def __init__(self, name: str, mode: str) -> None:
        if mode not in ("current", "stale"):
            raise ValueError("mode must be 'current' or 'stale'")
        self.name = name
        self.mode = mode
        self._statements: Dict[str, List[str]] = {}

    def reset(self, session_id: str) -> None:
        self._statements.pop(session_id, None)

    def load(self, session_id: str, documents: List[str]) -> None:
        self._statements.setdefault(session_id, []).extend(documents)

    def update(self, session_id: str, statement: str) -> None:
        self._statements.setdefault(session_id, []).append(statement)

    def ask(self, session_id: str, question: str) -> TurnResult:
        started = time.perf_counter()
        history = self._statements.get(session_id, [])
        if not history:
            context = ["(no memory for this session)"]
        elif self.mode == "current":
            context = [history[-1]]
        else:
            context = [history[0]]
        answer = "Based on the client record: {}".format(context[0])
        return TurnResult(
            answer=answer,
            retrieval_context=context,
            tokens_in=sum(len(c.split()) for c in context) + len(question.split()),
            tokens_out=len(answer.split()),
            latency_ms=(time.perf_counter() - started) * 1000,
        )


def build_current() -> StubBackend:
    return StubBackend("stub_current", "current")


def build_stale() -> StubBackend:
    return StubBackend("stub_stale", "stale")
