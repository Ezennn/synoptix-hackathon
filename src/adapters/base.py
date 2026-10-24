"""THE CONTRACT — owned by Agent A.

Copied verbatim from PLAN.md so Agent B can start before A's first push.
A's commit supersedes this file. Do not change a signature here without a PR
titled `CONTRACT:` agreed by both agents.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol


@dataclass
class TurnResult:
    answer: str                    # the LLM's answer to the probe
    retrieval_context: List[str]   # EXACT text chunks the memory layer injected
    tokens_in: int                 # prompt tokens actually sent
    tokens_out: int                # completion tokens
    latency_ms: float              # wall-clock for load+retrieve+generate this turn


class MemoryBackend(Protocol):
    name: str

    def reset(self, session_id: str) -> None: ...
    def load(self, session_id: str, documents: List[str]) -> None: ...
    def update(self, session_id: str, statement: str) -> None: ...
    def ask(self, session_id: str, question: str) -> TurnResult: ...
