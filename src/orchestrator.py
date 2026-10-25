"""FastAPI orchestrator. Agent B.

Fans every call out to all registered backends in parallel and returns one
envelope per backend. The hard requirement: a backend that is slow, broken or
not installed must degrade to a greyed-out column, never take down /ask. On
stage, four columns and one honest error beats a spinner.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from .adapters.base import TurnResult
from .evaluator import Probe, log_turn, score_live, scoreboard
from .registry import SPECS, BackendSpec, status

# A backend past this is reported as a timeout and the other columns still render.
BACKEND_TIMEOUT_S = 20.0

app = FastAPI(title="RegWatch AI — orchestrator")


class LoadRequest(BaseModel):
    session_id: str
    documents: List[str]


class UpdateRequest(BaseModel):
    session_id: str
    statement: str


class AskRequest(BaseModel):
    """Field names follow data/probes.json, so a record from A's generator can
    be posted through with no translation."""
    session_id: str
    question: str
    probe_id: str = "live"
    fact_ids: List[str] = []
    expected_answer: Optional[str] = None
    superseded_values: List[str] = []
    phase: Optional[int] = None
    tier: Optional[str] = None


async def _call(spec: BackendSpec, method: str, *args: Any) -> Dict[str, Any]:
    """Run one backend's sync method off the event loop, isolating every failure."""
    envelope: Dict[str, Any] = {"backend": spec.name, "status": "ok",
                                "result": None, "error": None}
    if not spec.available():
        envelope.update(status="unavailable",
                        error=next(s["reason"] for s in status() if s["name"] == spec.name))
        return envelope
    try:
        backend = spec.get()
        envelope["result"] = await asyncio.wait_for(
            asyncio.to_thread(getattr(backend, method), *args), timeout=BACKEND_TIMEOUT_S
        )
    except asyncio.TimeoutError:
        envelope.update(status="timeout",
                        error="exceeded {}s".format(BACKEND_TIMEOUT_S))
    except NotImplementedError as exc:
        envelope.update(status="not_implemented", error=str(exc) or "not implemented")
    except Exception as exc:
        envelope.update(status="error", error="{}: {}".format(type(exc).__name__, exc))
    return envelope


async def _fan_out(method: str, *args: Any) -> List[Dict[str, Any]]:
    return list(await asyncio.gather(*[_call(s, method, *args) for s in SPECS]))


@app.get("/backends")
async def backends() -> List[Dict[str, Any]]:
    """Column states for the UI: who is ready, who is missing and why."""
    return status()


@app.post("/load")
async def load(req: LoadRequest) -> Dict[str, Any]:
    results = await _fan_out("load", req.session_id, req.documents)
    return {"session_id": req.session_id, "backends": _strip(results)}


@app.post("/update")
async def update(req: UpdateRequest) -> Dict[str, Any]:
    results = await _fan_out("update", req.session_id, req.statement)
    return {"session_id": req.session_id, "backends": _strip(results)}


@app.post("/ask")
async def ask(req: AskRequest) -> Dict[str, Any]:
    probe = Probe(probe_id=req.probe_id, question=req.question,
                  fact_ids=req.fact_ids, expected_answer=req.expected_answer,
                  superseded_values=req.superseded_values, phase=req.phase,
                  tier=req.tier)
    envelopes = await _fan_out("ask", req.session_id, req.question)

    columns = []
    for env in envelopes:
        turn: Optional[TurnResult] = env["result"] if env["status"] == "ok" else None
        scores = score_live(turn, probe) if turn else None
        log_turn(req.session_id, env["backend"], probe, turn, scores,
                 env["status"], env["error"])
        columns.append({
            "backend": env["backend"],
            "status": env["status"],
            "error": env["error"],
            "answer": turn.answer if turn else None,
            "retrieval_context": turn.retrieval_context if turn else None,
            "scores": scores,
        })
    return {"session_id": req.session_id, "question": req.question, "columns": columns}


@app.get("/scores")
async def scores() -> List[Dict[str, Any]]:
    return scoreboard()


def _strip(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """load/update return None; report status only."""
    return [{"backend": r["backend"], "status": r["status"], "error": r["error"]}
            for r in results]


