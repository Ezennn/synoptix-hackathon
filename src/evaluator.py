"""Scoring and persistence. Agent B.

Two tiers, deliberately separated:

  score_live()     free, deterministic, millisecond — runs in the /ask path.
                   Axis 1 (consistency) + axes 4/5 (latency, tokens, cost).
  score_offline()  DeepEval / Ragas judge-model metrics — axes 2 and 3. Runs as
                   a batch pass over results/run_log.jsonl, never in a request.

PLAN.md says /ask returns live scores. The judge metrics cost seconds and real
money per probe, which is not survivable on stage, so the LLM-judged axes are
computed offline and joined into the scoreboard. See AGENT_B_PLAN.md, decision 2.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .adapters.base import TurnResult

RESULTS_DIR = Path("results")
RUN_LOG = RESULTS_DIR / "run_log.jsonl"
DB_PATH = RESULTS_DIR / "scores.sqlite"

# Answer-model price per 1M tokens. Override per run; the figure on the chart
# must match the model actually used.
PRICE_IN_PER_M = 3.00
PRICE_OUT_PER_M = 15.00


@dataclass
class Probe:
    """One question from A's generate.py.

    Mirrors the fields B's evaluator needs. Confirm against the schema A
    publishes in generate.py's docstring (sync point 2) before the full run.
    """
    probe_id: str
    question: str
    fact_id: Optional[str] = None
    expected_answer: Optional[str] = None
    superseded_answer: Optional[str] = None
    asked_at: Optional[str] = None
    phase: Optional[int] = None


def _consistency(turn: TurnResult, probe: Probe) -> Dict[str, Any]:
    """Axis 1, via A's DAG metric. No fallback score is invented.

    If A's scorer is not on this machine yet, the result is explicitly unscored
    so a missing dependency can never be mistaken for a passing backend.
    """
    try:
        from .metrics.dag_consistency import score as dag_score  # type: ignore
    except Exception:
        return {"value": None, "reason": "dag_consistency not available (owner A)"}
    try:
        value, reason = dag_score(answer=turn.answer, probe=asdict(probe))
        return {"value": value, "reason": reason}
    except Exception as exc:  # a scorer bug must not take down /ask
        return {"value": None, "reason": "scorer error: {}".format(exc)}


def _cost_usd(turn: TurnResult) -> float:
    return (turn.tokens_in / 1e6) * PRICE_IN_PER_M + (turn.tokens_out / 1e6) * PRICE_OUT_PER_M


def score_live(turn: TurnResult, probe: Probe) -> Dict[str, Any]:
    """Deterministic scores only. Safe to call on every request."""
    return {
        "consistency": _consistency(turn, probe),
        "latency_ms": round(turn.latency_ms, 1),
        "tokens_in": turn.tokens_in,
        "tokens_out": turn.tokens_out,
        "context_chunks": len(turn.retrieval_context),
        "cost_usd": round(_cost_usd(turn), 6),
        # Axes 2 and 3 are filled by score_offline().
        "context_precision": None,
        "faithfulness": None,
        "contextual_recall": None,
    }


def log_turn(session_id: str, backend: str, probe: Probe, turn: Optional[TurnResult],
             scores: Optional[Dict[str, Any]], status: str, error: Optional[str] = None) -> None:
    """One line per probe per backend. The UI replays this file; the offline
    scorer reads it. Failures are logged too — a backend that timed out is a
    result, not a gap."""
    RESULTS_DIR.mkdir(exist_ok=True)
    record = {
        "ts": time.time(),
        "session_id": session_id,
        "backend": backend,
        "status": status,
        "error": error,
        "probe": asdict(probe),
        "answer": turn.answer if turn else None,
        "retrieval_context": turn.retrieval_context if turn else None,
        "scores": scores,
    }
    with RUN_LOG.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    _persist(record)


def _connect() -> sqlite3.Connection:
    RESULTS_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scores (
            ts REAL, session_id TEXT, backend TEXT, probe_id TEXT, status TEXT,
            consistency REAL, latency_ms REAL, tokens_in INTEGER, tokens_out INTEGER,
            cost_usd REAL, context_precision REAL, faithfulness REAL, contextual_recall REAL
        )
    """)
    return conn


def _persist(record: Dict[str, Any]) -> None:
    s = record.get("scores") or {}
    consistency = (s.get("consistency") or {}).get("value")
    conn = _connect()
    with conn:
        conn.execute(
            "INSERT INTO scores VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (record["ts"], record["session_id"], record["backend"],
             record["probe"]["probe_id"], record["status"], consistency,
             s.get("latency_ms"), s.get("tokens_in"), s.get("tokens_out"),
             s.get("cost_usd"), s.get("context_precision"), s.get("faithfulness"),
             s.get("contextual_recall")),
        )
    conn.close()


def scoreboard() -> List[Dict[str, Any]]:
    """Per-backend aggregate across all five axes. Drives GET /scores."""
    if not DB_PATH.exists():
        return []
    conn = _connect()
    rows = conn.execute("""
        SELECT backend,
               COUNT(*)                                   AS probes,
               SUM(status = 'ok')                         AS ok,
               AVG(consistency)                           AS consistency_rate,
               AVG(latency_ms)                            AS latency_ms,
               AVG(tokens_in)                             AS tokens_in,
               SUM(cost_usd)                              AS cost_usd,
               AVG(context_precision)                     AS context_precision,
               AVG(faithfulness)                          AS faithfulness,
               AVG(contextual_recall)                     AS contextual_recall
        FROM scores GROUP BY backend ORDER BY consistency_rate DESC
    """).fetchall()
    cols = [d[0] for d in conn.execute("SELECT * FROM scores LIMIT 0").description]
    conn.close()
    keys = ["backend", "probes", "ok", "consistency_rate", "latency_ms", "tokens_in",
            "cost_usd", "context_precision", "faithfulness", "contextual_recall"]
    del cols
    return [dict(zip(keys, row)) for row in rows]


def score_offline(limit: Optional[int] = None) -> int:
    """Axes 2 and 3 over the run log, using DeepEval / Ragas judge models.

    Not wired yet — needs the judge model configured and A's probe schema
    confirmed. Kept as a separate entry point so it can be run between the
    full sweep and the scoreboard freeze at hour 21.
    """
    raise NotImplementedError(
        "offline judge metrics not wired yet: needs DeepEval/Ragas + judge model config"
    )
