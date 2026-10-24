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
import os
import sqlite3
import time
from dataclasses import asdict, dataclass
from dataclasses import field as dc_field
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Any, Dict, List, Optional

from .adapters.base import TurnResult

RESULTS_DIR = Path("results")
RUN_LOG = RESULTS_DIR / "run_log.jsonl"
DB_PATH = RESULTS_DIR / "scores.sqlite"

# Answer-model price per 1M tokens, by model.
#
# Cost is reported only for a model whose rate we actually know: an unrecognised
# model yields None rather than a plausible-looking number, because a wrong cost
# on a slide is worse than a missing one. Set the rate for any model with
# REGWATCH_PRICE_IN_PER_M / REGWATCH_PRICE_OUT_PER_M.
_PRICES_PER_M = {
    "gpt-4o-mini": (0.15, 0.60),
}


def _price_per_m(model: Optional[str]) -> Optional[tuple]:
    env_in = os.environ.get("REGWATCH_PRICE_IN_PER_M")
    env_out = os.environ.get("REGWATCH_PRICE_OUT_PER_M")
    if env_in and env_out:
        return float(env_in), float(env_out)
    name = (model or "").lower()
    for key, rate in _PRICES_PER_M.items():
        if key in name:
            return rate
    return None


@dataclass
class Probe:
    """One question from A's generate.py (data/probes.json).

    Field names mirror that file EXACTLY, because asdict(probe) is handed
    straight to A's score_probe(), which reads `expected_answer` and
    `superseded_values` by key. Renaming either here does not raise — the
    scorer just sees no superseded values, skips the stale check, and passes
    every answer that asserts the current value. Keep these names in step with
    generate.py (sync point 2).
    """
    probe_id: str
    question: str
    phase: Optional[int] = None
    tier: Optional[str] = None
    fact_ids: List[str] = dc_field(default_factory=list)
    expected_answer: Optional[str] = None
    superseded_values: List[str] = dc_field(default_factory=list)
    ask_after_session: Optional[int] = None
    ask_date: Optional[str] = None
    multi_hop: bool = False
    notes: Optional[str] = None

    @classmethod
    def from_record(cls, record: Dict[str, Any]) -> "Probe":
        """Build from one entry of data/probes.json, ignoring unknown keys so
        A can add fields without breaking the service."""
        known = {f.name for f in dc_fields(cls)}
        return cls(**{k: v for k, v in record.items() if k in known})


def _consistency(turn: TurnResult, probe: Probe) -> Dict[str, Any]:
    """Axis 1, via A's DAG metric. No fallback score is invented.

    If A's scorer is not on this machine yet, the result is explicitly unscored
    so a missing dependency can never be mistaken for a passing backend.
    """
    try:
        from .metrics.dag_consistency import score_probe  # type: ignore
    except Exception:
        return {"value": None, "reason": "dag_consistency not available (owner A)"}
    try:
        # A's signature is score_probe(probe_dict, answer) -> DagScore.
        result = score_probe(asdict(probe), turn.answer)
        return {"value": result.score, "passed": result.passed,
                "abstained": getattr(result, "abstained", False),
                "reason": result.reason}
    except Exception as exc:  # a scorer bug must not take down /ask
        return {"value": None, "reason": "scorer error: {}".format(exc)}


def _cost_usd(turn: TurnResult) -> Optional[float]:
    rate = _price_per_m(os.environ.get("REGWATCH_MODEL", "gpt-4o-mini"))
    if rate is None:
        return None
    price_in, price_out = rate
    return (turn.tokens_in / 1e6) * price_in + (turn.tokens_out / 1e6) * price_out


def score_live(turn: TurnResult, probe: Probe) -> Dict[str, Any]:
    """Deterministic scores only. Safe to call on every request."""
    return {
        "consistency": _consistency(turn, probe),
        "latency_ms": round(turn.latency_ms, 1),
        "tokens_in": turn.tokens_in,
        "tokens_out": turn.tokens_out,
        "context_chunks": len(turn.retrieval_context),
        "cost_usd": None if _cost_usd(turn) is None else round(_cost_usd(turn), 8),
        # Axes 2 and 3 are filled by score_offline().
        "context_precision": None,
        "faithfulness": None,
        "contextual_recall": None,
    }


def provenance() -> Dict[str, Any]:
    """Where this row's answer came from.

    Read from the environment, mirroring the gateway's own defaults in
    src/llm.py, so a row always records the mode and model that produced it. A
    stub row and a live row are otherwise indistinguishable once written.
    """
    return {
        "llm_mode": os.environ.get("REGWATCH_LLM_MODE", "stub").lower(),
        "model": os.environ.get("REGWATCH_MODEL", "gpt-4o-mini"),
        "cache_dir": os.environ.get("REGWATCH_CACHE_DIR"),
        # Which arm of the prompt experiment produced this row.
        "recency_hint": os.environ.get("REGWATCH_RECENCY_HINT", "1") != "0",
    }


def check_log_mode(mode: str, path: Path = RUN_LOG) -> None:
    """Refuse to append rows of one mode to a log holding another.

    Mixing stub and live results is unrecoverable after the fact — the rows look
    the same and the numbers quietly average together — so this is a hard stop,
    not a warning. Rows written before provenance existed read as "unknown" and
    also block, since they cannot be vouched for either.
    """
    if not path.exists():
        return
    seen = set()
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(json.loads(line).get("llm_mode") or "unknown")
            except json.JSONDecodeError:
                continue
    conflicting = sorted(m for m in seen if m != mode)
    if conflicting:
        raise RuntimeError(
            "{} already holds {} rows; refusing to append {!r} rows. "
            "Move it aside (e.g. results/run_log.{}.jsonl) and run again.".format(
                path, "/".join(conflicting), mode, conflicting[0])
        )


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
        **provenance(),
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
