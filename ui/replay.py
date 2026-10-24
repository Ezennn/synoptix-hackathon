"""Reading and summarising results/run_log.jsonl. Agent B.

Kept free of Streamlit so the aggregation can be tested, and so the figures
script can reuse it. The UI layer does nothing but render what this returns.

Stale-context rate is the deterministic contamination measure: did the
superseded phrase appear in the text the backend injected? It needs no judge
model, so it is honest and free even before axes 2 and 3 are wired.
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_LOG = Path("results/run_log.jsonl")


def load_log(path: Path = DEFAULT_LOG) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def backend_order() -> List[str]:
    """Declared order, so columns never reshuffle between probes on stage."""
    from src.registry import SPECS
    return [s.name for s in SPECS]


def probes(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        probe = rec["probe"]
        seen.setdefault(probe["probe_id"], probe)
    return list(seen.values())


def columns_for(records: List[Dict[str, Any]], probe_id: str) -> List[Dict[str, Any]]:
    order = backend_order()
    rows = [r for r in records if r["probe"]["probe_id"] == probe_id]
    rows.sort(key=lambda r: order.index(r["backend"]) if r["backend"] in order else 999)
    return rows


def has_stale(record: Dict[str, Any]) -> Optional[bool]:
    """True if the superseded phrase reached the model. None if unknowable.

    An EMPTY retrieval_context is unknowable, not clean. The frozen contract
    says a backend that cannot expose its injected text returns `[]`, so
    treating that as "no stale fact found" would score the least transparent
    backend as the cleanest one.
    """
    stale_values = (record["probe"] or {}).get("superseded_values") or []
    context = record.get("retrieval_context")
    if not stale_values or not context:
        # No superseded value means the probe is a control with nothing stale to
        # find, which is not the same as a clean context. Both are unknowable.
        return None
    blob = " ".join(context).lower()
    return any(value.lower() in blob for value in stale_values)


def highlight(text: str, stale_values: Optional[List[str]],
              current: Optional[str]) -> str:
    """Superseded phrases in red, current phrase in green. The money shot.

    A probe can carry several superseded values (a fact mutated more than
    once), so every one of them is marked, not just the first.
    """
    out = html.escape(text)
    marks = [(value, "stale") for value in (stale_values or [])]
    if current:
        marks.append((current, "current"))
    for phrase, cls in marks:
        if not phrase:
            continue
        escaped = html.escape(phrase)
        idx = out.lower().find(escaped.lower())
        if idx == -1:
            continue
        out = (out[:idx] + '<span class="{}">'.format(cls)
               + out[idx:idx + len(escaped)] + "</span>" + out[idx + len(escaped):])
    return out


def _mean(values: List[Any]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    return (sum(clean) / len(clean)) if clean else None


def _percentile(values: List[float], pct: float) -> Optional[float]:
    """Nearest-rank percentile — no interpolation, so the number returned is a
    latency that actually happened."""
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[max(0, min(len(ordered) - 1, index))]


def scoreboard(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    order = backend_order()
    by_backend: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        by_backend.setdefault(rec["backend"], []).append(rec)

    rows = []
    for name, recs in by_backend.items():
        ok = [r for r in recs if r["status"] == "ok"]
        stale_flags = [has_stale(r) for r in ok]
        known = [f for f in stale_flags if f is not None]
        scored = [r["scores"] for r in ok if r.get("scores")]
        consistencies = [s["consistency"]["value"] for s in scored
                         if s.get("consistency", {}).get("value") is not None]
        latencies = [s["latency_ms"] for s in scored if s.get("latency_ms") is not None]
        costs = [s["cost_usd"] for s in scored if s.get("cost_usd") is not None]
        correct = sum(1 for v in consistencies if v >= 1.0)
        total_cost = sum(costs) if costs else None

        rows.append({
            "backend": name,
            "probes": len(recs),
            "answered": len(ok),
            "correct": correct,
            "stale_context_rate": (sum(known) / len(known)) if known else None,
            "consistency_rate": (sum(consistencies) / len(consistencies))
                                if consistencies else None,
            "avg_tokens_in": _mean([s["tokens_in"] for s in scored]),
            "avg_tokens_out": _mean([s["tokens_out"] for s in scored]),
            "avg_latency_ms": _mean(latencies),
            "p95_latency_ms": _percentile(latencies, 95),
            "total_cost_usd": total_cost,
            # Both cost measures are None when the model's rate is unknown,
            # rather than silently reported as zero.
            "cost_per_1k_usd": (total_cost / len(costs) * 1000) if costs else None,
            "cost_per_correct_usd": (total_cost / correct)
                                    if (total_cost is not None and correct) else None,
        })
    rows.sort(key=lambda r: order.index(r["backend"]) if r["backend"] in order else 999)
    return rows


def stale_by_phase(records: List[Dict[str, Any]]) -> Dict[str, Dict[int, float]]:
    """Context-rot curve: stale-context rate per backend, per phase."""
    buckets: Dict[str, Dict[int, List[bool]]] = {}
    for rec in records:
        if rec["status"] != "ok":
            continue
        flag = has_stale(rec)
        if flag is None:
            continue
        phase = rec["probe"].get("phase") or 0
        buckets.setdefault(rec["backend"], {}).setdefault(phase, []).append(flag)
    return {backend: {phase: sum(v) / len(v) for phase, v in phases.items()}
            for backend, phases in buckets.items()}
