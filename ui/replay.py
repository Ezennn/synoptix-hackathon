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
                records.append(_price(json.loads(line)))
    return records


def _price(record: Dict[str, Any]) -> Dict[str, Any]:
    """Fill in cost for a row written before its model had a confirmed rate.

    A row stores what was known when it was written, so a run whose rate only
    reached config/pricing.yaml afterwards carries `cost_usd: null` for ever and
    the whole dashboard reads unpriced. Pricing it here — from the row's own
    model and its measured token counts, not the current environment's — keeps
    the committed evidence untouched while the figures stay current.

    A row that already has a cost keeps it, and a model still absent from the
    table stays unknown rather than quietly becoming zero.
    """
    scores = record.get("scores") or {}
    if scores.get("cost_usd") is not None:
        return record
    tokens_in, tokens_out = scores.get("tokens_in"), scores.get("tokens_out")
    if tokens_in is None or tokens_out is None:
        return record
    try:
        from src.evaluator import price_for
        rate = price_for(record.get("model"))
    except Exception:
        return record
    if not rate or rate.get("input_per_m") is None or rate.get("output_per_m") is None:
        return record
    scores["cost_usd"] = round(tokens_in / 1e6 * rate["input_per_m"]
                               + tokens_out / 1e6 * rate["output_per_m"], 8)
    record["scores"] = scores
    record.setdefault("pricing", None)
    if record["pricing"] is None:
        record["pricing"] = dict(rate, priced_at_read=True)
    return record


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

        # A cached turn returns from disk: its latency is a file read, and it
        # cost nothing. Keeping them in the averages reports a run as faster
        # and cheaper than the same work would be in production, so the two are
        # separated rather than blended.
        fresh = [r for r in ok if r.get("cached") is False]
        fresh_latencies = [(r["scores"] or {}).get("latency_ms") for r in fresh]
        fresh_latencies = [v for v in fresh_latencies if v is not None]
        spent = [(r["scores"] or {}).get("cost_usd") for r in fresh]
        spent = [v for v in spent if v is not None]
        totals_tokens = [(s.get("tokens_in") or 0) + (s.get("tokens_out") or 0)
                         for s in scored]

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
            "avg_total_tokens": _mean(totals_tokens),
            "avg_latency_ms": _mean(latencies),
            "median_latency_ms": _percentile(latencies, 50),
            "p95_latency_ms": _percentile(latencies, 95),
            # Fresh-only: what the same work costs and takes with no cache.
            "fresh_calls": len(fresh),
            "cached_calls": sum(1 for r in ok if r.get("cached") is True),
            "fresh_avg_latency_ms": _mean(fresh_latencies),
            "fresh_p95_latency_ms": _percentile(fresh_latencies, 95),
            # Modelled cost prices every answered probe as if it were a fresh
            # call — the production economics. Spent is what this run actually
            # paid, which is lower whenever the cache served a row.
            "total_cost_usd": total_cost,
            "spent_usd": sum(spent) if spent else None,
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


TIERS = ["baseline", "direct", "partial", "transitive", "reinstatement"]

# What went wrong, and what an engineer should do about it. The cause is read
# off the evidence — what the context actually held — never assumed. Claiming
# "retrieval failed" when the fact was present would send someone to fix the
# wrong layer.
CAUSES = {
    "fact_missing": (
        "The current fact never reached the model.",
        "Memory layer — the fact was dropped before the prompt was built. Raise "
        "the token budget, or use a reducer that keeps the latest value per fact, "
        "and retest this probe.",
    ),
    "only_stale": (
        "Only the superseded value was in the context.",
        "Memory layer — the update did not displace the old value. Check how this "
        "backend handles a fact being overwritten, then retest.",
    ),
    "both_present": (
        "Both the current and the superseded value were in the context; the model "
        "chose the wrong one.",
        "Prompt or reducer — the context was complete, so this is a resolution "
        "failure. Keep the recency instruction in the system prompt, or drop "
        "superseded values before the prompt is built.",
    ),
    "declined_despite_fact": (
        "The current fact WAS in the context, but the model declined to answer.",
        "Prompt — not a memory failure. The context was sufficient; the model was "
        "too cautious to use it.",
    ),
    "fact_present_wrong_answer": (
        "The current fact was in the context and the answer was still wrong.",
        "Model or question — the memory layer did its job here. Check the probe "
        "wording before blaming retrieval.",
    ),
}


def _value_present(context: str, expected: str) -> bool:
    """Did the expected value actually reach the model?

    Delegates to the scorer so the dashboard and the metric agree on what
    "present" means. A plain substring test fails on compound multi-hop values
    like "yes (v2 (...))" and would blame retrieval for a failure it did not
    cause.
    """
    if not context or not expected:
        return False
    try:
        from src.metrics.dag_consistency import value_in_text
    except Exception:
        return expected.lower() in context.lower()
    return value_in_text(context, expected)


def diagnose(record: Dict[str, Any]) -> Optional[str]:
    """Why this probe failed, judged only from the recorded evidence."""
    scores = record.get("scores") or {}
    consistency = scores.get("consistency") or {}
    if consistency.get("value") != 0.0:
        return None
    probe = record.get("probe") or {}
    context = " ".join(record.get("retrieval_context") or []).lower()
    expected = probe.get("expected_answer") or ""
    current_present = _value_present(context, expected)
    stale = has_stale(record)

    if consistency.get("abstained"):
        return "declined_despite_fact" if current_present else "fact_missing"
    if stale and not current_present:
        return "only_stale"
    if stale and current_present:
        return "both_present"
    if current_present:
        return "fact_present_wrong_answer"
    return "fact_missing"


def failures(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Every failed probe, with its cause attached."""
    out = []
    for record in records:
        cause = diagnose(record)
        if cause:
            out.append({"record": record, "cause": cause})
    return out


def by_tier(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Optional[float]]]:
    """Consistency per backend per difficulty tier — where each method breaks."""
    buckets: Dict[str, Dict[str, List[float]]] = {}
    for record in records:
        if record["status"] != "ok":
            continue
        value = ((record.get("scores") or {}).get("consistency") or {}).get("value")
        if value is None:
            continue
        tier = (record.get("probe") or {}).get("tier") or "unknown"
        buckets.setdefault(record["backend"], {}).setdefault(tier, []).append(value)
    return {backend: {tier: sum(v) / len(v) for tier, v in tiers.items()}
            for backend, tiers in buckets.items()}


def run_meta(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Provenance of the run, for the header."""
    if not records:
        return {}
    first = records[0]
    return {
        "model": first.get("model"),
        "mode": first.get("llm_mode"),
        "recency_hint": first.get("recency_hint"),
        "probes": len({r["probe"]["probe_id"] for r in records}),
        "backends": len({r["backend"] for r in records}),
        "answers": sum(1 for r in records if r["status"] == "ok"),
    }


def efficiency_pick(board: List[Dict[str, Any]],
                    min_accuracy: float = 0.80) -> Dict[str, Any]:
    """The cheapest backend that still meets a minimum accuracy.

    Cost ranks on tokens per query rather than dollars, so the comparison works
    even when the model has no confirmed rate — an unpriced model should not
    mean no recommendation, and dollars are a linear function of tokens anyway.

    Returns the chosen backend plus every candidate's verdict, so the answer is
    never just a name: a backend that failed the bar is shown failing it.
    """
    candidates = []
    for row in board:
        if not row.get("answered") or row.get("consistency_rate") is None:
            continue
        candidates.append({
            "backend": row["backend"],
            "accuracy": row["consistency_rate"],
            "tokens": row.get("avg_tokens_in") or 0,
            "cost_per_1k_usd": row.get("cost_per_1k_usd"),
            "meets": row["consistency_rate"] >= min_accuracy,
        })
    passing = [c for c in candidates if c["meets"]]
    chosen = min(passing, key=lambda c: c["tokens"]) if passing else None
    return {
        "min_accuracy": min_accuracy,
        "chosen": chosen,
        "candidates": sorted(candidates, key=lambda c: -c["accuracy"]),
        "reason": ("cheapest of {} meeting {:.0%}".format(len(passing), min_accuracy)
                   if chosen else
                   "no configuration reached {:.0%} — none is shippable at this "
                   "bar".format(min_accuracy)),
    }
