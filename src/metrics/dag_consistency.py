"""State-Consistency metric — the custom DAG the benchmark can't buy.

This is axis 1. It is a DETERMINISTIC decision tree over the temporal answer
key, not an LLM judge: given a probe's expected current value and the values it
has superseded, it asks binary questions about the backend's answer and returns
0 or 1 with a reason. Determinism is the whole point — the score is repeatable,
immune to judge drift, and cheap.

The decision tree (per probe):

    node A: does the answer ASSERT the current expected value?
            no  -> FAIL (reason: missing current value)
            yes -> node B
    node B: does the answer ALSO assert a superseded value?
            yes -> FAIL (reason: asserts stale value X; contradiction unresolved)
            no  -> PASS

"Assert" is matched case-insensitively with light normalisation (yes/no
answers, currencies, tiers). For multi-value expected answers (the multi-hop
compliance probe, e.g. "no (v2 ...)"), every required token must appear.

Call signature is STABLE — Agent B imports ``score_probe`` and
``consistency_rate`` and must not need them to change.

    from src.metrics.dag_consistency import score_probe, consistency_rate
    result = score_probe(probe, answer_text)   # -> DagScore(score, passed, reason)
    rate   = consistency_rate([(probe, answer_text), ...])  # -> aggregate dict

Optionally wraps as a DeepEval custom metric via ``as_deepeval_metric`` if
DeepEval is installed; the raw scorer never needs it.

Python target: 3.9.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass
class DagScore:
    score: float          # 1.0 pass, 0.0 fail
    passed: bool
    reason: str
    probe_id: str = ""


_YESNO = {
    "yes": {"affirmative", "compliant"},
    "no": {"negative", "non-compliant", "prohibited"},
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _asserts_value(answer: str, value: str) -> bool:
    """Does ``answer`` assert ``value``?

    - yes/no values match a word-boundary 'yes'/'no' or a tight synonym set.
      (Deliberately NOT single letters or broad domain phrases — 'y' would
      otherwise match the 'y' in 'crypto'.)
    - otherwise require the value (or all its significant tokens) to appear.
    """
    a = _normalize(answer)
    v = _normalize(value)
    if v in ("yes", "no"):
        if re.search(r"\b" + re.escape(v) + r"\b", a):
            return True
        return any(re.search(r"\b" + re.escape(syn) + r"\b", a) for syn in _YESNO[v])
    if not v:
        return False
    # for multi-token expected answers, treat the leading yes/no + key tokens
    return v in a or all(tok in a for tok in _significant_tokens(v))


def _significant_tokens(value: str) -> List[str]:
    toks = re.findall(r"[a-z0-9]+", _normalize(value))
    stop = {"the", "a", "an", "of", "for", "to", "is", "and", "v", "clients", "tiered"}
    sig = [t for t in toks if t not in stop and len(t) > 2]
    return sig or toks


def score_probe(probe: Dict, answer: str) -> DagScore:
    """Score one backend answer against one probe's temporal answer key."""
    pid = probe.get("probe_id", "")
    expected = probe.get("expected_answer", "")
    superseded = probe.get("superseded_values", []) or []

    # node A — current value asserted?
    if not _asserts_value(answer, expected):
        return DagScore(0.0, False, "does not assert current value %r" % expected, pid)

    # node B — any superseded value also asserted?
    for stale in superseded:
        # a superseded value that is a substring of the current value isn't a
        # real contradiction (e.g. current 'United Kingdom' vs stale 'Kingdom')
        if _normalize(stale) and _normalize(stale) in _normalize(expected):
            continue
        if _asserts_value(answer, stale):
            return DagScore(
                0.0, False,
                "asserts current %r but ALSO stale %r (contradiction unresolved)"
                % (expected, stale),
                pid,
            )

    return DagScore(1.0, True, "asserts current %r, no stale value" % expected, pid)


def score(answer: str, probe: Dict) -> Tuple[float, str]:
    """Compatibility entry point for B's evaluator (sync point 2).

    B's ``src/evaluator.py`` calls ``score(answer=..., probe=...)`` and expects
    ``(value, reason)``. B's Probe dataclass uses singular field names
    (``fact_id``, ``superseded_answer``) while A's generate.py emits plural
    (``fact_ids``, ``superseded_values``). This wrapper accepts EITHER shape so
    neither side has to rewrite: it maps B's singular fields onto the plural
    ones ``score_probe`` reads, then delegates.
    """
    p = dict(probe)
    # reconcile field names B <-> A
    if "superseded_values" not in p:
        sv = p.get("superseded_answer")
        p["superseded_values"] = [sv] if sv else []
    if "fact_ids" not in p and p.get("fact_id"):
        p["fact_ids"] = [p["fact_id"]]
    res = score_probe(p, answer)
    return res.score, res.reason


def consistency_rate(cases: Sequence[Tuple[Dict, str]]) -> Dict:
    """Aggregate Consistency Rate over (probe, answer) pairs.

    Returns the overall rate plus a per-tier breakdown — the per-tier numbers
    are what make the scoreboard tell a story (e.g. a backend that aces 'direct'
    but fails 'transitive').
    """
    scores: List[DagScore] = [score_probe(p, a) for p, a in cases]
    total = len(scores) or 1
    passed = sum(s.passed for s in scores)

    by_tier: Dict[str, List[int]] = {}
    for (p, _), s in zip(cases, scores):
        t = p.get("tier", "unknown")
        by_tier.setdefault(t, [0, 0])
        by_tier[t][1] += 1
        by_tier[t][0] += int(s.passed)

    return {
        "consistency_rate": round(passed / total, 4),
        "passed": passed,
        "total": len(scores),
        "by_tier": {t: {"passed": c[0], "total": c[1], "rate": round(c[0] / (c[1] or 1), 4)}
                    for t, c in sorted(by_tier.items())},
        "failures": [
            {"probe_id": s.probe_id, "reason": s.reason} for s in scores if not s.passed
        ],
    }


def as_deepeval_metric(threshold: float = 0.99):
    """Optional: wrap ``score_probe`` as a DeepEval custom metric.

    Lets Agent B's evaluator treat axis 1 uniformly with the DeepEval/Ragas
    metrics. Imported lazily so this module never requires DeepEval.
    """
    try:
        from deepeval.metrics import BaseMetric  # type: ignore
        from deepeval.test_case import LLMTestCase  # type: ignore  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("as_deepeval_metric needs deepeval installed") from exc

    class StateConsistencyMetric(BaseMetric):  # pragma: no cover - thin wrapper
        def __init__(self, probe: Dict):
            self.threshold = threshold
            self.probe = probe
            self.score = 0.0
            self.reason = ""
            self.success = False

        def measure(self, test_case) -> float:
            res = score_probe(self.probe, test_case.actual_output)
            self.score, self.reason = res.score, res.reason
            self.success = res.passed
            return self.score

        async def a_measure(self, test_case) -> float:
            return self.measure(test_case)

        def is_successful(self) -> bool:
            return self.success

        @property
        def __name__(self):  # type: ignore[override]
            return "State Consistency (DAG)"

    return StateConsistencyMetric
