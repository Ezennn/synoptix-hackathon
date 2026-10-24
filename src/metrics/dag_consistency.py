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
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class DagScore:
    score: float          # 1.0 pass, 0.0 fail
    passed: bool
    reason: str
    probe_id: str = ""
    abstained: bool = False   # the model said it could not answer


# An answer that declines to answer is not an assertion of the current value —
# even though it usually repeats the question, and the question contains the
# expected value. Without this, "The provided facts don't include Acme Capital
# Partners LLP's full legal name" scores 1.0, which flatters precisely the
# backends that dropped the fact.
#
# Scoring an abstention 0.0 is the right semantics here: the question is whether
# the memory layer still held the fact, and it did not. `abstained` is reported
# separately so a backend that admits ignorance can be told apart from one that
# confidently states a stale value — the first is safer in production.
_ABSTENTION_PATTERNS = [
    r"do(?:es)?\s*n[o']t\s+(?:include|contain|specify|mention|state|provide)",
    r"not\s+(?:included|contained|specified|mentioned|stated|provided|available)",
    r"(?:no|insufficient)\s+(?:information|facts|data|record)",
    r"(?:can|could)\s*n[o']t\s+(?:determine|tell|say|answer)",
    r"unable\s+to\s+(?:determine|tell|say|answer)",
    r"i\s+do\s*n[o']t\s+(?:know|have)",
]


def _abstains(answer: str) -> bool:
    return any(re.search(p, answer) for p in _ABSTENTION_PATTERNS)


_VERSION_RE = re.compile(r"v\d+")

# Polarity words, tried in this order; the earliest match in the answer wins.
#
# Deliberately NOT included: "prohibited" and "allowed". Both appear verbatim in
# the suitability-policy value text ("v2 (crypto prohibited ...)" / "v1 (crypto
# allowed ...)"), so an answer that merely quotes the governing policy would
# assert a polarity it never stated.
#
# The negative compliance forms come first so a bare "compliant" can never win
# from inside "non-compliant" or "not compliant".
_POLARITY_PATTERNS = [
    (r"non[-\s]?compliant", "no"),
    (r"not\s+compliant", "no"),
    (r"not\s+permitted", "no"),
    (r"negative", "no"),
    # "no" as a verdict, not as part of "no longer holds crypto".
    (r"no(?!\s+(?:longer|doubt|change|further|more|other))", "no"),
    (r"affirmative", "yes"),
    (r"compliant", "yes"),
    (r"permitted", "yes"),
    (r"yes", "yes"),
]


def _bounded(pattern: str) -> str:
    """Word boundary that also treats '-' as part of a word, so 'compliant'
    does not match inside 'non-compliant' (plain \\b matches at the hyphen)."""
    return r"(?<![\w-])(?:" + pattern + r")(?![\w-])"


def _answer_polarity(answer: str) -> Optional[str]:
    """The yes/no verdict an answer states, or None if it states neither."""
    best = None  # (position, pattern order, polarity)
    for order, (pattern, polarity) in enumerate(_POLARITY_PATTERNS):
        match = re.search(_bounded(pattern), answer)
        if match is None:
            continue
        candidate = (match.start(), order, polarity)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    return best[2] if best else None


def _split_polarity(value: str) -> Tuple[Optional[str], str]:
    """Split 'no (v2 (crypto prohibited ...))' into ('no', 'v2 (crypto ...)').

    The polarity carries the verdict and the remainder cites the governing
    policy; both have to hold, and the polarity must never be droppable.
    """
    if value in ("yes", "no"):
        return value, ""
    match = re.match(r"^(yes|no)\b\s*(.*)$", value)
    if match:
        return match.group(1), match.group(2).strip()
    return None, value


def _has_all_tokens(answer: str, phrase: str) -> bool:
    tokens = _significant_tokens(phrase)
    return bool(tokens) and all(
        re.search(_bounded(re.escape(token)), answer) for token in tokens
    )


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _asserts_value(answer: str, value: str) -> bool:
    """Does ``answer`` assert ``value``?

    - a value carrying a yes/no verdict ('no', or 'no (v2 (...))') requires the
      answer to state that same verdict, AND to cite the policy named after it.
      The verdict is never droppable: a wrong polarity fails even if every word
      of the citation is present.
    - otherwise require the value, or all its significant tokens, to appear as
      whole words.
    """
    a = _normalize(answer)
    v = _normalize(value)
    if not v:
        return False
    polarity, citation = _split_polarity(v)
    if polarity is not None:
        if _answer_polarity(a) != polarity:
            return False
        return True if not citation else _has_all_tokens(a, citation)
    if re.search(_bounded(re.escape(v)), a):
        return True
    return _has_all_tokens(a, v)


def _significant_tokens(value: str) -> List[str]:
    toks = re.findall(r"[a-z0-9]+", _normalize(value))
    stop = {"the", "a", "an", "of", "for", "to", "is", "and", "v", "clients", "tiered"}
    # Version tokens survive the length filter: 'v1' and 'v2' are two characters
    # but they are the whole point of a policy citation.
    sig = [t for t in toks
           if (t not in stop and len(t) > 2) or _VERSION_RE.fullmatch(t)]
    return sig or toks


def score_probe(probe: Dict, answer: str) -> DagScore:
    """Score one backend answer against one probe's temporal answer key."""
    pid = probe.get("probe_id", "")
    expected = probe.get("expected_answer", "")
    superseded = probe.get("superseded_values", []) or []

    # node 0 — did the model decline to answer? Checked before anything else,
    # because a refusal normally quotes the question, and the question contains
    # the expected value.
    if _abstains(_normalize(answer)):
        return DagScore(
            0.0, False, "abstains — the fact was not in the context", pid,
            abstained=True,
        )

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


def value_in_text(text: str, value: str) -> bool:
    """Is an answer-key value present in `text` — typically the retrieved context?

    Polarity is stripped first: a context states the policy, it never says
    "yes". Matching the compound form verbatim would fail on every multi-hop
    probe and report the fact as missing when it was there.

    The dashboard uses this to separate "the fact never arrived" from "the fact
    arrived and the answer was wrong anyway". Those send an engineer to
    different layers, so guessing between them is worse than not saying.
    """
    body = _normalize(text)
    _, citation = _split_polarity(_normalize(value))
    target = citation or _normalize(value)
    if not body or not target:
        return False
    if re.search(_bounded(re.escape(target)), body):
        return True
    return _has_all_tokens(body, target)
