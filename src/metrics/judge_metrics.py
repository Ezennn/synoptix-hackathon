"""Axes 2 & 3 — the order-free, memory-revealing judge metrics.

These score the RETRIEVAL step (what the memory layer surfaced), not the LLM
that wrote the answer. The answer model is identical across every backend
(base.py rule 4), so any spread in these numbers can only come from the memory
policy — which is the whole point of the benchmark.

Deliberately NOT used: Contextual *Precision* (a RANKING metric) and
Faithfulness (a GENERATOR metric).
  - Precision assumes the backend returns a relevance-RANKED list. Our
    rule-based backends (trim/summary, reduce) return chronological sets, not
    ranked ones, so a ranking score would be noise. We use order-free set
    metrics instead.
  - Faithfulness measures the answer LLM, which is constant here, so it would
    not separate the columns. We expose it as None, not as a headline number.

Metrics we DO use (both order-independent):
  Axis 2 — Contextual Relevancy (DeepEval ``ContextualRelevancyMetric``):
      of the retrieved chunks, what fraction are relevant to the question?
      A set measure (relevant / total). litellm-summary injecting 3 chunks of
      which 1 helps scores ~0.33; sanity injecting 40 of which 2 help scores
      ~0.05. Rewards a clean, low-noise memory.
  Axis 3 — Contextual Recall (DeepEval ``ContextualRecallMetric``):
      of the claims in the expected answer, how many are supported by something
      in the retrieved context? The money metric — a trimmer that dropped the
      superseding fact scores low; a graph that kept it scores high. Order does
      not matter, only presence.

Both are LLM-judged. The judge is held FIXED across backends (fairness) and is
chosen cheap via REGWATCH_JUDGE_MODEL (default Claude 3.5 Haiku). DeepEval is
imported lazily and inside a guard: on a box without it (or without a judge
key), this module reports unavailable and scores nothing, exactly like the
dag_consistency fallback — a missing judge can never look like a passing
backend.

Python target: 3.9 (``from __future__ import annotations`` + typing generics).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

# Axis 2 stores Contextual Relevancy under the existing DB column
# ``context_precision`` (both are "axis 2: how clean is the retrieved context",
# range 0..1) so no schema migration is needed. The emitted score dict also
# carries the honest key ``contextual_relevancy`` for anyone reading the JSON.
_JUDGE_DEFAULT = os.environ.get("REGWATCH_JUDGE_MODEL", "anthropic/claude-3-5-haiku-20241022")


def judge_available() -> Dict[str, Any]:
    """Can the judge metrics run here? Cheap, import-only probe.

    Returns {"ok": bool, "reason": str}. Never raises — callers branch on it so
    a missing dependency degrades to 'unscored', not a crash.
    """
    try:
        import deepeval  # type: ignore  # noqa: F401
        from deepeval.metrics import (  # type: ignore  # noqa: F401
            ContextualRecallMetric,
            ContextualRelevancyMetric,
        )
    except Exception as exc:  # ImportError or a broken partial install
        return {"ok": False, "reason": "deepeval not installed: {}".format(exc)}
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        return {"ok": False, "reason": "no judge key (ANTHROPIC_API_KEY / OPENAI_API_KEY)"}
    return {"ok": True, "reason": "ready", "judge_model": _JUDGE_DEFAULT}


def _make_judge(model: Optional[str] = None):
    """Build the DeepEval judge model wrapper. Imported lazily by callers that
    already confirmed judge_available()['ok']."""
    from deepeval.models import LiteLLMModel  # type: ignore

    return LiteLLMModel(model=model or _JUDGE_DEFAULT)


def score_record(
    record: Dict[str, Any],
    judge=None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Score ONE run-log record on axes 2 and 3.

    ``record`` is a parsed line of results/run_log.jsonl: it must carry
    ``answer``, ``retrieval_context`` (list[str]) and ``probe.question`` +
    ``probe.expected_answer``. Returns a dict with:

        contextual_relevancy : float|None   -> stored in column context_precision
        contextual_recall    : float|None
        reason               : str

    A record that cannot be scored (no retrieval_context, no expected_answer,
    or a judge error) returns None values with a reason — never a fabricated
    number.
    """
    probe = record.get("probe") or {}
    question = probe.get("question")
    expected = probe.get("expected_answer")
    answer = record.get("answer")
    ctx: List[str] = record.get("retrieval_context") or []

    # Cheap guards first — these need no judge and must not require deepeval to
    # be importable (so a caller can pre-filter records without the dep).
    if not ctx:
        return {"contextual_relevancy": None, "contextual_recall": None,
                "reason": "no retrieval_context (backend did not expose surfaced facts)"}
    if not question or answer is None:
        return {"contextual_relevancy": None, "contextual_recall": None,
                "reason": "record missing question/answer"}

    from deepeval.metrics import (  # type: ignore
        ContextualRecallMetric,
        ContextualRelevancyMetric,
    )
    from deepeval.test_case import LLMTestCase  # type: ignore

    judge = judge or _make_judge()
    out: Dict[str, Any] = {"contextual_relevancy": None, "contextual_recall": None,
                           "reason": "ok"}

    # Axis 2 — Contextual Relevancy: needs input + actual_output + retrieval_context.
    try:
        tc = LLMTestCase(input=question, actual_output=answer or "",
                         retrieval_context=ctx)
        m = ContextualRelevancyMetric(threshold=threshold, model=judge,
                                      include_reason=False)
        m.measure(tc)
        out["contextual_relevancy"] = float(m.score)
    except Exception as exc:  # judge error on this probe must not kill the pass
        out["reason"] = "relevancy error: {}".format(exc)

    # Axis 3 — Contextual Recall: ALSO needs expected_output (the ground truth
    # whose claims we check are present in the retrieved context).
    if expected:
        try:
            tc = LLMTestCase(input=question, actual_output=answer or "",
                             expected_output=expected, retrieval_context=ctx)
            m = ContextualRecallMetric(threshold=threshold, model=judge,
                                       include_reason=False)
            m.measure(tc)
            out["contextual_recall"] = float(m.score)
        except Exception as exc:
            out["reason"] = (out["reason"] + "; " if out["reason"] != "ok" else "") + \
                "recall error: {}".format(exc)
    else:
        out["reason"] = (out["reason"] + "; " if out["reason"] != "ok" else "") + \
            "no expected_answer -> recall unscored"

    return out
