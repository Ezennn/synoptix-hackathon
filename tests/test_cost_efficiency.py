"""Cost and efficiency aggregation.

The rule these tests defend: an unknown cost is never reported as zero, and a
cached turn is never counted as a fresh measurement. Both failures understate
cost and latency, which is the direction that flatters a result.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluator import price_for  # noqa: E402
from ui.replay import efficiency_pick  # noqa: E402


def board(*rows):
    out = []
    for backend, accuracy, tokens in rows:
        out.append({"backend": backend, "answered": 100, "probes": 100,
                    "consistency_rate": accuracy, "avg_tokens_in": tokens,
                    "cost_per_1k_usd": None})
    return out


# --- pricing ---------------------------------------------------------------

def test_unpriced_model_is_unknown_not_zero():
    # Any model absent from config/pricing.yaml, not a specific one: the entry
    # this test used to name (claude-opus-5-5) later got a confirmed rate, and
    # the invariant being guarded is "no entry means unknown", not which models
    # happen to be missing today.
    assert price_for("anthropic/claude-nonexistent-9-9") is None
    assert price_for(None) is None


def test_benchmark_model_is_priced_with_a_source():
    # The committed runs are on Opus; an unpriced benchmark model makes every
    # dollar figure on the dashboard read unknown.
    rate = price_for("anthropic/claude-opus-5-5")
    assert (rate["input_per_m"], rate["output_per_m"]) == (4.00, 20.00)
    assert rate["source"] and rate["as_of"]


def test_priced_model_carries_its_source():
    rate = price_for("gpt-4o-mini")
    assert rate["input_per_m"] == 0.15
    assert rate["source"]          # a number without a source is not auditable
    assert rate["as_of"]


def test_env_override_beats_the_table():
    os.environ["REGWATCH_PRICE_IN_PER_M"] = "2.0"
    os.environ["REGWATCH_PRICE_OUT_PER_M"] = "8.0"
    try:
        rate = price_for("gpt-4o-mini")
        assert (rate["input_per_m"], rate["output_per_m"]) == (2.0, 8.0)
    finally:
        del os.environ["REGWATCH_PRICE_IN_PER_M"]
        del os.environ["REGWATCH_PRICE_OUT_PER_M"]


# --- efficiency pick -------------------------------------------------------

def test_picks_cheapest_that_meets_the_bar():
    pick = efficiency_pick(board(("rich", 0.92, 1000), ("mid", 0.85, 400),
                                 ("thin", 0.22, 200)), 0.80)
    assert pick["chosen"]["backend"] == "mid"      # cheapest of the two passing


def test_reports_when_nothing_is_shippable():
    pick = efficiency_pick(board(("rich", 0.70, 1000), ("thin", 0.22, 200)), 0.80)
    assert pick["chosen"] is None
    assert "no configuration" in pick["reason"]
    # Every candidate still shown, each marked against the bar.
    assert [c["meets"] for c in pick["candidates"]] == [False, False]


def test_unscored_backends_are_not_candidates():
    rows = board(("rich", 0.92, 1000))
    rows.append({"backend": "broken", "answered": 0, "probes": 100,
                 "consistency_rate": None, "avg_tokens_in": 0,
                 "cost_per_1k_usd": None})
    pick = efficiency_pick(rows, 0.80)
    assert [c["backend"] for c in pick["candidates"]] == ["rich"]
