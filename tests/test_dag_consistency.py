"""Regression tests for the state-consistency scorer.

Each case here is a verified defect: an answer that scored 1.0 while being
wrong, or 0.0 while being right. They exist so the scorer cannot quietly go
back to passing wrong answers — which is worse than failing right ones,
because it flatters every backend at once.

    pytest tests/test_dag_consistency.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics.dag_consistency import score_probe  # noqa: E402

POLICY_V2 = "v2 (crypto prohibited for clients tiered conservative)"
POLICY_V1 = "v1 (crypto allowed for professional clients up to 10% AUM)"


def probe(expected, superseded=None, pid="t0001"):
    return {
        "probe_id": pid,
        "expected_answer": expected,
        "superseded_values": superseded or [],
    }


# --- defect 1: the policy text collided with the yes/no synonym set ---------

def test_quoting_prohibited_policy_does_not_assert_no():
    """'prohibited' appears verbatim in policy v2, so quoting the policy used
    to assert "no" even while the answer said yes."""
    answer = "Crypto is prohibited, so the client still has exposure: yes."
    assert score_probe(probe("no"), answer).score == 0.0


def test_not_compliant_does_not_assert_yes():
    """\\b matches at a hyphen, so 'compliant' matched inside 'non-compliant'."""
    assert score_probe(probe("yes"), "Not compliant.").score == 0.0
    assert score_probe(probe("yes"), "The position is non-compliant.").score == 0.0
    assert score_probe(probe("no"), "Not compliant.").score == 1.0


# --- defect 2: polarity was droppable on multi-hop answers ------------------

def test_wrong_polarity_fails_even_when_citation_matches():
    """Verified failure: this scored 1.0. Every citation token is present, but
    the verdict is the opposite of the key."""
    answer = (
        "YES — holding crypto IS compliant. Governing policy: " + POLICY_V2 + "."
    )
    assert score_probe(probe("no (%s)" % POLICY_V2), answer).score == 0.0


def test_correct_polarity_with_citation_passes():
    answer = "No — holding crypto is non-compliant under " + POLICY_V2 + "."
    assert score_probe(probe("no (%s)" % POLICY_V2), answer).score == 1.0


def test_bare_polarity_without_citation_fails():
    """'no' alone does not cite the governing policy the key names."""
    assert score_probe(probe("no (%s)" % POLICY_V2), "No.").score == 0.0


def test_wrong_policy_version_fails():
    """'v1'/'v2' are two characters and were dropped by the length filter, so
    citing the wrong policy version used to pass."""
    answer = "No — crypto is not permitted. Governing policy: " + POLICY_V1 + "."
    assert score_probe(probe("no (%s)" % POLICY_V2), answer).score == 0.0


# --- defect 3: 'no' matched inside idioms ----------------------------------

def test_no_longer_is_not_a_verdict():
    answer = "The client no longer holds crypto, so exposure is yes."
    assert score_probe(probe("yes"), answer).score == 1.0


# --- unchanged behaviour: plain values and the stale check ------------------

def test_plain_value_matching():
    assert score_probe(probe("conservative"), "Conservative.").score == 1.0
    assert score_probe(probe("conservative"), "Aggressive.").score == 0.0


def test_asserting_both_current_and_superseded_fails():
    answer = "Conservative now, though previously aggressive."
    assert score_probe(probe("conservative", ["aggressive"]), answer).score == 0.0


def test_whole_word_matching_for_plain_values():
    """Substring matching would pass 'Luxembourgish' for 'Luxembourg'."""
    assert score_probe(probe("luxembourg"), "The domicile is Luxembourg.").score == 1.0
