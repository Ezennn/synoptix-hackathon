"""Produce a rehearsal run log without A's engine or any network. Agent B.

The stage demo replays results/run_log.jsonl, so the UI needs a log to exist
before the real sweep does. This drives the fixture backends through a small
four-phase scenario and records the unavailable backends too, so the degraded
column states are visible in rehearsal rather than discovered live.

    python scripts/seed_demo_log.py

Replace with the real sweep output once A's engine lands.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluator import RUN_LOG, Probe, log_turn, score_live  # noqa: E402
from src.registry import SPECS, status  # noqa: E402

SESSION = "client_acme"

JANUARY = ("Jan 2026 review: client is crypto-heavy, roughly 40% of the portfolio "
           "in digital assets, and describes an aggressive risk appetite.")
MARCH = ("Mar 2026 call: client has fully divested from crypto after a family "
         "illness and now describes a conservative risk appetite.")

PROBES = [
    Probe("p1", "What is this client's current risk appetite?",
          fact_id="risk_appetite", expected_answer="conservative",
          superseded_answer="aggressive", phase=2),
    Probe("p2", "Is this client suitable for a high-risk crypto ETF?",
          fact_id="crypto_exposure", expected_answer="fully divested",
          superseded_answer="crypto-heavy", phase=2),
    Probe("p3", "What proportion of the portfolio is in digital assets?",
          fact_id="crypto_exposure", expected_answer="fully divested",
          superseded_answer="40%", phase=3),
    Probe("p4", "Should we flag this client for a suitability review?",
          fact_id="risk_appetite", expected_answer="conservative",
          superseded_answer="aggressive", phase=4),
]


def main() -> None:
    if RUN_LOG.exists():
        RUN_LOG.unlink()

    reasons = {s["name"]: s["reason"] for s in status()}
    live = [s for s in SPECS if s.available()]
    for spec in live:
        backend = spec.get()
        backend.reset(SESSION)
        backend.load(SESSION, [JANUARY])      # phase 1: ingest
        backend.update(SESSION, MARCH)        # phase 2: the contradiction

    for probe in PROBES:
        for spec in SPECS:
            if not spec.available():
                log_turn(SESSION, spec.name, probe, None, None,
                         "unavailable", reasons[spec.name])
                continue
            turn = spec.get().ask(SESSION, probe.question)
            log_turn(SESSION, spec.name, probe, turn, score_live(turn, probe), "ok")

    print("wrote {} ({} lines)".format(RUN_LOG, sum(1 for _ in RUN_LOG.open())))


if __name__ == "__main__":
    main()
