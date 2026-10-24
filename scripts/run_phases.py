"""End-to-end benchmark run driver (PLAN sync point 3).

The seam between A's engine and B's service: it conducts the ACTUAL benchmark.
Walks the four phases in chronological order, feeds each session's statements
(and phase-4 distractors) to every available backend, fires A's 100 probes at
each probe's correct ask-time, scores each answer with B's evaluator, and writes
``results/run_log.jsonl`` — the exact file the replay UI and the offline scorer
already consume. Replaces scripts/seed_demo_log.py's fixture output with real
numbers.

    python scripts/run_phases.py                 # all available backends, stub mode
    REGWATCH_LLM_MODE=live python scripts/run_phases.py   # real model

Design:
- Backends and ask-times come from A's data, so a probe asked after session 2
  only ever sees sessions 1-2. That is what makes phase-3 multi-hop and the
  reinstatement tier meaningful: a backend is judged on the state as of ask-time.
- A's probe schema (fact_ids/expected_answer/superseded_values/ask_after_session)
  is converted to B's Probe dataclass here, in ONE place, so the field-name
  reconciliation lives at the seam and nowhere else.
- Unavailable backends are logged as 'unavailable' (not skipped), so degraded
  columns show in the UI rather than vanishing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluator import Probe, log_turn, score_live  # noqa: E402
from src.registry import SPECS, status  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SESSION_ID = "client_acme"


def load_sessions():
    out = {}
    for p in sorted((DATA / "sessions").glob("session_*.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        out[int(s["session"])] = s
    return out


def load_probes():
    return json.loads((DATA / "probes.json").read_text(encoding="utf-8"))["probes"]


def to_b_probe(p: dict) -> Probe:
    """A's probe dict -> B's Probe dataclass (the one field-mapping site)."""
    fact_ids = p.get("fact_ids") or []
    superseded = p.get("superseded_values") or []
    return Probe(
        probe_id=p["probe_id"],
        question=p["question"],
        fact_id=fact_ids[0] if fact_ids else None,
        expected_answer=p.get("expected_answer"),
        superseded_answer=superseded[0] if superseded else None,
        asked_at=p.get("ask_date"),
        phase=p.get("phase"),
    )


def session_texts(sess: dict):
    """All statements + phase-4 distractors for one session, in file order."""
    texts = [st["text"] for st in sess.get("statements", [])]
    texts += [d["text"] for d in sess.get("distractors", [])]
    return texts


def feed_up_to(backend, sessions, upto: int):
    """Replay sessions 1..upto into a freshly-reset backend.

    Statements are loaded; the first session is phase-1 ingest, later sessions'
    statements are the phase-2 mutations — the backend sees them as update()s so
    a temporal backend can reconcile. Distractors are always load()ed as noise.
    """
    backend.reset(SESSION_ID)
    for n in range(1, upto + 1):
        sess = sessions[n]
        stmts = [st["text"] for st in sess.get("statements", [])]
        distract = [d["text"] for d in sess.get("distractors", [])]
        if n == 1:
            backend.load(SESSION_ID, stmts)
        else:
            for s in stmts:
                backend.update(SESSION_ID, s)
        if distract:
            backend.load(SESSION_ID, distract)


def main() -> None:
    from src.evaluator import RUN_LOG
    if RUN_LOG.exists():
        RUN_LOG.unlink()

    sessions = load_sessions()
    probes = load_probes()
    reasons = {s["name"]: s["reason"] for s in status()}
    available = [s for s in SPECS if s.available()]

    print("backends available: %s" % ", ".join(s.name for s in available))
    print("probes: %d   sessions: %d" % (len(probes), len(sessions)))

    n_ok = 0
    for spec in SPECS:
        if not spec.available():
            # one unavailable row per probe so the column renders as degraded
            for p in probes:
                log_turn(SESSION_ID, spec.name, to_b_probe(p), None, None,
                         "unavailable", reasons[spec.name])
            continue

        backend = spec.get()
        # group probes by ask-time so we replay history once per ask depth
        for p in probes:
            upto = int(p.get("ask_after_session", len(sessions)))
            upto = max(1, min(upto, len(sessions)))
            feed_up_to(backend, sessions, upto)
            bp = to_b_probe(p)
            try:
                turn = backend.ask(SESSION_ID, p["question"])
                log_turn(SESSION_ID, spec.name, bp, turn, score_live(turn, bp), "ok")
                n_ok += 1
            except Exception as exc:  # a backend failure is a result, not a gap
                log_turn(SESSION_ID, spec.name, bp, None, None, "error", str(exc))

    print("wrote %s (%d scored ok, %d total lines)"
          % (RUN_LOG, n_ok, sum(1 for _ in RUN_LOG.open(encoding="utf-8"))))


if __name__ == "__main__":
    main()
