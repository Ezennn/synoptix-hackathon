"""Export the benchmark's INPUTS to human-readable files, so the full set of
questions and the exact data fed to the models is stored, not just buried in
data/probes.json and data/sessions/*.json.

    python scripts/export_inputs.py

Writes:
  results/question_log.csv  — every probe: id, phase, tier, scenario, ask step,
                              ask date, question, expected answer, superseded
                              values, fact ids, multi-hop flag. One row per
                              probe, so the "same question at a later timestep"
                              structure is visible at a glance.
  results/data_feed.md      — the four session transcripts: every statement and
                              distractor fed into the backends, in order, which
                              is exactly what each memory system ingested.

Pure read of data/; makes no model calls.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"

# tier -> scenario title, mirroring app.SCENARIOS (kept in sync by hand; the two
# non-baseline storylines beyond the 4 map to their tier).
TIER_SCENARIO = {
    "baseline": "— (control: name / currency)",
    "direct": "1 · Crypto risk appetite (+ regulation version)",
    "partial": "2 · Scoped account risk tiers",
    "transitive": "3 · Subsidiary divestment → crypto exposure",
    "reinstatement": "4 · Tax domicile reinstatement",
}


def export_question_log() -> Path:
    probes = json.loads((DATA / "probes.json").read_text())["probes"]
    probes = sorted(probes, key=lambda p: p["probe_id"])
    path = OUT / "question_log.csv"
    OUT.mkdir(exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "probe_id", "phase", "tier", "scenario", "ask_after_session",
            "ask_date", "multi_hop", "question", "expected_answer",
            "superseded_values", "fact_ids", "notes",
        ])
        for p in probes:
            writer.writerow([
                p.get("probe_id"), p.get("phase"), p.get("tier"),
                TIER_SCENARIO.get(p.get("tier"), p.get("tier")),
                p.get("ask_after_session"), p.get("ask_date"),
                "yes" if p.get("multi_hop") else "no",
                p.get("question"), p.get("expected_answer"),
                " | ".join(p.get("superseded_values") or []),
                " ".join(p.get("fact_ids") or []), p.get("notes") or "",
            ])
    return path


def export_data_feed() -> Path:
    path = OUT / "data_feed.md"
    OUT.mkdir(exist_ok=True)
    dates = {1: "2026-01-10", 2: "2026-03-15", 3: "2026-04-20", 4: "2026-06-01"}
    lines = [
        "# Data feed — what every backend ingested",
        "",
        "The four session transcripts, in order. Session 1 is loaded; sessions "
        "2–4 are applied as updates. `distractors` are plausible but irrelevant "
        "notes added to make retrieval harder — they are fed in too.",
        "",
    ]
    for i in (1, 2, 3, 4):
        session = json.loads((DATA / "sessions" / "session_{}.json".format(i)).read_text())
        lines.append("## Session {} — {}".format(i, dates.get(i, "")))
        lines.append("")
        lines.append("**Statements (the real facts):**")
        lines.append("")
        for s in session.get("statements", []):
            lines.append("- `{}` [{}] — {}".format(
                s.get("fact_id"), s.get("tier"), s.get("text")))
        distractors = session.get("distractors", [])
        if distractors:
            lines.append("")
            lines.append("**Distractors (noise fed in alongside):**")
            lines.append("")
            for d in distractors:
                text = d if isinstance(d, str) else d.get("text", "")
                about = "" if isinstance(d, str) else " (about {})".format(
                    d.get("about_fact_id", "?"))
                lines.append("- {}{}".format(text, about))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    q = export_question_log()
    f = export_data_feed()
    probes = json.loads((DATA / "probes.json").read_text())["probes"]
    distinct = len({(p["question"], p["ask_after_session"], tuple(p.get("fact_ids") or []))
                    for p in probes})
    print("wrote {} ({} probe rows, {} distinct tests)".format(
        q, len(probes), distinct))
    print("wrote {}".format(f))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
