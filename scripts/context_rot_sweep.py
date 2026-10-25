"""Context-rot sweep — the money chart's data (README "context rot breakdown").

Measures how each backend's state-consistency decays as the conversation gets
longer. For each target depth in DEPTHS, it rebuilds the real four-session
scenario and then PADS the history with plausible-but-stale distractor turns up
to that depth, so the supporting fact a probe needs is buried under growing
clutter. It then fires the memory-sensitive probes (direct / partial /
transitive / reinstatement — NOT baseline, which never changes) and records the
consistency rate at that depth.

Hypothesis the chart should show:
- sanity holds flat but at rising cost (it keeps everything);
- litellm-trim falls off a cliff once the needle scrolls past its token budget;
- langgraph degrades gracefully (its reducer keeps latest-per-key);
- zep/letta (when live) stay high (temporal graph / paging).

    python scripts/context_rot_sweep.py          # stub mode
    REGWATCH_LLM_MODE=live python scripts/context_rot_sweep.py

Writes results/context_rot.jsonl: one line per (backend, depth) with the rate.
Then plot with figures/plot_context_rot.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics.dag_consistency import score_probe  # noqa: E402
from src.registry import SPECS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
OUT = RESULTS / "context_rot.jsonl"
SESSION_ID = "client_acme"

DEPTHS = [5, 20, 50, 100]          # conversation turns, per the README
MEMORY_TIERS = {"direct", "partial", "transitive", "reinstatement"}


def load_sessions():
    out = {}
    for p in sorted((DATA / "sessions").glob("session_*.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        out[int(s["session"])] = s
    return out


def load_probes():
    return json.loads((DATA / "probes.json").read_text(encoding="utf-8"))["probes"]


def _distractor(i: int) -> str:
    """A plausible-but-irrelevant advisory turn used to pad history depth."""
    topics = ["liquidity ladder", "FX hedging band", "custody arrangement",
              "quarterly rebalance", "fee schedule", "ESG screen", "cash buffer"]
    t = topics[i % len(topics)]
    return ("Routine note #%d: the team reviewed the %s and confirmed it is "
            "unchanged this period; no action required." % (i, t))


def build_history(sessions, depth: int):
    """Real scenario statements first (the signal), then pad with distractors
    (the clutter) until the history is `depth` turns long. Returns an ordered
    list of (kind, text) where kind is 'load' or 'update'."""
    turns = []
    for n in sorted(sessions):
        stmts = [st["text"] for st in sessions[n].get("statements", [])]
        for s in stmts:
            turns.append(("load" if n == 1 else "update", s))
    i = 0
    while len(turns) < depth:
        turns.append(("load", _distractor(i)))
        i += 1
    return turns[:depth]


def replay(backend, turns):
    backend.reset(SESSION_ID)
    # first load carries the batch of leading 'load' turns; updates applied singly
    loads = [t for k, t in turns if k == "load"]
    if loads:
        backend.load(SESSION_ID, loads)
    for k, t in turns:
        if k == "update":
            backend.update(SESSION_ID, t)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    if OUT.exists():
        OUT.unlink()

    sessions = load_sessions()
    probes = [p for p in load_probes() if p.get("tier") in MEMORY_TIERS]
    available = [s for s in SPECS if s.available()]
    print("sweep: %d memory-sensitive probes x %d depths x %d backends"
          % (len(probes), len(DEPTHS), len(available)))

    with OUT.open("w", encoding="utf-8") as fh:
        for spec in available:
            backend = spec.get()
            for depth in DEPTHS:
                turns = build_history(sessions, depth)
                replay(backend, turns)
                passed = total = 0
                for p in probes:
                    ans = backend.ask(SESSION_ID, p["question"]).answer
                    s = score_probe(p, ans)
                    passed += int(s.passed)
                    total += 1
                rate = round(passed / (total or 1), 4)
                fh.write(json.dumps({
                    "backend": spec.name, "depth": depth,
                    "consistency_rate": rate, "passed": passed, "total": total,
                }) + "\n")
                print("  %-16s depth=%3d  rate=%.2f (%d/%d)"
                      % (spec.name, depth, rate, passed, total))

    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
