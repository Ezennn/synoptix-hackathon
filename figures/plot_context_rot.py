"""Plot the context-rot curve — accuracy vs conversation length per backend.

Reads results/context_rot.jsonl (from scripts/context_rot_sweep.py) and writes
figures/context_rot.png: one line per backend, consistency rate on the y-axis,
history depth on the x-axis. This is the single most persuasive slide — it shows
trimming collapsing while a temporal backend holds.

    python figures/plot_context_rot.py

Headless-safe (Agg backend); no display required.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "context_rot.jsonl"
OUT = ROOT / "figures" / "context_rot.png"


def main() -> None:
    if not SRC.exists():
        raise SystemExit("no %s — run scripts/context_rot_sweep.py first" % SRC)

    series = defaultdict(list)  # backend -> list of (depth, rate)
    for line in SRC.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        series[r["backend"]].append((r["depth"], r["consistency_rate"]))

    OUT.parent.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for backend in sorted(series):
        pts = sorted(series[backend])
        xs = [d for d, _ in pts]
        ys = [100 * rate for _, rate in pts]
        ax.plot(xs, ys, marker="o", linewidth=2, label=backend)

    ax.set_title("Context rot: state-consistency vs conversation length", fontsize=13)
    ax.set_xlabel("conversation depth (turns)")
    ax.set_ylabel("state-consistency rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(title="memory backend", loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
