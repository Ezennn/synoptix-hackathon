"""Compare the two arms of the recency-hint experiment.

    python scripts/analyse_ab.py

Reads the two committed Opus arms — results/run_log.opus.jsonl (hint ON) and
results/run_log.opus.recency-off.jsonl (hint OFF) — same 100 probes, same 3
backends, same model, identical prompts apart from one sentence: "If facts
conflict, prefer the most recent." Pass --haiku for the earlier Haiku pair.

Three readings, weakest to strongest:

1. Consistency per arm. Shows the size of the effect.
2. Accuracy split by whether a stale fact reached the model. Suggestive only:
   the two buckets hold different probes, so they are not matched on difficulty.
3. The paired test. Every probe appears in both arms for the same backend, so
   a flip is attributable to the one sentence. If the hint works by resolving
   contradictions, the breakages should concentrate on probes whose context
   held a stale fact.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ui.replay import has_stale, load_log, scoreboard  # noqa: E402

def _arms(argv):
    """Opus is the committed pair; --haiku reads the earlier smaller-model run."""
    results = ROOT / "results"
    if "--haiku" in argv:
        return {
            "hint ON": results / "run_log.recency-on.jsonl",
            "hint OFF": results / "run_log.recency-off.jsonl",
        }
    # The ON arm is the deck's headline run, so it keeps its own name rather
    # than being duplicated under a .recency-on alias.
    return {
        "hint ON": results / "run_log.opus.jsonl",
        "hint OFF": results / "run_log.opus.recency-off.jsonl",
    }


ARMS = _arms(sys.argv)
BACKENDS = ["sanity", "langgraph_state", "litellm_trim"]


def _score(record):
    scores = record.get("scores") or {}
    return (scores.get("consistency") or {}).get("value")


def main() -> int:
    missing = [str(p) for p in ARMS.values() if not p.exists()]
    if missing:
        print("missing run log(s): {}".format(", ".join(missing)))
        print("run scripts/run_benchmark.py for each arm first")
        return 1
    data = {name: load_log(path) for name, path in ARMS.items()}

    print("1. CONSISTENCY PER ARM")
    print("{:18}{:>10}{:>10}{:>8}".format("backend", "hint ON", "hint OFF", "delta"))
    for backend in BACKENDS:
        rates = {}
        for arm, records in data.items():
            row = next(r for r in scoreboard(records) if r["backend"] == backend)
            rates[arm] = row["consistency_rate"] or 0.0
        print("{:18}{:>9.0f}%{:>9.0f}%{:>8.0f}".format(
            backend, rates["hint ON"] * 100, rates["hint OFF"] * 100,
            (rates["hint OFF"] - rates["hint ON"]) * 100))

    print("\n2. ACCURACY BY WHETHER A STALE FACT REACHED THE MODEL")
    print("   (not matched on probe difficulty — suggestive only)")
    print("{:18}{:>24}{:>24}".format("backend", "hint ON: +stale|-stale",
                                     "hint OFF: +stale|-stale"))
    for backend in BACKENDS:
        cells = []
        for arm, records in data.items():
            buckets = {True: [], False: []}
            for record in records:
                if record["backend"] != backend or record["status"] != "ok":
                    continue
                flag = has_stale(record)
                if flag is None:
                    continue
                buckets[flag].append(_score(record) or 0.0)
            fmt = lambda v: "—" if not v else "{:.0f}%(n={})".format(
                100 * sum(v) / len(v), len(v))
            cells.append("{}|{}".format(fmt(buckets[True]), fmt(buckets[False])))
        print("{:18}{:>24}{:>24}".format(backend, cells[0], cells[1]))

    print("\n3. PAIRED: what the one sentence actually changed")
    on = {(r["backend"], r["probe"]["probe_id"]): r for r in data["hint ON"]}
    off = {(r["backend"], r["probe"]["probe_id"]): r for r in data["hint OFF"]}
    print("{:18}{:>11}{:>11}{:>22}".format(
        "backend", "ok->wrong", "wrong->ok", "of ok->wrong, stale"))
    broke = stale_broke = 0
    for backend in BACKENDS:
        fell = fixed = stale_fell = 0
        for key, record in on.items():
            if key[0] != backend or key not in off:
                continue
            before, after = _score(record), _score(off[key])
            if before is None or after is None:
                continue
            if before == 1.0 and after == 0.0:
                fell += 1
                if has_stale(off[key]):
                    stale_fell += 1
            elif before == 0.0 and after == 1.0:
                fixed += 1
        broke += fell
        stale_broke += stale_fell
        print("{:18}{:>11}{:>11}{:>22}".format(
            backend, fell, fixed, "{}/{}".format(stale_fell, fell) if fell else "—"))

    if broke:
        print("\n{} probes broke when the sentence was removed; {} ({:.0f}%) had a "
              "stale fact in context.".format(broke, stale_broke,
                                              100 * stale_broke / broke))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
