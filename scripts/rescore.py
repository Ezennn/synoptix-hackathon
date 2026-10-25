"""Re-score an existing run log against the current scorer. No API calls.

A scorer fix is worthless if the stored results still carry the old verdicts —
the dashboard reads `scores` out of the log, not the scorer. This recomputes
them from the answers already recorded, so the model's words are untouched and
only the judgement changes.

    python scripts/rescore.py results/run_log.jsonl

Writes in place and records `rescored_at` on every row it changes, so a log can
always say whether its verdicts came from the run or from a later pass.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics.dag_consistency import score_probe  # noqa: E402


def rescore(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    changed = 0
    for row in rows:
        if row.get("status") != "ok" or not row.get("scores"):
            continue
        before = (row["scores"].get("consistency") or {}).get("value")
        result = score_probe(row["probe"], row.get("answer") or "")
        row["scores"]["consistency"] = {
            "value": result.score,
            "passed": result.passed,
            "abstained": getattr(result, "abstained", False),
            "reason": result.reason,
        }
        if before != result.score:
            changed += 1
            row["rescored_at"] = time.time()
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return {"rows": len(rows), "changed": changed}


def main(argv) -> int:
    if len(argv) < 2:
        print(__doc__.strip())
        return 2
    for name in argv[1:]:
        path = Path(name)
        if not path.exists():
            print("missing: {}".format(path))
            return 1
        stats = rescore(path)
        print("{}: {rows} rows, {changed} verdicts changed".format(path, **stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
