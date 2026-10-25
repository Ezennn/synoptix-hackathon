"""End-to-end benchmark run driver — now a thin shim over run_benchmark.py.

This script and scripts/run_benchmark.py were written in parallel on two
branches and did the same job: walk the four phases, fire the 100 probes at
their correct ask-times, score with the evaluator, write results/run_log.jsonl.
Two runners for one job is not a convenience, it is a hazard — the original
version here deleted the run log on start and wrote without the concurrency
lock or the provenance guard, which is exactly how one run log ended up with
two processes' rows interleaved and untanglable.

So this entry point is kept (the command still works, and the ask-time replay
it described is what run_benchmark.py does) but the implementation is the
guarded one:

  * refuses to start while another run holds the lock
  * refuses to append to a log from a different mode, model or prompt arm
  * records per-row cache status, so cached reads are never counted as fresh
    measurements of latency or spend
  * counts ingestion calls separately from answer generation
  * logs unavailable backends as rows, so degraded columns still render
    (that behaviour came from this file and was carried across)

    python scripts/run_phases.py --mode live --model anthropic/claude-opus-5-5

See run_benchmark.py --help for the full set of flags.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_benchmark import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
