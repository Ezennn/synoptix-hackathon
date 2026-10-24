"""Run the benchmark: every probe, every backend, into results/run_log.jsonl.

Replays the session timeline the way docs/ENGINE.md specifies — reset, load
session 1, update sessions 2-4, asking each session's probes after that session
— so a probe asked after session 1 sees only what was known then.

    python scripts/run_benchmark.py --limit 3            # smoke test
    python scripts/run_benchmark.py --mode live --model anthropic/claude-haiku-5-5

Reuses src/registry.py for backends and src/evaluator.py for scoring and
logging; it adds no scoring path of its own.

IMPORTANT: src/llm.py reads REGWATCH_MODEL and REGWATCH_CACHE_DIR at import and
get_gateway() memoises a singleton, so this script sets the environment from its
arguments BEFORE importing anything under src/. That is why the imports live
inside main() rather than at the top of the file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKENDS = ["sanity", "litellm_trim", "langgraph_state"]
SESSION = "acme"


def _load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines from .env without printing anything.

    Existing environment variables win, so an explicit export always beats the
    file.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and value and key not in os.environ:
            os.environ[key] = value


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--backends", nargs="+", default=DEFAULT_BACKENDS,
                   help="backend names from src/registry.py")
    p.add_argument("--limit", type=int, default=None,
                   help="probes per session, so --limit 2 asks 8 per backend "
                        "across the 4 sessions (smoke test)")
    p.add_argument("--mode", choices=("stub", "live"), default="stub",
                   help="stub is offline and free; live calls the model")
    p.add_argument("--model", default=None,
                   help="LiteLLM model string, e.g. anthropic/claude-haiku-5-5")
    p.add_argument("--cache-dir", default=None,
                   help="gateway cache. Use a fresh dir to measure real latency: "
                        "cached turns return in ~0 ms")
    p.add_argument("--dry-run", action="store_true",
                   help="show the plan and make no calls")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # --- environment first, before any src import -------------------------
    _load_dotenv(ROOT / ".env")
    os.environ["REGWATCH_LLM_MODE"] = args.mode
    if args.model:
        os.environ["REGWATCH_MODEL"] = args.model
    if args.cache_dir:
        os.environ["REGWATCH_CACHE_DIR"] = args.cache_dir

    sys.path.insert(0, str(ROOT))
    from src.evaluator import (  # noqa: E402
        RUN_LOG, Probe, check_log_mode, log_turn, provenance, score_live,
    )
    from src.registry import get_spec  # noqa: E402

    probes = json.loads((ROOT / "data" / "probes.json").read_text())["probes"]
    sessions = [
        json.loads((ROOT / "data" / "sessions" / "session_{}.json".format(i)).read_text())
        for i in (1, 2, 3, 4)
    ]
    by_session = {}
    for record in probes:
        by_session.setdefault(record.get("ask_after_session", 1), []).append(record)
    if args.limit:
        for index in by_session:
            by_session[index] = by_session[index][: args.limit]

    planned = sum(len(v) for v in by_session.values())
    prov = provenance()
    print("mode={llm_mode}  model={model}  cache={cache_dir}".format(**prov))
    print("backends: {}".format(", ".join(args.backends)))
    print("probes per backend: {}  ->  {} calls".format(
        planned, planned * len(args.backends)))

    if args.dry_run:
        print("dry run: no calls made")
        return 0

    # Refuse to mix modes in one log before doing any work. An expected stop,
    # so report it plainly rather than as a traceback.
    try:
        check_log_mode(args.mode, RUN_LOG)
    except RuntimeError as exc:
        print("\nrefusing to run: {}".format(exc), file=sys.stderr)
        return 2

    started = time.time()
    totals = {}
    for name in args.backends:
        spec = get_spec(name)
        if not spec.available():
            print("  {:16} SKIPPED — not available".format(name))
            continue
        backend = spec.get()
        backend.reset(SESSION)
        ok = errors = 0

        for index, session in enumerate(sessions, start=1):
            texts = [s["text"] for s in session.get("statements", [])]
            texts += [d if isinstance(d, str) else d.get("text", "")
                      for d in session.get("distractors", [])]
            if index == 1:
                backend.load(SESSION, texts)
            else:
                for text in texts:
                    backend.update(SESSION, text)

            for record in by_session.get(index, []):
                probe = Probe.from_record(record)
                # One probe failing is a result, not the end of the run.
                try:
                    turn = backend.ask(SESSION, probe.question)
                    log_turn(SESSION, name, probe, turn, score_live(turn, probe), "ok")
                    ok += 1
                except Exception as exc:
                    log_turn(SESSION, name, probe, None, None, "error",
                             "{}: {}".format(type(exc).__name__, exc))
                    errors += 1
                done = ok + errors
                if done % 10 == 0 or done == planned:
                    print("  {:16} {}/{}".format(name, done, planned), flush=True)

        totals[name] = (ok, errors)
        print("  {:16} done — {} ok, {} errors".format(name, ok, errors))

    print("\nwrote {} in {:.0f}s".format(RUN_LOG, time.time() - started))
    for name, (ok, errors) in totals.items():
        print("  {:16} ok={:3} errors={}".format(name, ok, errors))
    return 1 if any(e for _, e in totals.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
