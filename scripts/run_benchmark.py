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
from typing import Optional

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
    p.add_argument("--no-recency-hint", action="store_true",
                   help="drop 'If facts conflict, prefer the most recent' from "
                        "the system prompt — the B arm of that experiment")
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
    os.environ["REGWATCH_RECENCY_HINT"] = "0" if args.no_recency_hint else "1"

    sys.path.insert(0, str(ROOT))
    from src.evaluator import (  # noqa: E402
        RUN_LOG, Probe, check_log_mode, log_turn, provenance, score_live,
    )
    from src.llm import get_gateway  # noqa: E402
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
    print("mode={llm_mode}  model={model}  cache={cache_dir}  "
          "recency_hint={recency_hint}".format(**prov))
    print("backends: {}".format(", ".join(args.backends)))
    print("probes per backend: {}  ->  {} calls".format(
        planned, planned * len(args.backends)))

    if args.dry_run:
        print("dry run: no calls made")
        return 0

    # Refuse to mix modes in one log before doing any work. An expected stop,
    # so report it plainly rather than as a traceback.
    try:
        check_log_mode(args.mode, RUN_LOG, model=prov.get("model"),
                       recency_hint=prov.get("recency_hint"))
    except RuntimeError as exc:
        print("\nrefusing to run: {}".format(exc), file=sys.stderr)
        return 2

    lock = RUN_LOG.parent / ".run.lock"
    holder = _claim_lock(lock)
    if holder:
        print("\nrefusing to run: another benchmark (pid {}) is writing to {}. "
              "Two runners appending to one log interleave their rows and the "
              "result cannot be untangled afterwards.".format(holder, RUN_LOG),
              file=sys.stderr)
        return 3

    started = time.time()
    totals = {}
    ingestion = {}
    gateway = get_gateway()
    try:
      for name in args.backends:
        spec = get_spec(name)
        if not spec.available():
            print("  {:16} SKIPPED — not available".format(name))
            continue
        backend = spec.get()
        backend.reset(SESSION)
        ok = errors = 0
        # Ingestion cost is whatever the gateway is asked for during
        # load/update. For the in-process backends this is a measured zero —
        # they assemble context in Python — but a hosted memory service would
        # spend tokens here, and that must not be mixed into answer cost.
        ingest_before = gateway.usage()["calls"]

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
                    # Read immediately: a cached turn returns in ~0 ms, so a
                    # row that does not say which it was makes the latency
                    # column meaningless on a warm cache.
                    log_turn(SESSION, name, probe, turn, score_live(turn, probe),
                             "ok", cached=gateway.last_cached)
                    ok += 1
                except Exception as exc:
                    log_turn(SESSION, name, probe, None, None, "error",
                             "{}: {}".format(type(exc).__name__, exc))
                    errors += 1
                done = ok + errors
                if done % 10 == 0 or done == planned:
                    print("  {:16} {}/{}".format(name, done, planned), flush=True)

        ingest_total = gateway.usage()["calls"] - ingest_before - ok - errors
        totals[name] = (ok, errors)
        ingestion[name] = max(0, ingest_total)
        print("  {:16} done — {} ok, {} errors, {} ingestion calls".format(
            name, ok, errors, ingest_total))
    finally:
        lock.unlink(missing_ok=True)

    usage = gateway.usage()
    summary = {
        "finished": time.time(), "seconds": round(time.time() - started, 1),
        "provenance": prov, "gateway": usage,
        "backends": {n: {"ok": ok, "errors": e, "ingestion_calls": ingestion.get(n)}
                     for n, (ok, e) in totals.items()},
    }
    (RUN_LOG.parent / "run_summary.json").write_text(json.dumps(summary, indent=2))

    print("\nwrote {} in {:.0f}s".format(RUN_LOG, time.time() - started))
    for name, (ok, errors) in totals.items():
        print("  {:16} ok={:3} errors={}".format(name, ok, errors))
    print("  gateway: {calls} calls, {cache_hits} cached, {cache_misses} fresh"
          .format(**usage))
    return 1 if any(e for _, e in totals.values()) else 0


def _claim_lock(path) -> Optional[int]:
    """Take the run lock, or return the pid already holding it.

    Two runners appending to one JSONL file interleave their rows, and nothing
    afterwards can say which process wrote which — the rows look identical.
    A stale lock from a killed run is reclaimed.
    """
    path.parent.mkdir(exist_ok=True)
    try:
        existing = int(path.read_text().strip())
    except Exception:
        existing = None
    if existing is not None and existing != os.getpid():
        try:
            os.kill(existing, 0)          # signal 0 only tests for existence
            return existing
        except ProcessLookupError:
            pass                          # holder is gone; the lock is stale
        except PermissionError:
            # The process exists, we just may not signal it. Alive.
            return existing
    path.write_text(str(os.getpid()))
    return None


if __name__ == "__main__":
    raise SystemExit(main())
