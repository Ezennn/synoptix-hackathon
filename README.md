# RegWatch

**An automated safety exam for AI memory — run before an assistant reaches real customers.**

Built for the Synoptix ethical-AI hackathon. Primary track: **Governance & Compliance**,
supported by **Explainability & Transparency**.

---

## The problem

A client tells their adviser in January that they are crypto-heavy and aggressive. In
March they divest and turn conservative. In April someone asks the assistant whether a
high-risk crypto ETF is suitable.

Nobody typed anything wrong. The January note was true in January. But the memory layer
in front of the model decides which of those facts reaches the prompt, and it can fail
in two opposite directions:

- **It keeps the dead** — the superseded fact is still in the context window.
- **It loses the living** — trimming to save tokens drops the current fact entirely.

From outside, both look the same: a fluent, confident assistant. Model evaluation does
not catch either one, because it scores the *answer*, not the context the answer came
from. That context layer is what RegWatch tests.

This is not hypothetical. In *Moffatt v. Air Canada* (2024 BCCRT 149) the airline argued
its chatbot was a "separate legal entity" responsible for its own answers. The tribunal
disagreed and ordered the airline to pay. What your assistant says is what your company
said.

## What it does

1. **Replays a change.** A synthetic client and a MiFID-style suitability policy evolve
   across four dated sessions. Facts carry validity intervals, so a question asked in
   January is scored against January's truth, not the latest value.
2. **Scores what the memory layer handed over.** Correctness is checked by code against a
   temporal answer key — not one model grading another. An answer that declines to answer
   is recorded as an abstention, not a wrong answer.
3. **Says what to change**, with the token cost of the change.

## What we measured

100 questions, 3 memory systems, 300 live answers. Same model (Claude Haiku), same system
prompt for every backend — the only variable is the memory layer.

| Memory configuration | Correct | Couldn't answer | Stale fact in context | Tokens/query |
|---|---|---|---|---|
| Keep everything | **92 / 100** | 2 | 71% | 1,086 |
| Latest-per-key reducer | 75 / 100 | 13 | 54% | 480 |
| Trim to token budget | **22 / 100** | **61** | 17% | 250 |

Two things to read here. The spread — 22% to 92% — comes from the memory layer alone.
And trimming has the *least* stale context of the three while being by far the worst,
because it holds almost nothing: it declined 61 times out of 100. A metric that flatters
the emptiest system is why these are reported together.

### The result we did not want

Our hypothesis was that stale facts poison answers. We tested it: the system prompt
contained one line, *"If facts conflict, prefer the most recent."* We removed it and
re-ran all 300.

- Removing it costs **4–7 points** of accuracy across all three systems.
- But of the **19** questions that broke, only **7** had a stale fact in context.

If contamination were the mechanism, the damage would have concentrated there. It did
not. On this model, what breaks the adviser is **forgetting**, not remembering too well.
A benchmark that only confirms what you hoped is not measuring anything.

`python scripts/analyse_ab.py` reproduces all of this from the committed logs.

## Running it

Python 3.9+.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

**Generate the data and answer key** (deterministic):

```bash
.venv/bin/python src/generate.py        # -> data/sessions/*.json, data/probes.json
```

**Run the benchmark.** Stub mode is offline, free and needs no API key — it exercises the
whole pipeline but the stub does not reason, so its accuracy means nothing:

```bash
.venv/bin/python scripts/run_benchmark.py --limit 2            # smoke test, offline
```

For real answers, put a key in `.env` (`ANTHROPIC_API_KEY=...`, gitignored) and:

```bash
.venv/bin/python scripts/run_benchmark.py \
    --mode live --model anthropic/claude-haiku-5-5 \
    --cache-dir results/cache-live
```

300 calls, about three minutes, roughly $0.40. Responses are cached on disk, so a re-run
is free. Use a fresh `--cache-dir` when you care about latency: a cached turn returns in
~0 ms, and each answer records whether it was served from cache.

Only one benchmark may write the run log at a time — a second one refuses rather than
interleaving rows into a log nothing can untangle afterwards.

**Costs** come from `config/pricing.yaml`, where each model's rate carries its source and
the date it was taken. A model with no entry reports cost as unknown, never zero.
Override for one run with `REGWATCH_PRICE_IN_PER_M` / `REGWATCH_PRICE_OUT_PER_M`.

Add `--no-recency-hint` for the B arm of the prompt experiment.

**See the results:**

```bash
.venv/bin/streamlit run app.py          # dashboard, reads results/run_log*.jsonl
.venv/bin/python scripts/analyse_ab.py  # the A/B comparison
.venv/bin/python -m pytest tests/       # scorer regression tests
```

The dashboard opens on the committed run, so a fresh clone shows real results without
running anything. The sidebar names the mode, model and prompt arm of whichever log you
select — demoing the wrong one should take deliberate effort.

**The service** (used by the live demo, not needed for the benchmark):

```bash
.venv/bin/uvicorn src.orchestrator:app --port 8077
```

## Layout

| Path | What it is |
|---|---|
| `src/adapters/base.py` | The frozen contract every memory backend implements |
| `src/adapters/` | sanity (full history), litellm_trim, langgraph_state, Zep/Letta scaffolds |
| `src/generate.py`, `data/facts.yaml` | The temporal answer key and the data built from it |
| `src/llm.py` | Cached gateway; tokens and latency measured here, never estimated |
| `src/metrics/dag_consistency.py` | Deterministic state-consistency scoring |
| `src/registry.py`, `src/orchestrator.py` | Backend plugins and the FastAPI service |
| `src/evaluator.py` | Scoring, run-log provenance, the scoreboard store |
| `scripts/run_benchmark.py` | The benchmark runner |
| `app.py`, `ui/replay.py` | Dashboard and its aggregation |
| `results/run_log.recency-*.jsonl` | The two committed A/B runs |

Built by two agents against one frozen interface; see [PLAN.md](PLAN.md) and
[AGENT_B_PLAN.md](AGENT_B_PLAN.md).

## Limitations

- **Synthetic data only.** One fictional client (Acme Capital Partners LLP) and an
  invented versioned policy. No real client information. MiFID is real regulation; the
  policy versions here are not.
- **One model, one timeline.** Claude Haiku, four sessions, 100 questions. Per-bucket
  samples run from 7 to 34. We are not claiming a general law.
- **The transitive tier does not yet require inference** — session 3 states the derived
  conclusion verbatim, so the model is handed the answer rather than deducing it.
- **Abstention is scored as incorrect.** That is the right semantics for "did the memory
  keep the fact", but a system that admits ignorance is safer in production than one that
  states a stale value confidently, so the two are reported separately.
- Zep and Letta adapters are scaffolds; their integration spikes were never run.
