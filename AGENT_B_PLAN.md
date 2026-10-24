# Agent B — Build Plan

Companion to `PLAN.md`. B owns the **service, the two hosted-memory integrations, the scoring wiring, and the demo**. A owns the engine (data, gateway, in-process adapters, DAG metric).

Branch: `agent-b/service`. Never push to `main` directly. Never edit A's files.

---

## The one thing to do first

**Two 30-minute spikes, before writing anything else: Zep and Letta.**

They are the only parts of this project that can fail for reasons outside the team's control (signup, API keys, rate limits, SDK churn). Everything else is code you control.

The spike's success criterion is **not** "did it install" or "did it answer". It is:

> Can I get, out of this tool, the **exact text it injected into the model's prompt** for one question?

That string becomes `TurnResult.retrieval_context`, and axes 2 and 3 (precision, faithfulness/recall) are unscoreable without it. A backend that answers beautifully but won't show its context is only scoreable on axes 1/4/5 — which must be known in hour 1, not hour 14.

**Go / no-go at the end of hour 1:**

| Outcome | Move |
|---|---|
| Both expose context | Full five-column plan, as written |
| One works | Four columns, say so openly on the limitations slide |
| Neither works | Drop to Plan 2 — B builds Memory Guard + the two-column demo instead |

Record the decision and the reason. It becomes the "what we cut and why" slide, which judges respect.

---

## Build order

### Hour 0–1 — Spikes
Zep and Letta, in parallel if two pairs of hands are free. Go/no-go above.

### Hour 1–3 — Orchestrator skeleton (unblocked by A)
`src/orchestrator.py` with the plugin registry and **one fake backend** that returns a canned `TurnResult`. The service runs end to end before any real backend exists. Copy `base.py` from A the moment it lands; until then, type against the version in `PLAN.md`.

Registry rules:
- Backends register by `name`; adding one is a one-liner.
- **Lazy-import the hosted adapters.** A must be able to run the engine without `zep-python` or `letta` installed. A top-level import of either breaks A's machine.
- Each backend exposes `available() -> bool` so a missing key degrades to a greyed-out column, not a crash.

### Hour 3–6 — Zep adapter
`src/adapters/zep_graph.py`. The showcase backend: temporal graph, contradiction resolution. `load` ingests, `update` adds the contradicting statement, `ask` queries and returns the surfaced facts/edges as `retrieval_context`.

### Hour 4–8 — Evaluator (overlaps)
`src/evaluator.py`. Wire per axis:

| Axis | Metric | Cost |
|---|---|---|
| 1 Consistency | A's `dag_consistency` | free, deterministic |
| 2 Context precision | Ragas `context_precision`, DeepEval `ContextualPrecisionMetric` | LLM judge |
| 3 Faithfulness / recall | DeepEval `FaithfulnessMetric`, `ContextualRecall` | LLM judge |
| 4–5 Latency / cost | read off `TurnResult`, tokens→cost | free |

Persist every probe×backend to `results/run_log.jsonl` (one line each) and scores to SQLite.

### Hour 8–11 — Letta adapter
`src/adapters/letta_paging.py`. Capture the core-memory block as `retrieval_context`. Letta self-edits its memory, so note in the docstring if context capture is lossy, and run it **3× per probe** — it is the one backend with real run-to-run variance.

### Hour 11–16 — Demo UI
`ui/` + `app.py`. Five columns, one per backend, same probe answered differently, plus a scoreboard tab and the context-rot chart.

Built to **replay `results/run_log.jsonl`**, with an optional "re-run live" button that hits `/ask`. The stage demo must not depend on a live API call.

### Hour 16–19 — Scoreboard + charts
`figures/` generation script. Accuracy vs total cost, consistency by phase, the context-rot curve.

### Hour 20–24 — Freeze and rehearse
Numbers frozen at hour 21. Two rehearsals. Record a fallback video of the whole demo.

---

## Four decisions that will bite if made late

**1. One slow backend must not kill `/ask`.**
Fan out with `asyncio.gather(..., return_exceptions=True)` plus a per-backend timeout. Every column returns a status — `ok` / `timeout` / `error` — and the UI renders a failed column greyed out with the reason. On stage, four columns and one honest error beats a spinner.

**2. Do not run LLM-judge metrics in the live request path.**
DeepEval and Ragas call a judge model — seconds of latency and real cost per probe. `/ask` should return only the **free deterministic score** (`dag_consistency`) plus tokens and latency. The LLM-judged axes run offline over `run_log.jsonl` and land in the scoreboard. `PLAN.md` says "live scores" on `/ask`; this is the one place to deviate, and the reason is stage safety.

**3. Zep and Letta make their own internal LLM calls.**
Those do not pass through A's gateway, so A's token and cost capture is blind to them. Either read the tool's own usage reporting or report their cost as "answer-only, ingestion excluded" — and say which on the chart. Silently comparing a backend's full cost against another's partial cost is the kind of error a judge will find.

**4. "Identical model and prompt across adapters" is not literally achievable.**
Zep and Letta have internal prompts you don't control. Hold constant what you can — the answer model, the probe template, the budget — and document the rest as configuration. State it on the fairness slide before anyone asks.

---

## Sync points with A

1. A pushes `base.py` + stub `llm.py` → B starts for real.
2. A pushes `generate.py`'s output schema → B wires the evaluator to read `fact_id` and `expected_answer`.
3. Integration: B's orchestrator imports all five adapters and runs the full four-phase script end to end.

`src/adapters/base.py` changes only via a PR titled `CONTRACT:`, agreed by both.

---

## Definition of done

- `/ask` returns five `TurnResult`s with deterministic scores, surviving one backend being down
- `run_log.jsonl` + SQLite populated; offline scorer fills axes 2 and 3
- UI shows five columns + scoreboard, replaying from the log with no live API dependency
- **The money shot:** Zep visibly invalidates the stale fact while LiteLLM-trim does not — on a probe rehearsed and captured in the log
