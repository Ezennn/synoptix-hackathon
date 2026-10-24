# RegWatch AI — Memory-Tool Benchmark Harness

**Which agent-memory framework actually keeps track of the truth when facts change over time — and which one quietly makes things up?**

Built for the Synoptix hackathon. We benchmark five memory/context backends head-to-head on a stress test designed to break them, and score them with metrics that test **the memory layer itself**, not the LLM's reasoning.

> A standard RAG system retrieves the January fact *and* the March correction with equal confidence, then hallucinates conflicting advice. We measure which memory tools resolve that contradiction, which forget, and what it costs.

## The problem (plain English)

An AI advising a client over months has to track facts that **change**. A client says in January "we have a high-risk appetite for crypto," then in March "we are now strictly conservative and have divested from crypto." Ask a naive system what the client's strategy is and it retrieves both statements with equal similarity — and gives contradictory advice.

Good agent memory is supposed to fix this: invalidate the stale fact, keep the current one, and still remember the history. Different tools claim to do this very differently — temporal knowledge graphs, OS-style memory paging, hand-written state reducers, or just trimming old messages. **Nobody benchmarks them against each other on temporal correctness.** That's what this project does.

We drive the same multi-session scenario through five backends, ask the same probe questions, and score each on whether it keeps the current truth, resists clutter, and avoids hallucination — plus latency and cost.

## What makes this a real test (not an API-wrapper demo)

The whole point is to test the **memory tool**, not the model behind it. So:

1. The LLM is held constant across all backends (same model, same prompt template, routed through one cached gateway).
2. Every backend is wrapped by a **uniform adapter** returning the same shape: `(answer, retrieval_context, tokens, latency)`. The only thing that varies between runs is *which memory layer assembled the context*.
3. Metrics score the **retrieved context** and the **answer against a temporal answer key**, so a right answer for the wrong reason (luck, recency bias) is caught.

## The five backends under test

| Backend | Pattern | Hypothesis |
|---|---|---|
| **Sanity** (full history, no memory) | Everything in context, no management | Ceiling on accuracy, floor on cost — the control |
| **LiteLLM trim** | Hard message truncation | Fast and cheap, catastrophic forgetting on multi-hop |
| **LangGraph** | Checkpointer + custom state reducers | Full control, low cost, but you hand-write the pruning |
| **Zep / Graphiti** | Temporal knowledge graph | Best contradiction resolution, higher write latency |
| **Letta / MemGPT** | OS-style self-editing memory paging | Strong autonomy, risk of self-edit corruption |

All five run through the identical adapter interface so they're directly comparable.

## The stress test — 4 phases, graded difficulty

A deterministic multi-session script. Each phase has escalating sub-tiers so results show a difficulty curve, not a single pass/fail.

**Phase 1 — Baseline ingestion.** Feed complex corporate financial policies, historical client statements, and meeting notes across several simulated sessions.

**Phase 2 — Entity mutation & contradiction.**
- *Direct overwrite* (easy): "risk appetite is now conservative" — tests recency.
- *Scoped overwrite* (medium): "conservative for the pension sub-account, keep aggressive on the prop book" — does the tool scope the invalidation, or nuke the fact globally?
- *Transitive invalidation* (hard): divest Subsidiary B, where an earlier fact said Subsidiary B holds the crypto position — the crypto exposure must be invalidated *by inference*.
- *Reinstatement* (adversarial): A→B in session 2, then B→A in session 4 — does it track full history or just overwrite?

**Phase 3 — Multi-hop reasoning.** A question needing synthesis across Session 1 (January), Session 2 (April), **and** a policy document that was itself superseded — forcing temporal validity over two independent timelines (client state + regulation version) at once.

**Phase 4 — Adversarial noise.** Flood active history with *plausible-but-stale* distractors (a long meeting note discussing the OLD risk tier), not random JSON — the hard case that actually stresses precision and faithfulness.

**Context-length sweep:** rerun the probes at 5 / 20 / 50 / 100 turns to produce the "context rot" curve.

## Metrics — testing the memory, not the LLM

| Axis | Method(s) | Why | Tooling |
|---|---|---|---|
| **1. State Consistency** (overwrites stale facts) | Custom **DAG metric** (primary) + G-Eval (backstop) | Not a standard metric: faithfulness can't tell "old true fact" from "current true fact" — both are grounded. Needs a deterministic yes/no against the answer key. | DeepEval `DAGMetric` |
| **2. Context Precision** (low filler / noise resistance) | **Ragas `context_precision`** + DeepEval `ContextualPrecisionMetric` (ranking-aware) + `ContextualRelevancy` | Scores Phase 4: how much stale/irrelevant text the memory layer injected. | Ragas, DeepEval |
| **3. Faithfulness / Grounding** (no hallucination) | DeepEval `FaithfulnessMetric` (atomic-claim) + **`ContextualRecall`** | Faithfulness catches invented claims; recall catches *silent forgetting* — a tool can be faithful but drop a fact it should have surfaced. | DeepEval |
| **4. Retrieval Latency (ms)** | Wall-clock timing in the adapter | Direct measurement | Harness |
| **5. Cost Efficiency ($/query)** | Token accounting at the gateway | Direct measurement | Harness |

The headline per-backend result is a row on the scoreboard across all five axes; the headline *story* is the context-rot curve (axes 1–3 vs. session length).

## Architecture — Option 2: one FastAPI service, hot-swappable backend plugins

A single service orchestrates all five backends as plugins registered by name and called in parallel; a demo UI shows live side-by-side columns handling the same conflicting update.

```
        [Demo UI: five live columns]
                   │
                   ▼
        ┌───────────────────────┐        ┌───────────────────────┐
        │ FastAPI orchestrator  │◄──────►│ Fact extractor        │
        │ /load /update /ask    │        │ proposes, user confirms│
        │ /scores               │        └───────────────────────┘
        └──────────┬────────────┘
                   │  registered by name, called in parallel
   ┌────────┬──────┴─────┬────────┬────────┐
   ▼        ▼            ▼        ▼        ▼
 Sanity  LiteLLM-trim  LangGraph  Zep    Letta      (each optionally wrapped by Memory Guard)
   │                                               │
   ▼                                               ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────────────────┐
│ LLM gateway   │   │ Evaluator     │   │ SQLite store              │
│ LiteLLM,cached│   │ scores live   │   │ runs, scores, live facts  │
└───────────────┘   └───────────────┘   └───────────────────────────┘
```

**Demo-safety variant:** the engine runs the full benchmark offline and writes results; the UI *replays* those results in live-looking columns, with an optional "re-run live" button. Keeps the wow factor without live-API flakiness on stage.

## Repo layout (planned)

```
synoptix-hackathon/
├── data/
│   ├── templates/            # YAML scenario templates + seed
│   ├── facts.yaml            # temporal answer key: valid_from / valid_until / superseded_by
│   └── sessions/             # generated multi-session transcripts
├── src/
│   ├── generate.py           # builds facts, session phrasing, probe checks from templates
│   ├── adapters/             # ONE interface, five implementations
│   │   ├── base.py           # MemoryBackend protocol: load/update/ask -> (answer, context, tokens, latency)
│   │   ├── sanity.py
│   │   ├── litellm_trim.py
│   │   ├── langgraph_state.py
│   │   ├── zep_graph.py
│   │   └── letta_paging.py
│   ├── llm.py                # cached LLM gateway: retries, cost accounting
│   ├── orchestrator.py       # FastAPI app: /load /update /ask /scores
│   ├── evaluator.py          # DeepEval + Ragas + custom DAG metric
│   └── metrics/
│       └── dag_consistency.py# custom temporal state-consistency DAG metric
├── ui/                       # demo: five live columns
├── results/                  # run_log.jsonl, scores, cached outputs
├── figures/                  # scoreboard + context-rot charts
├── app.py                    # launches orchestrator + UI
└── README.md
```

## Getting started (planned)

```bash
git clone https://github.com/Ezennn/synoptix-hackathon
cd synoptix-hackathon
pip install -r requirements.txt   # fastapi, deepeval, ragas, zep-python, letta, langgraph, litellm, ...

python src/generate.py                      # build facts.yaml + sessions
uvicorn src.orchestrator:app --reload       # start the service
# POST /load (ingest Phase 1) -> /update (Phase 2 mutations) -> /ask (probes) -> /scores
```

Seeds, model versions and prompts are recorded in `results/` so every chart regenerates with one command.

## The demo

1. Load the January client profile into all five backends at once.
2. Push the March contradiction ("divested from crypto, now conservative") through `/update`.
3. Ask "what is the client's current crypto strategy?" — watch five columns answer side by side.
4. Zep invalidates the old fact; LiteLLM-trim confidently gives the stale January answer.
5. Flip to the scoreboard: it's not one example, it's the whole 100-question test set, with the context-rot curve.

## Why this wins

1. **Rigorous:** it benchmarks and scores trade-offs (accuracy vs latency vs cost), not just wraps an API.
2. **High business impact:** temporal correctness is a real multi-billion-dollar compliance/advisory bottleneck.
3. **Visually impressive:** live side-by-side columns show each framework handling the same contradiction differently.

## Division of work

Two agents, two machines, one repo. See [`PLAN.md`](PLAN.md) for the full split, the shared interface contract, and the branch workflow.
