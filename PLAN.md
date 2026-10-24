# RegWatch AI — Split-Work Plan (two agents, two machines, one repo)

This file is the contract between **Agent A** and **Agent B**. Read it fully before writing code. The whole point of the split is that the two halves meet at **one frozen interface** (`src/adapters/base.py`) and never edit the same files, so two machines can work in parallel without merge conflicts.

## The golden rule: one interface, frozen first

Everything hinges on a single Python Protocol that every memory backend implements and that the evaluator scores against. **Agent A writes it first and commits it before anyone builds on it.** Once committed, neither agent changes its signature without telling the other — a change here breaks both halves.

```python
# src/adapters/base.py  — THE CONTRACT. Freeze this before parallel work starts.
from dataclasses import dataclass
from typing import Protocol

@dataclass
class TurnResult:
    answer: str              # the LLM's answer to the probe
    retrieval_context: list[str]  # EXACT text chunks the memory layer injected (for precision/faithfulness/recall)
    tokens_in: int           # prompt tokens actually sent
    tokens_out: int          # completion tokens
    latency_ms: float        # wall-clock for the whole load+retrieve+generate for this turn

class MemoryBackend(Protocol):
    name: str
    def reset(self, session_id: str) -> None: ...
    def load(self, session_id: str, documents: list[str]) -> None: ...      # Phase 1 ingest
    def update(self, session_id: str, statement: str) -> None: ...          # Phase 2 mutation
    def ask(self, session_id: str, question: str) -> TurnResult: ...        # probe -> scored result
```

Non-negotiable notes for whoever writes an adapter:
- `retrieval_context` must be the **real** text the backend fed the model this turn. If a tool won't expose it, say so in the adapter docstring — that backend can only be scored on axes 1/4/5, and we need to know up front.
- Latency and tokens come from the **gateway** (`src/llm.py`), not estimated. Every adapter calls the LLM only through that gateway.
- The LLM model + prompt template are **identical** across all adapters. Only the memory layer differs.

## Who owns what (no shared files)

| Area | Owner | Files (owned exclusively) |
|---|---|---|
| Interface contract | **A (first)** | `src/adapters/base.py` |
| Data generation + answer key | **A** | `src/generate.py`, `data/templates/`, `data/facts.yaml`, `data/sessions/` |
| LLM gateway | **A** | `src/llm.py` |
| Simple adapters | **A** | `src/adapters/sanity.py`, `src/adapters/litellm_trim.py`, `src/adapters/langgraph_state.py` |
| Custom DAG metric | **A** | `src/metrics/dag_consistency.py` |
| Hosted-memory adapters | **B** | `src/adapters/zep_graph.py`, `src/adapters/letta_paging.py` |
| FastAPI orchestrator | **B** | `src/orchestrator.py` |
| Evaluator (DeepEval + Ragas wiring) | **B** | `src/evaluator.py` |
| Demo UI (five live columns) | **B** | `ui/`, `app.py` |
| Scoreboard + context-rot charts | **B** | `figures/` generation script |
| Shared, edit-with-a-heads-up | both | `requirements.txt`, `README.md` |

Rule of thumb: **A owns the engine and the two things nothing off-the-shelf gives us (temporal answer key + DAG metric). B owns the service, the two hard external integrations (Zep/Letta), the scoring wiring, and the demo.** The interface sits between them.

---

## Agent A — the engine (the machine running the KiroCrew session)

**Mission:** produce deterministic data with a temporal answer key, a cached LLM gateway, the three in-process backends, and the one metric that can't be bought. Everything B builds depends on A's interface and gateway, so **A commits the contract + a stub gateway on day one.**

1. **`src/adapters/base.py`** — write and commit the `MemoryBackend` protocol + `TurnResult` above. **Do this first.** Tell B the moment it's pushed.
2. **`data/facts.yaml` schema + `src/generate.py`** — the heart of the project. Every fact carries temporal validity:
   ```yaml
   - id: crypto_appetite
     entity: client_acme
     attribute: risk_appetite_crypto
     value: aggressive
     stated_in_session: 1
     valid_from: 2026-01-10
     valid_until: 2026-03-15        # null while current
     superseded_by: crypto_appetite_v2
   ```
   `generate.py` emits: `data/sessions/*.json` (multi-session transcripts for all 4 phases, with the graded Phase-2 tiers and plausible-but-stale Phase-4 distractors), and 100 probe questions each tagged with the fact id(s) and the **expected current answer** at ask-time. Deterministic (fixed seed).
3. **`src/llm.py`** — one cached gateway (LiteLLM), returns `(text, tokens_in, tokens_out)` and records latency. Retries + on-disk response cache so reruns are free. **All adapters call only this.**
4. **Three in-process adapters** implementing the contract:
   - `sanity.py` — stuff all history into context (the control/ceiling).
   - `litellm_trim.py` — hard message trimming to a token budget.
   - `langgraph_state.py` — LangGraph checkpointer + a custom reducer that summarises/prunes.
5. **`src/metrics/dag_consistency.py`** — the custom State-Consistency metric. A deterministic decision tree over the answer key: given the probe's fact id and ask-time, does the answer assert the *current* value and NOT the superseded one? Binary per probe, averaged to a Consistency Rate. (Use DeepEval's `DAGMetric` if it fits; otherwise a plain scorer with the same contract — returns a 0/1 and a reason string per test case.)

**A's definition of done:** `python src/generate.py` produces data + answer key; the three adapters each answer a probe through the gateway and return a valid `TurnResult`; `dag_consistency` scores a known-good and a known-stale answer correctly.

---

## Agent B — the service, external memory, scoring, and demo (the other machine)

**Mission:** wrap the two hosted memory tools in A's interface, stand up the FastAPI orchestrator that runs all five in parallel, wire DeepEval/Ragas scoring, and build the five-column demo. **B can start against A's committed `base.py` with A's three adapters as reference implementations — do not wait for the whole engine.**

1. **`src/adapters/zep_graph.py`** — implement `MemoryBackend` over Zep/Graphiti. `load` ingests documents, `update` adds the contradicting statement, `ask` queries the temporal graph and returns `retrieval_context` = the facts/edges Zep surfaced. This is the showcase backend for contradiction resolution — get `retrieval_context` out of it.
2. **`src/adapters/letta_paging.py`** — implement the same over Letta/MemGPT. Capture its core-memory block as `retrieval_context`. Note in the docstring if self-editing makes context capture lossy.
3. **`src/orchestrator.py`** — FastAPI app, endpoints:
   - `POST /load {session_id, documents}` → fan out to all registered backends.
   - `POST /update {session_id, statement}` → fan out.
   - `POST /ask {session_id, question}` → call all backends **in parallel**, return each one's `TurnResult` + live scores.
   - `GET /scores` → the scoreboard across all five axes.
   Backends are registered by `name` from a plugin list, so adding one is a one-liner.
4. **`src/evaluator.py`** — wire the metrics to each `TurnResult`:
   - Axis 1: call A's `dag_consistency`.
   - Axis 2: Ragas `context_precision` + DeepEval `ContextualPrecisionMetric` / `ContextualRelevancy` on `retrieval_context`.
   - Axis 3: DeepEval `FaithfulnessMetric` + `ContextualRecall`.
   - Axes 4/5: read latency/tokens straight off `TurnResult`, convert tokens→cost.
   Persist every run to `results/run_log.jsonl` (one line per probe per backend) and scores to SQLite.
5. **`ui/` + `app.py`** — five live columns (one per backend) showing the same probe answered differently, plus a scoreboard tab and the context-rot chart. Build it to **replay `results/run_log.jsonl`** so the stage demo has no live-API risk, with an optional "re-run live" button that hits `/ask`.

**B's definition of done:** `/ask` returns five scored `TurnResult`s; the evaluator writes `run_log.jsonl` + SQLite scores; the UI shows five columns + scoreboard; Zep visibly invalidates the stale fact while LiteLLM-trim doesn't.

---

## Shared workflow (how two machines avoid stepping on each other)

- **Branch per agent:** A works on `agent-a/engine`, B on `agent-b/service`. Never push to `main` directly. Open a PR into `main`; the other reviews.
- **The contract is sacred:** `src/adapters/base.py` changes only by a PR explicitly titled `CONTRACT:` and only after both agree. A signature change there is the one thing that breaks both halves.
- **Integration points to agree on now:**
  - `TurnResult` fields (frozen above).
  - The probe-question JSON schema (`generate.py` output) — B's evaluator reads `fact_id` and `expected_answer` from it, so A publishes that schema in `generate.py`'s docstring.
  - `dag_consistency`'s call signature — B imports it; A keeps it stable.
- **requirements.txt:** append-only where possible; if you must pin, flag it in the PR. Heavy deps (zep-python, letta) are B's; keep A's engine importable without them (lazy-import the hosted adapters in the orchestrator's plugin registry, so A can run the engine without installing Zep/Letta).
- **Sync points:** (1) after A pushes `base.py` + `llm.py` stub — B starts. (2) after A pushes `generate.py` output schema — B wires the evaluator. (3) integration: B's orchestrator imports all five adapters and runs the full 4-phase script end to end.

## First three commits (unblock everyone fast)

1. **A:** `src/adapters/base.py` + a stub `src/llm.py` that returns canned text → B is unblocked.
2. **A:** `data/facts.yaml` + `generate.py` producing sessions + 100 tagged probes → the test data exists.
3. **B:** `orchestrator.py` with the plugin registry + sanity adapter wired → the service runs end to end on one backend, then the rest slot in.
