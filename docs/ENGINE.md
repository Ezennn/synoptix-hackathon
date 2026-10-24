# Engine (Agent A) — quickstart

Agent A's half of RegWatch: the frozen contract, deterministic data + temporal
answer key, cached LLM gateway, three in-process backends, and the custom DAG
state-consistency metric. It runs **fully offline** in stub mode — no API key,
no heavy deps — so Agent B and reviewers can exercise the whole pipeline.

## Install

```bash
pip install -r requirements.txt    # only PyYAML is required for the engine
```

## Generate the data + answer key

```bash
python src/generate.py
# -> data/sessions/session_{1..4}.json   (transcripts + phase-4 distractors)
# -> data/probes.json                     (100 probes, each tagged fact_ids + expected_answer)
```
Deterministic (seed=42): a rerun reproduces byte-identical output.

## Run a backend through the gateway

```python
from src.adapters.sanity import SanityBackend
b = SanityBackend()                 # stub mode by default (offline)
b.reset("s1")
b.load("s1", ["The client's risk appetite for crypto assets is now aggressive."])
b.update("s1", "The client's risk appetite for crypto assets is now conservative.")
r = b.ask("s1", "What is the current crypto risk appetite?")
# r.answer, r.retrieval_context, r.tokens_in, r.tokens_out, r.latency_ms
```

Switch to a real model:

```bash
REGWATCH_LLM_MODE=live  REGWATCH_MODEL=gpt-4o-mini  python ...
```

## Score an answer (axis 1 — state consistency)

```python
from src.metrics.dag_consistency import score_probe, consistency_rate
res  = score_probe(probe, r.answer)                 # DagScore(score, passed, reason)
agg  = consistency_rate([(probe, r.answer), ...])   # rate + per-tier breakdown + failures
```

## The pieces (all A-owned)

| File | Role |
|---|---|
| `src/adapters/base.py` | **frozen contract** — `MemoryBackend` + `TurnResult` |
| `src/llm.py` | cached gateway (stub/live); tokens + latency measured here |
| `data/facts.yaml` | temporal answer key (valid_from/until/superseded_by, 5 tiers) |
| `src/generate.py` | sessions + 100 tagged probes (probe schema in its docstring) |
| `src/adapters/sanity.py` | full-history control / ceiling |
| `src/adapters/litellm_trim.py` | hard token-budget truncation |
| `src/adapters/langgraph_state.py` | custom latest-per-key reducer |
| `src/metrics/dag_consistency.py` | deterministic state-consistency metric |

Agent B builds the Zep/Letta adapters, the FastAPI orchestrator, the
DeepEval/Ragas evaluator wiring, and the five-column UI against this contract —
see `PLAN.md`.
