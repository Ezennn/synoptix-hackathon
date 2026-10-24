# Erased & Replaced

**Does AI prompt compression delete who you are, and does the AI then make someone up to fill the gap?**

Built for the Synoptix ethical-AI hackathon, Track 1: Bias Mitigation.

> Compression erases who you are; the AI replaces you with someone else. We measure both steps and fix them.

---

## What we're actually doing (plain English)

AI is charged by the word, technically by the "token". To save money, many companies **shorten** text before an AI reads it. Tools such as LLMLingua automatically delete the words they think are unimportant, or a model summarises the text into a few lines.

Nobody checks *which* words get deleted. We think the deleted words are often the ones that describe **who a person is** or **what they need**:

- their name
- their pronouns ("my pronouns are they/them")
- an adjustment request ("I'll need interview materials in large print")
- a caring responsibility ("I care for my mum, so I need morning slots")

When that information disappears, the AI doesn't say "I don't know". We expect it to **fill the gap with a default**: it calls the person "he", assumes a typical name, or assumes they need nothing.

So the harm happens in two steps:

1. **Erased:** compression deletes the detail.
2. **Replaced:** the AI invents a stereotypical stand-in.

The person told the system. The system forgot. Then it made them up.

**What we do about it:**

1. Write fake CVs and support requests. Each one comes in versions that are identical apart from one detail (for example, he vs she vs they).
2. Shorten every version using several real compression tools, at several strengths.
3. Check what got deleted.
4. Ask an AI questions about the shortened text ("What pronouns should we use?", "Does this person need an adjustment?") and see whether it answers correctly, admits it doesn't know, or makes something up.
5. Have the AI do a real task, such as writing an interview invitation, and see whether the outcome changes.
6. Test cheap fixes and show they stop the problem while still saving most of the money.

**Why it matters:** companies test their AI models for bias, but the compression step in front of the model is usually never tested. If someone asks for a reasonable adjustment and the pipeline quietly deletes it, the person loses out, and the organisation may still be responsible.

---

## How the harm happens

```
 Person writes ──► Compressor ──► Short prompt ──► LLM reads ──► Decision
 name, pronouns,    deletes words   details gone,    fills each gap  invitation,
 stated needs       to save tokens  nothing marked   with a default  triage, reply
                         │                                │              │
                   Erasure Rate              Refill Rate,           Silent Flip
                                             Default Drift
```

---

## Research questions

| # | Question | Hypothesis |
|---|---|---|
| RQ1 | Does compression delete identity and needs information more than neutral information of the same length? | Yes, and more at higher compression |
| RQ2 | Is the deletion unequal across groups? | Non-Western names, they/them pronouns and adjustment requests are erased more |
| RQ3 | When information is gone, does the model refill it or say "unknown"? | It refills, leaning towards defaults (male, Western name, no needs) |
| RQ4 | Does *silent* deletion cause more refilling than *marked* deletion ("[not stated]")? | Yes |
| RQ5 | Can cheap fixes cut the harm while keeping most of the savings? | Protected fields plus erasure markers keep at least 80% of the savings |

Any result is useful. If compression makes some decisions *fairer* by hiding identity, we report that too, and show where it breaks down.

---

## Metrics

| Metric | What it measures |
|---|---|
| **Erasure Rate** | Share of identity or needs facts lost after compression, per group and compression level |
| **Refill Rate** | When a fact was lost, how often the AI gives a specific answer anyway instead of "unknown" |
| **Default Drift** | What the AI fills in, e.g. what share of lost pronouns became "he" |
| **Silent Flip** | How often the real decision changes (e.g. adjustment no longer offered) because of compression |

The headline fairness number is the **gap between groups** (e.g. Erasure Rate for they/them minus he/she), with a 95% confidence interval.

---

## Experiment design

**Scenarios**
- **A: Hiring.** CVs and cover letters compressed before an AI screens them and drafts an interview invitation.
- **B: Support requests.** People describing their situation to an assistant, compressed before it triages and replies.

**Counterfactual variants:** each base document has one slot that changes. Everything else stays identical.

| Slot | Variants | Length-matched control |
|---|---|---|
| Name | Western vs non-Western (several names each, from published lists) | None needed |
| Pronouns | he / she / they | None needed |
| Adjustment request | e.g. large print, screen reader | A hobby sentence |
| Caring responsibility | e.g. needs morning slots | A commute sentence |

**Compressors**
- LLMLingua-2 (main tool; gives a keep-probability for every word)
- LLMLingua (original, perplexity-based)
- LLM summarisation
- Truncation (keep the first N tokens)
- Random deletion (baseline)

**Compression ratios:** 2×, 3×, 5×, 10×

**Three versions of every document**
1. **Original:** the ground truth
2. **Compressed:** information may be silently missing
3. **Marked neutral:** the detail replaced with "[not stated]"

**Probes:** fact questions (name, pronouns, needs), one open "describe this person" question, and the real decision task. "Not stated" is always an allowed answer.

**Target models:** one small open model run locally, plus one API model.

---

## Mitigations tested

| Fix | Targets | How it works |
|---|---|---|
| Protected fields | Erasure | Force-keep names, pronouns and stated needs |
| Erasure markers | Replacement | Insert "[personal details removed]" wherever such text was cut |
| Abstain instruction | Replacement | Tell the model never to assume unstated personal details |
| Extract then compress | Both | Pull key facts into a short header, then compress the rest |

Each fix is scored on the fairness metrics **and** on how much compression (cost saving) it keeps.

---

## Analysis

- 95% bootstrap confidence intervals, resampling base documents
- Paired tests within each counterfactual pair (McNemar's test for yes/no outcomes)
- Logistic regression on what predicts a fact being deleted (slot type, group, compressor, ratio, position)
- LLMLingua-2 keep-probabilities and tokens per word, to explain *why* words are dropped
- Benjamini–Hochberg correction for multiple comparisons
- Hand-check of 50 probe answers against the automatic scorer
- Hypotheses and primary comparisons pre-registered before the full run

---

## Planned repo layout

```
erased-and-replaced/
├── data/
│   ├── base_documents/      # synthetic CVs and support requests
│   ├── slots/               # names, pronouns, adjustment and caring phrases, controls
│   └── variants/            # generated counterfactual documents
├── src/
│   ├── generate_variants.py # builds counterfactual docs from bases + slots
│   ├── compress.py          # runs every compressor at every ratio
│   ├── probe.py             # asks the target models fact, open and decision probes
│   ├── score.py             # turns answers into Erasure / Refill / Drift / Flip
│   ├── analyse.py           # CIs, tests, regression, charts
│   └── fixes.py             # protected fields, markers, abstain prompt, extract-then-compress
├── results/                 # cached model outputs and metric tables
├── figures/                 # charts for the pitch
├── app.py                   # Streamlit demo
├── PREREGISTRATION.md       # hypotheses and primary comparisons, fixed before the full run
└── README.md
```

## Getting started (planned)

```bash
git clone <repo-url>
cd erased-and-replaced
pip install -r requirements.txt   # llmlingua, pandas, scipy, statsmodels, streamlit, ...

python src/generate_variants.py
python src/compress.py --ratios 2 3 5
python src/probe.py --model <local-or-api-model>
python src/score.py
python src/analyse.py             # writes tables to results/ and charts to figures/

streamlit run app.py              # live demo
```

Random seeds, model versions and prompts are recorded in `results/` so every chart can be regenerated with one command.

---

## The demo

1. Paste a cover letter that includes "they/them" and a large-print request.
2. Slide the compression level up and watch deleted words turn red and disappear.
3. See the AI's invitation: wrong pronoun, no adjustment offered.
4. Turn on the fix. The details survive, the invitation is correct, and the prompt is still short.
5. Show the chart: it's not one example, it's the whole test set.

---

## Ethics

- **Synthetic data only.** No real people's CVs, disabilities or pronouns.
- **Names used respectfully.** They come from published research lists, and results are reported by group, never by individual name.
- **We fix the system, not the person.** We never suggest people drop their pronouns or needs to survive compression.
- **Protected fields are a value judgement.** We say so, and recommend affected groups help define them.
- **Legal points are questions, not advice.** We raise the UK Equality Act "knowledge" issue around reasonable adjustments as a governance question, not a legal conclusion.

## Limitations

- Synthetic documents are cleaner than real ones.
- Results cover the compressors and models we tested.
- "Western vs non-Western" names is a crude grouping, so each name set is also reported separately.
- English only. Code-switched and non-English text is future work.
- An LLM judge scores open answers. Hand checks reduce that error but don't remove it.
- Hackathon-sized samples give wide confidence intervals; we report them rather than hide them.

---

## Related work

- Łajewska et al., [Understanding and Improving Information Preservation in Prompt Compression for LLMs](https://aclanthology.org/2025.findings-emnlp.949.pdf), EMNLP Findings 2025. Compressors often lose names, dates and numbers; tested in English only, with no fairness analysis.
- [An Empirical Study on Prompt Compression for Large Language Models](https://arxiv.org/html/2505.00019v1), 2025. Six compressors compared on 13 datasets; no demographic analysis.
- [LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression](https://arxiv.org/html/2403.12968v2), 2024.
- [Prompt Fairness: Sub-group Disparities in LLMs](https://arxiv.org/html/2511.19956v1), 2025. The same question phrased in different demographic styles gets different answers; does not cover compression.
- Haq & Saldías, [Dialect vs Demographics: Quantifying LLM Bias](https://arxiv.org/html/2604.21152), 2026.

**Our contribution:** to our knowledge, this is the first study of whether compression erases identity and needs *unequally*, and whether the model then *refills* the gap with stereotyped defaults.
