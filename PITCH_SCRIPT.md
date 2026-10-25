# TraceWise AI — pitch script

Extracted from the speaker notes in `TraceWise_AI_Broader_Pitch.pptx`, so this file
and presenter view are the same words. Edit the deck, not this file.

**Timing:** ~6:40 spoken plus a 90-second demo. Lines starting in CAPS are stage
directions, not words to say.

**The one sentence to land if you only land one:** the memory layer alone moved
accuracy from 24 to 88 out of 100, and we have 300 live answers to prove it.

**Contents**

- [Slide 01 — Two questions for the room](#slide-01)
- [Slide 02 — Confident AI answers can have real consequences](#slide-02)
- [Slide 03 — One memory failure, many consequences](#slide-03)
- [Slide 04 — TraceWise: a memory check for AI systems](#slide-04)
- [Slide 05 — Live demonstration](#slide-05)
- [Slide 06 — Before: you trust it. After: you can test it](#slide-06)
- [Slide 07 — What we measured](#slide-07)
- [Slide 08 — Who uses it — and who benefits](#slide-08)
- [Slide 09 — One assurance engine, different ways to use it](#slide-09)
- [Slide 10 — Appendix — references and Q&A](#slide-10)

---

<a id="slide-01"></a>

## Slide 01 — Two questions for the room

00:00-00:25 - Opening. Two questions, one pause.

"Quick question. Have you ever corrected an AI assistant - and then watched it use the old fact again, later?"

(PAUSE. Let hands go up. That's your permission to keep going.)

"Second question. What if that assistant wasn't writing text, but booking the meeting, checking the safety spec, or approving the refund?

That's the gap we built TraceWise to test."

---

<a id="slide-02"></a>

## Slide 02 — Confident AI answers can have real consequences

00:25-01:05 - Evidence. Three cases, escalating.

"Three documented failures. They escalate.

AIR CANADA, 2024. Their chatbot invented a refund policy. The airline argued in front of a tribunal that the chatbot was a SEPARATE LEGAL ENTITY, responsible for its own answers. The tribunal called that remarkable, and rejected it. What your assistant says is what your company said.

ZILLOW, 2021. Three hundred and four million dollars of inventory written off in a single quarter, because the pricing model's picture of the market fell behind the market. They shut the business down and cut a quarter of the company. A stale view of the world, priced at scale.

NEW YORK CITY, 2024. The city's official business chatbot told owners they could take their workers' tips, and refuse tenants on rental assistance. Reporters published it. It stayed online.

Three different causes. One shared gap - nothing tested the answer before someone relied on it."

STAGE: The third one is the point. Liability and cost are setup; "no test existed" is why you're on stage. Land it and move.

---

<a id="slide-03"></a>

## Slide 03 — One memory failure, many consequences

01:05-01:35 - Two directions.

"This isn't a finance problem or an airline problem. A travel assistant plans around dates you changed. A scheduling agent picks availability a candidate withdrew. A support bot quotes the old refund window.

And it breaks in two directions, not one. Memory can KEEP THE DEAD - the superseded fact is still sitting in the context. Or it can LOSE THE LIVING - you trimmed to save tokens and the current fact went with it.

From the outside both look identical: a fluent, confident assistant."

STAGE: "Keep the dead / lose the living" is the most memorable phrasing in the deck. Say it slowly.

CUT THIS SLIDE TO ONE SENTENCE if you are running long.

---

<a id="slide-04"></a>

## Slide 04 — TraceWise: a memory check for AI systems

01:35-02:10 - Method and stack.

"Four steps. REPLAY a controlled history: dated facts, then later updates. PROBE it with targeted questions. VERIFY against an answer key built independently, by code - not one model grading another. EXPLAIN the failure with the evidence that caused it.

Underneath: Python, LiteLLM, Claude Opus, a deterministic scorer with a pytest suite, and a Streamlit dashboard over a JSONL log.

The important part is the bottom strip. Three memory backends - full history, a latest-per-key reducer, a token-budget trimmer - behind one frozen interface. Same model, same system prompt, every time. THE MEMORY LAYER IS THE ONLY VARIABLE."

STAGE: For a non-technical room the four verbs are enough. The stack line is for the engineers - say it fast, don't explain it.

---

<a id="slide-05"></a>

## Slide 05 — Live demonstration

02:10-03:45 - DEMO. 90 seconds, hard stop. Switch to the browser.

"A client told their adviser in January they were crypto-heavy and aggressive. In March they divested and turned conservative. In April, someone asks the assistant whether a high-risk crypto ETF is suitable.

Nobody typed anything wrong. The January note was true in January."

(Show: old fact, the correction, the probe.)

"Same question, three memory layers, side by side. Here's what each one actually answered. Here's what the dated answer key says was true in April."

(Open one real failure.)

"And this is the part that matters for governance - not just that it's wrong, but the exact context the memory layer handed the model. That's the evidence an assurance team needs to act."

IF THE DEMO MISBEHAVES: "the committed run is in the repo and the dashboard opens on it" - then go to slide 7. Do not debug on stage.

BEFORE YOU START: have the dashboard already open on the Opus run in a second tab.

---

<a id="slide-06"></a>

## Slide 06 — Before: you trust it. After: you can test it

03:45-04:20 - Before and after.

"Without a check, the wrong answer sounds right and cites the file. There's no systematic test for updates, so you find out late - from a customer, or a tribunal.

With TraceWise: the wrong answer is flagged, the old and the applicable fact are both shown, and the owner can investigate and retest.

We identify and evidence failures. We don't automatically repair them - that's deliberate. A human makes the call."

---

<a id="slide-07"></a>

## Slide 07 — What we measured

04:20-05:25 - THE SLIDE. Slow down. Point at the table, don't read it aloud.
NEVER CUT TIME FROM THIS SLIDE.

"A hundred dated questions. Three memory layers. Three hundred live answers on Claude Opus, same prompt throughout, scored by code.

Keep everything: eighty-eight out of a hundred. The latest-per-key reducer: eighty-six - at FORTY-FOUR PERCENT OF THE TOKENS. And trim-to-budget: TWENTY-FOUR.

That spread, twenty-four to eighty-eight, is the memory layer alone. Nothing else changed.

Now look at the cost. Trimming is the cheapest per query - and the most expensive per CORRECT answer, nearly five times the reducer. Because you pay for all forty-nine answers it couldn't give. 'Cheapest' is the wrong question.

And then the thing we didn't want." (POINT RIGHT.) "Our hypothesis was that stale facts poison answers. We tested it - removed the one line telling the model to prefer recent facts, re-ran all three hundred. Twelve answers broke. NONE OF THEM had a stale fact in context. Fifty-eight other answers did.

So the damage wasn't contamination. Every loss landed on the layer that DROPS facts. On this model, the adviser fails by forgetting - not by remembering too much.

We shipped the result that disagreed with us. A benchmark that only confirms what you hoped isn't measuring anything."

STAGE: That last line is your strongest moment. Say it and STOP. Let it sit before slide 8.

---

<a id="slide-08"></a>

## Slide 08 — Who uses it — and who benefits

05:25-06:05 - Users and governance.

"Today: the engineers who build these systems, and the assurance teams who have to sign them off before release. Test, record the evidence, flag the risk, human review.

Individuals testing their own agents is future work - it's labelled on the slide because it isn't built.

Main track is Governance and Compliance, supported by Explainability and Transparency. One honest limit: this is an assurance tool. It is not a compliance certification, and we don't claim it is."

CUT TO TWO SENTENCES if running long.

---

<a id="slide-09"></a>

## Slide 09 — One assurance engine, different ways to use it

06:05-06:40 - Roadmap and close.

"Today it's a browser prototype on controlled synthetic histories. Next is the developer integration - a Python package and a CI check, so memory gets retested every time a prompt or a retrieval rule changes. Longer term, an opt-in check for personal assistants.

Everything you've seen is in the repo: the logs, the answer key, and the script that reproduces the numbers on slide seven from a clean clone.

TEST AI MEMORY BEFORE DECISIONS DEPEND ON IT. Thank you."

---

<a id="slide-10"></a>

## Slide 10 — Appendix — references and Q&A

APPENDIX - DO NOT PRESENT. Here for judges' questions.

IF THEY ASK:

"Isn't this just evals?" - Model evals score the ANSWER. We score the CONTEXT the answer came from. A model that's perfect can still be handed a dead fact, and no model eval catches that, because the model did nothing wrong.

"Why synthetic data?" - One fictional client and an invented versioned policy, so there's no real client data and the answer key is provably correct. The method transfers to real approved histories; honest scoring depends on knowing ground truth, which we only do here.

"Only one model and a hundred questions - thin?" - Yes, and the limitations say so. Per-bucket samples run 7 to 34. We're not claiming a general law. We're claiming the memory layer is a measurable variable nobody was measuring, and here's the harness.

"Could you test my agent?" - If its memory is accessible and you can authorise a test history, yes. We cannot read hidden memory inside a third-party service, and we don't pretend to.

"Why Opus?" - We ran both. On Haiku the spread is wider, 22 to 92, because the reducer degrades more on the smaller model. Both logs are committed. We quote Opus because it's the harder case for our own argument.

"What does it cost to run?" - About seven dollars for all 300 answers on Opus. Rates come from a pricing file where every number carries its source and date; an unconfirmed model reports cost as unknown, never zero.

---
