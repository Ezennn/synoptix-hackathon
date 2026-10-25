"""RegWatch — AI memory assurance gate.

The dashboard a team runs before shipping an assistant: what we tested, where
each memory system fails, the evidence for any single failure, and what to
change. Replays results/run_log*.jsonl — nothing on the first four tabs touches
the network, so the stage demo cannot fail on a cold API.

    streamlit run app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ui.replay import (  # noqa: E402
    CAUSES, DEFAULT_LOG, TIERS, backend_order, by_tier, columns_for,
    efficiency_pick, failures, has_stale, highlight, load_log, probes, run_meta,
    scoreboard, stale_by_phase,
)

ORCHESTRATOR = "http://127.0.0.1:8077"
FROZEN_LOG = DEFAULT_LOG.parent / "run_log.opus.jsonl"

INK = "#121A2B"
PAPER = "#FBFBF8"
MUTED = "#6A7179"
WARN = "#C25E12"
GOOD = "#1F6F60"

STATUS_LABEL = {
    "ok": ("answered", GOOD),
    "unavailable": ("not available", MUTED),
    "not_implemented": ("spike pending", MUTED),
    "timeout": ("timed out", "#f5a524"),
    "error": ("error", "#e5484d"),
}

CSS = """
<style>
  .card { border:1px solid rgba(128,128,128,.28); border-radius:10px;
          padding:.7rem .8rem; margin-bottom:.6rem; }
  .card h4 { margin:0 0 .35rem 0; font-size:.95rem; }
  .ctx { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.76rem;
         line-height:1.45; white-space:pre-wrap; }
  .stale   { background:rgba(229,72,77,.22);  color:#e5484d; font-weight:600;
             border-radius:3px; padding:0 2px; }
  .current { background:rgba(48,163,108,.20); color:#30a46c; font-weight:600;
             border-radius:3px; padding:0 2px; }
  .answer { font-size:.84rem; line-height:1.5; margin-top:.2rem; }
  .meta { font-size:.72rem; color:#8b8d98; margin-top:.4rem; }
  .pill { display:inline-block; font-size:.68rem; padding:.08rem .45rem;
          border-radius:99px; border:1px solid currentColor; }
  .verdict { font-size:.78rem; font-weight:600; margin-top:.35rem; }
  .kv { font-size:.78rem; color:#8b8d98; }
  .big { font-size:1.6rem; font-weight:700; line-height:1.1; }
  .tag { display:inline-block; font-size:.66rem; font-weight:600; padding:.1rem .5rem;
         border-radius:99px; margin:0 .25rem .25rem 0; border:1px solid currentColor; }
  .scn h4 { margin:0 0 .2rem 0; font-size:.9rem; }
  .scn .answer { margin-top:.15rem; }
  .step { font-family:ui-monospace,Menlo,monospace; font-size:.74rem; color:#8b8d98; }
</style>
"""

# --------------------------------------------------------------------------
# What the benchmark tests, stated once so the UI can show it plainly.
# --------------------------------------------------------------------------

# The 4 client storylines (scenarios). Each is one entity's fact-timeline that
# mutates across the four sessions; a memory system is tested on whether it
# tracks that timeline correctly.
SCENARIOS = [
    {
        "key": "crypto_appetite",
        "title": "1 · Crypto risk appetite",
        "entity": "Acme Capital Partners LLP",
        "story": "Risk appetite for crypto flips aggressive → conservative in "
                 "session 2 and stays there.",
        "timeline": "S1 aggressive  →  S2 conservative  →  (holds)",
        "tests": "direct",
        "catches": "A forgetful backend still answers “aggressive” after the flip.",
    },
    {
        "key": "account_risk",
        "title": "2 · Scoped account risk tiers",
        "entity": "Acme pension vs Acme prop book",
        "story": "Only the PENSION sub-account turns conservative in session 2. "
                 "The prop book stays aggressive — a scoped change.",
        "timeline": "pension: S1 aggressive → S2 conservative   |   "
                    "propbook: aggressive (unchanged)",
        "tests": "partial",
        "catches": "A backend that overwrites globally wrongly flips the prop "
                   "book too.",
    },
    {
        "key": "subsidiary_divest",
        "title": "3 · Subsidiary divestment → crypto exposure",
        "entity": "Acme subsidiary B",
        "story": "Subsidiary B holds the crypto position (S1), then is divested "
                 "(S3). Nothing SAYS “no crypto exposure” — it must be inferred.",
        "timeline": "S1 subB holds crypto  →  S3 subB divested  ⟹  no exposure",
        "tests": "transitive",
        "catches": "A backend with no inference rule still answers “yes, crypto "
                   "exposure”. This is the hard, showcase scenario.",
    },
    {
        "key": "domicile",
        "title": "4 · Tax domicile reinstatement",
        "entity": "Acme Capital Partners LLP",
        "story": "Tax domicile moves UK → Luxembourg (S2), then back to UK (S4). "
                 "At the S3 ask-time the answer is Luxembourg.",
        "timeline": "S1 UK  →  S2 Luxembourg  →  S4 UK  (A/B/A)",
        "tests": "reinstatement",
        "catches": "A “latest-value-only” backend cannot answer the mid-timeline "
                   "ask (Luxembourg) correctly — it only knows the final value.",
    },
]

# The 5 mutation types (test parameters). These ARE the TIERS column in every
# chart; defined here with a plain-language gloss so the orientation tab and the
# heatmap use the same five words.
PARAMETERS = [
    {
        "tier": "baseline",
        "title": "Baseline (control)",
        "what": "A fact that NEVER changes (legal name, base currency).",
        "passes": "A system failing these has a problem unrelated to updates.",
    },
    {
        "tier": "direct",
        "title": "Direct overwrite",
        "what": "A value is replaced once; the question wants the current one.",
        "passes": "Tests plain recency — does the latest value win.",
    },
    {
        "tier": "partial",
        "title": "Partial / scoped overwrite",
        "what": "One sub-entity changes; a sibling must NOT be touched.",
        "passes": "Tests that an update is scoped, not applied globally.",
    },
    {
        "tier": "transitive",
        "title": "Transitive invalidation",
        "what": "The answer is never stated — it follows by inference from two "
                "facts (held-via + divested).",
        "passes": "Tests real inference. The ≤7% “nearly unsolved” tier.",
    },
    {
        "tier": "reinstatement",
        "title": "Reinstatement (A/B/A)",
        "what": "A value flips away and later flips back.",
        "passes": "Tests timeline tracking, not just the final value.",
    },
]

# Which colour a tier gets in the orientation tags (reused from the palette).
_TIER_COLOUR = {
    "baseline": MUTED, "direct": GOOD, "partial": "#2563EB",
    "transitive": WARN, "reinstatement": "#9333EA",
}


# --------------------------------------------------------------------------
# run selection
# --------------------------------------------------------------------------

def _available_logs():
    preferred = [DEFAULT_LOG, FROZEN_LOG]

    def rank(path):
        return (preferred.index(path) if path in preferred else len(preferred),
                path.name)

    return sorted(DEFAULT_LOG.parent.glob("run_log*.jsonl"), key=rank)


def _log_summary(path) -> str:
    try:
        with path.open() as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    break
            else:
                return "empty"
    except Exception:
        return "unreadable"
    mode = row.get("llm_mode") or "unknown"
    model = (row.get("model") or "?").split("/")[-1]
    if mode == "stub":
        return "stub · no model"
    hint = row.get("recency_hint")
    arm = "" if hint is None else (" · hint ON" if hint else " · hint OFF")
    return "{} · {}{}".format(mode, model, arm)


# --------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="RegWatch", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    logs = _available_logs()
    if not logs:
        st.title("RegWatch")
        st.warning("No run log in `{}`. Run `python scripts/run_benchmark.py`."
                   .format(DEFAULT_LOG.parent))
        return

    with st.sidebar:
        st.subheader("Run")
        chosen = st.selectbox(
            "Which run", logs,
            format_func=lambda p: p.stem.replace("run_log", "") or "current",
            help="Each log is one benchmark run. The caption names the mode, "
                 "model and prompt arm that produced it.")
        st.caption(_log_summary(chosen))

    records = load_log(chosen)
    if not records:
        st.title("RegWatch")
        st.warning("`{}` is empty.".format(chosen))
        return

    meta = run_meta(records)
    board = scoreboard(records)
    answered = {r["backend"] for r in records if r["status"] == "ok"}
    default = [b for b in backend_order() if b in answered][:5] or backend_order()[:5]

    with st.sidebar:
        st.subheader("Columns")
        shown = st.multiselect("Backends", backend_order(), default=default)
        live = st.toggle("Enable live re-run", value=False,
                         help="Calls the orchestrator. Off for the stage demo.")
        st.divider()
        st.caption("**Under test**")
        st.caption("{} · {}".format(meta.get("model") or "—",
                                    "live" if meta.get("mode") == "live" else "stub"))
        st.caption("**Stack**")
        st.caption("LiteLLM gateway · LangGraph reducer · deterministic scorer · "
                   "FastAPI + Streamlit")

    _header(meta)

    tabs = st.tabs(["Overview", "Evaluation", "Cost & efficiency",
                    "Failure explorer", "Recommendation", "Side-by-side",
                    "Context rot", "Data & logs"])
    with tabs[0]:
        _overview(records, meta)
    with tabs[1]:
        _evaluation(records, board)
    with tabs[2]:
        _cost_efficiency(records, board, meta)
    with tabs[3]:
        _failure_explorer(records)
    with tabs[4]:
        _recommendation(records, board)
    with tabs[5]:
        _side_by_side(records, probes(records), shown, live)
    with tabs[6]:
        _rot(records, shown)
    with tabs[7]:
        _data_and_logs(records)


def _header(meta) -> None:
    st.title("RegWatch")
    st.caption("Does your assistant still know what changed? "
               "An assurance gate for AI memory, run before customers see it.")
    cols = st.columns(4)
    cols[0].metric("Questions", meta.get("probes", 0))
    cols[1].metric("Memory systems", meta.get("backends", 0))
    cols[2].metric("Answers scored", meta.get("answers", 0))
    hint = meta.get("recency_hint")
    # Short value: st.metric truncates anything longer.
    cols[3].metric("Recency hint", "—" if hint is None else ("ON" if hint else "OFF"))
    st.caption("Model under test: **{}** · every backend answers with the same "
               "model and the same system prompt, so the only variable is the "
               "memory layer.".format(meta.get("model") or "—"))
    st.divider()


# --------------------------------------------------------------------------
# 0. overview — what the benchmark tests (4 scenarios × 5 parameters)
# --------------------------------------------------------------------------

def _overview(records, meta) -> None:
    st.subheader("What this benchmark tests")
    st.caption("Four client storylines (scenarios) are each probed along five "
               "mutation types (test parameters). Every memory system under test "
               "answers the same questions with the same model — the only "
               "variable is how it remembers.")

    answered = sorted({r["backend"] for r in records if r["status"] == "ok"})
    cols = st.columns(3)
    cols[0].metric("Scenarios", len(SCENARIOS))
    cols[1].metric("Test parameters", len(PARAMETERS))
    cols[2].metric("Memory systems in this run", len(answered) or meta.get("backends", 0))

    st.divider()
    st.markdown("### The 4 scenarios")
    st.caption("Each is one entity's fact-timeline across sessions S1–S4.")
    grid = st.columns(2)
    for i, scn in enumerate(SCENARIOS):
        colour = _TIER_COLOUR.get(scn["tests"], MUTED)
        with grid[i % 2]:
            st.markdown(
                '<div class="card scn" style="border-color:{c}">'
                '<h4>{title}</h4>'
                '<span class="tag" style="color:{c}">parameter: {tier}</span>'
                '<div class="answer"><b>Entity:</b> {entity}</div>'
                '<div class="answer">{story}</div>'
                '<div class="step">{timeline}</div>'
                '<div class="answer" style="margin-top:.4rem;color:{w}">'
                '<b>Catches:</b> {catches}</div>'
                '</div>'.format(c=colour, w=WARN, title=scn["title"],
                                tier=scn["tests"], entity=scn["entity"],
                                story=scn["story"], timeline=scn["timeline"],
                                catches=scn["catches"]),
                unsafe_allow_html=True)

    st.divider()
    st.markdown("### The 5 test parameters")
    st.caption("These are the five difficulty tiers scored in every chart on the "
               "Evaluation tab — rising from a never-changing control to inference.")
    for prm in PARAMETERS:
        colour = _TIER_COLOUR.get(prm["tier"], MUTED)
        st.markdown(
            '<div class="card" style="border-left:4px solid {c}">'
            '<span class="tag" style="color:{c}">{tier}</span>'
            '<b>{title}</b>'
            '<div class="answer">{what}</div>'
            '<div class="meta">{passes}</div>'
            '</div>'.format(c=colour, tier=prm["tier"], title=prm["title"],
                            what=prm["what"], passes=prm["passes"]),
            unsafe_allow_html=True)

    st.divider()
    st.markdown("### Coverage — scenarios × parameters")
    st.caption("How many probes in this run exercise each (scenario-parameter) "
               "cell. Baseline and the two-timeline compliance probes span "
               "scenarios, so they are counted under their parameter.")
    _coverage_matrix(records)


def _coverage_matrix(records) -> None:
    """Probe count per (scenario, parameter), from the loaded run."""
    # Map each probe to a scenario by the tier it exercises (1:1 with the four
    # non-baseline storylines); baseline probes form their own row.
    tier_to_scenario = {s["tests"]: s["title"] for s in SCENARIOS}
    seen = {}
    rows = {}
    for rec in records:
        probe = rec["probe"]
        pid = probe["probe_id"]
        if pid in seen:
            continue
        seen[pid] = True
        tier = probe.get("tier") or "unknown"
        scenario = tier_to_scenario.get(tier, "— (control / cross-scenario)")
        rows.setdefault(scenario, {t: 0 for t in TIERS})
        if tier in rows[scenario]:
            rows[scenario][tier] += 1
    if not rows:
        st.info("No probes in this run.")
        return
    frame = pd.DataFrame(
        [{"scenario": scn, **{t: counts[t] for t in TIERS}}
         for scn, counts in rows.items()])
    st.dataframe(frame, width="stretch", hide_index=True)


# --------------------------------------------------------------------------
# 7. data & logs — the full questions asked and the data fed to the models
# --------------------------------------------------------------------------

def _data_and_logs(records) -> None:
    st.subheader("Everything fed in, and every question asked")
    st.caption("The benchmark's INPUTS, in full — so nothing about what the "
               "models saw is hidden. Exported copies live in `results/`.")

    all_probes = probes(records)
    distinct = len({(p["question"], p.get("ask_after_session"),
                     tuple(p.get("fact_ids") or [])) for p in all_probes})
    c = st.columns(3)
    c[0].metric("Probe rows", len(all_probes))
    c[1].metric("Distinct tests", distinct)
    c[2].metric("Distinct question texts",
                len({p["question"] for p in all_probes}))
    st.caption("“Probe rows” is the raw count; “distinct tests” collapses rows "
               "that share a question, ask-step and target fact. The same "
               "wording asked after a LATER session is a different test, because "
               "the correct answer changed when the fact mutated — that is why "
               "the Failure Explorer shows repeated-looking questions.")

    st.divider()
    st.markdown("### Full question log")
    st.caption("Every probe with the session it was asked after and the answer "
               "that was correct at that moment. Sort by question to see one "
               "wording tracked across the timeline.")
    qrows = []
    for p in sorted(all_probes, key=lambda x: x["probe_id"]):
        qrows.append({
            "probe_id": p["probe_id"],
            "phase": p.get("phase"),
            "tier (parameter)": p.get("tier"),
            "after session": p.get("ask_after_session"),
            "ask date": p.get("ask_date"),
            "multi-hop": "yes" if p.get("multi_hop") else "",
            "question": p.get("question"),
            "expected now": p.get("expected_answer"),
            "superseded": " | ".join(p.get("superseded_values") or []),
        })
    st.dataframe(qrows, width="stretch", hide_index=True, height=360)
    csv_path = Path("results/question_log.csv")
    if csv_path.exists():
        st.download_button("Download question_log.csv", csv_path.read_bytes(),
                           file_name="question_log.csv", mime="text/csv")

    st.divider()
    st.markdown("### Data feed — what each backend ingested")
    st.caption("The four session transcripts in order: session 1 is loaded, "
               "sessions 2–4 are applied as updates. Distractors are irrelevant "
               "notes fed in alongside to make retrieval harder.")
    feed_path = Path("results/data_feed.md")
    if feed_path.exists():
        with st.expander("Show the full data feed (all four sessions)", expanded=False):
            st.markdown(feed_path.read_text(encoding="utf-8"))
        st.download_button("Download data_feed.md", feed_path.read_bytes(),
                           file_name="data_feed.md", mime="text/markdown")
    else:
        st.info("Run `python scripts/export_inputs.py` to generate the data feed.")

    st.divider()
    st.markdown("### Per-turn context dump")
    st.caption("For one question and one memory system, the EXACT text injected "
               "into the model this turn — the same evidence the Failure "
               "Explorer shows, browsable for any probe (not only failures).")
    labels = {"{}  —  {}".format(p["probe_id"], p["question"][:70]): p
              for p in all_probes}
    chosen = labels[st.selectbox("Question", list(labels))]
    turn_rows = columns_for(records, chosen["probe_id"])
    for r in turn_rows:
        ctx = r.get("retrieval_context") or []
        chars = sum(len(x) for x in ctx)
        with st.expander("{} — {} items, {} chars  ({})".format(
                r["backend"], len(ctx), chars, r["status"])):
            if ctx:
                st.code("\n".join(ctx), language="text")
            else:
                st.write("_(empty context — backend exposed nothing this turn)_")
            st.caption("answer: {}".format(r.get("answer") or "—"))


# --------------------------------------------------------------------------
# 1. evaluation
# --------------------------------------------------------------------------

def _evaluation(records, board) -> None:
    st.subheader("How often was the answer still correct?")
    st.caption("State consistency: the answer asserts the value that was true at "
               "the time the question was asked, and not a superseded one.")

    frame = pd.DataFrame([
        {"backend": r["backend"],
         "consistency": (r["consistency_rate"] or 0) * 100,
         "correct": r["correct"]}
        for r in board if r["answered"]
    ])
    if frame.empty:
        st.info("No scored answers in this run.")
        return

    bars = alt.Chart(frame).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("consistency:Q", title="correct (%)", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("backend:N", sort="-x", title=None),
        color=alt.Color("consistency:Q", legend=None,
                        scale=alt.Scale(scheme="teals", domain=[0, 100])),
        tooltip=["backend", "correct", "consistency"],
    ).properties(height=46 * len(frame))
    labels = bars.mark_text(align="left", dx=6, color="#8b8d98").encode(
        text=alt.Text("consistency:Q", format=".0f"))
    st.altair_chart(bars + labels, use_container_width=True)

    st.subheader("Where does each method break?")
    st.caption("Consistency by question type. Baseline facts never change — a "
               "system failing those has a problem unrelated to updates.")
    tiers = by_tier(records)
    rows = [{"backend": b, "tier": t, "rate": (tiers[b].get(t) or 0) * 100}
            for b in tiers for t in TIERS if t in tiers[b]]
    if rows:
        heat_frame = pd.DataFrame(rows)
        heat = alt.Chart(heat_frame).mark_rect().encode(
            x=alt.X("tier:N", sort=TIERS, title=None),
            y=alt.Y("backend:N", title=None),
            color=alt.Color("rate:Q", title="correct (%)",
                            scale=alt.Scale(scheme="teals", domain=[0, 100])),
            tooltip=["backend", "tier", alt.Tooltip("rate:Q", format=".0f")],
        ).properties(height=52 * len(tiers))
        text = heat.mark_text(baseline="middle", fontSize=13).encode(
            text=alt.Text("rate:Q", format=".0f"),
            color=alt.condition("datum.rate > 55", alt.value("#0d1b17"),
                                alt.value("#d8dee9")))
        st.altair_chart(heat + text, use_container_width=True)

    st.subheader("Reliability against cost")
    st.caption("Tokens per query is the cost you pay on every call. The useful "
               "question is not which is cheapest, but what each extra token buys.")
    scatter_frame = pd.DataFrame([
        {"backend": r["backend"], "tokens": r["avg_tokens_in"] or 0,
         "consistency": (r["consistency_rate"] or 0) * 100}
        for r in board if r["answered"]
    ])
    points = alt.Chart(scatter_frame).mark_circle(size=320, opacity=0.85).encode(
        x=alt.X("tokens:Q", title="input tokens per query"),
        y=alt.Y("consistency:Q", title="correct (%)", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("backend:N", legend=None),
        tooltip=["backend", "tokens", "consistency"],
    )
    names = points.mark_text(align="left", dx=12, fontSize=12, color="#8b8d98").encode(
        text="backend:N")
    st.altair_chart((points + names).properties(height=320), use_container_width=True)

    _judge_metrics(board)

    with st.expander("Full scoreboard"):
        _scoreboard_table(board)


def _judge_metrics(board) -> None:
    """Axes 2 & 3 — DeepEval Contextual Relevancy + Recall, per backend.

    These are LLM-judged and scored OFFLINE by evaluator.score_offline(). Until
    `deepeval` is installed and that pass has run, every value is None — so this
    section says so plainly instead of rendering empty bars that look like zero.
    """
    st.subheader("Judge metrics — retrieval quality (axes 2 & 3)")
    st.caption("Scored by DeepEval over each backend's retrieval_context: "
               "**Contextual Recall** = did the facts needed for the answer "
               "survive in the injected context; **Contextual Relevancy** = what "
               "fraction of the injected context was actually relevant (order-"
               "free, so it fits backends that don't rank). Faithfulness is "
               "excluded from the headline — it scores the answer model, which "
               "is identical across all backends.")

    scored = [r for r in board if r["answered"]]
    any_judge = any((r.get("judge_scored") or 0) > 0 for r in scored)
    if not any_judge:
        st.info("**Not scored in this run.** The judge metrics need `deepeval` "
                "installed and a live run — `deepeval` requires Python 3.11+ "
                "(this host is on 3.9) and a judge API key, and a stub run has no "
                "real answers to judge. The columns, DB fields and this panel are "
                "wired and will populate once `evaluator.score_offline()` runs "
                "against a live log. Run: `python scripts/run_benchmark.py "
                "--mode live` then the offline scorer.")
        # Still show the table so the five/four backends are visibly listed as
        # awaiting a score, rather than the section looking broken.
        st.dataframe([{
            "backend": r["backend"],
            "contextual recall": "unscored",
            "contextual relevancy": "unscored",
            "turns judged": "{}/{}".format(r.get("judge_scored") or 0, r["answered"]),
        } for r in scored], width="stretch", hide_index=True)
        return

    frame = pd.DataFrame([
        {"backend": r["backend"], "metric": label, "value": (val or 0) * 100}
        for r in scored
        for label, val in (("Contextual recall", r.get("contextual_recall")),
                           ("Contextual relevancy", r.get("contextual_relevancy")))
        if val is not None
    ])
    if not frame.empty:
        chart = alt.Chart(frame).mark_bar().encode(
            x=alt.X("value:Q", title="score (%)", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("backend:N", title=None),
            color=alt.Color("metric:N", title=None),
            yOffset="metric:N",
            tooltip=["backend", "metric", alt.Tooltip("value:Q", format=".0f")],
        ).properties(height=60 * len(scored))
        st.altair_chart(chart, use_container_width=True)
    st.dataframe([{
        "backend": r["backend"],
        "contextual recall": _pct(r.get("contextual_recall")),
        "contextual relevancy": _pct(r.get("contextual_relevancy")),
        "faithfulness": _pct(r.get("faithfulness")),
        "turns judged": "{}/{}".format(r.get("judge_scored") or 0, r["answered"]),
    } for r in scored], width="stretch", hide_index=True)


def _cost_efficiency(records, board, meta) -> None:
    scored = [r for r in board if r["answered"]]
    if not scored:
        st.info("Nothing scored in this run.")
        return

    pricing = (records[0] or {}).get("pricing") if records else None
    if pricing:
        st.caption("Priced at ${:.2f} in / ${:.2f} out per million tokens — "
                   "source: {} (as of {}).".format(
                       pricing.get("input_per_m"), pricing.get("output_per_m"),
                       pricing.get("source"), pricing.get("as_of")))
    else:
        st.warning("No confirmed token rate for **{}**, so every dollar figure "
                   "reads unknown rather than zero. Add it to `config/pricing.yaml`, "
                   "or set REGWATCH_PRICE_IN_PER_M / _OUT_PER_M."
                   .format(meta.get("model") or "this model"))

    unknown_cache = sum(r["fresh_calls"] + r["cached_calls"] for r in scored) == 0
    if unknown_cache:
        st.info("This run predates cache tracking, so latency below blends real "
                "calls with cache reads — note the median against the p95. "
                "Runs from now on record it per answer.")

    st.subheader("Cost and speed per memory system")
    rows = []
    for row in board:
        if not row["answered"]:
            continue
        rows.append({
            "backend": row["backend"],
            "correct": "{}/{}".format(row["correct"], row["answered"]),
            "tokens in": _num(row["avg_tokens_in"]),
            "tokens out": _num(row["avg_tokens_out"]),
            "tokens total": _num(row["avg_total_tokens"]),
            "latency mean": _num(row["avg_latency_ms"]),
            "median": _num(row["median_latency_ms"]),
            "p95": _num(row["p95_latency_ms"]),
            "fresh calls": "{}".format(row["fresh_calls"] or "—"),
            "$ / 1k questions": _usd(row["cost_per_1k_usd"]),
            "$ / correct answer": _usd(row["cost_per_correct_usd"]),
        })
    st.dataframe(rows, width="stretch", hide_index=True)
    st.caption("Answer generation only. The in-process backends make no model "
               "calls while ingesting, so their ingestion cost is a measured "
               "zero; a hosted memory service would report it separately.")

    st.subheader("Is the cheap one good enough?")
    # Whole percents: a 0-1 float with a "%.0f%%" format renders 0.80 as "1%".
    threshold = st.slider("Minimum acceptable accuracy", 0, 100, 80, 5,
                          format="%d%%") / 100.0
    pick = efficiency_pick(board, threshold)
    if pick["chosen"]:
        st.success("**{}** — {} ({:.0%} correct, {:.0f} tokens per query)".format(
            pick["chosen"]["backend"], pick["reason"],
            pick["chosen"]["accuracy"], pick["chosen"]["tokens"]))
    else:
        st.error(pick["reason"])
    st.dataframe([{
        "backend": c["backend"], "accuracy": "{:.0%}".format(c["accuracy"]),
        "tokens / query": _num(c["tokens"]),
        "meets bar": "yes" if c["meets"] else "no",
    } for c in pick["candidates"]], width="stretch", hide_index=True)

    st.subheader("Accuracy against cost")
    priced = all(r["cost_per_1k_usd"] is not None for r in scored)
    frame = pd.DataFrame([{
        "backend": r["backend"],
        "cost": r["cost_per_1k_usd"] if priced else (r["avg_tokens_in"] or 0),
        "accuracy": (r["consistency_rate"] or 0) * 100,
    } for r in scored])
    axis = "$ per 1,000 questions" if priced else "input tokens per query"
    points = alt.Chart(frame).mark_circle(size=320, opacity=0.85).encode(
        x=alt.X("cost:Q", title=axis),
        y=alt.Y("accuracy:Q", title="correct (%)", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("backend:N", legend=None),
        tooltip=["backend", "cost", "accuracy"])
    names = points.mark_text(align="left", dx=12, fontSize=12, color="#8b8d98").encode(
        text="backend:N")
    st.altair_chart((points + names).properties(height=320), use_container_width=True)


def _scoreboard_table(board) -> None:
    rows = []
    for row in board:
        rows.append({
            "backend": row["backend"],
            "answered": "{}/{}".format(row["answered"], row["probes"]),
            "correct": "{}".format(row["correct"]),
            "consistency": _pct(row["consistency_rate"]),
            "ctx recall": _pct(row.get("contextual_recall")),
            "ctx relevancy": _pct(row.get("contextual_relevancy")),
            "stale context": _pct(row["stale_context_rate"]),
            "tokens in": _num(row["avg_tokens_in"]),
            "tokens out": _num(row["avg_tokens_out"]),
            "latency ms": _num(row["avg_latency_ms"]),
            "p95 ms": _num(row["p95_latency_ms"]),
            "$ / 1k questions": _usd(row["cost_per_1k_usd"]),
            "$ / correct answer": _usd(row["cost_per_correct_usd"]),
        })
    st.dataframe(rows, width="stretch", hide_index=True)
    st.caption("Stale context is deterministic: did the superseded phrase appear "
               "in the injected text. Cost shows — when the model's token rate is "
               "not configured, rather than a figure we cannot stand behind.")


# --------------------------------------------------------------------------
# 2. failure explorer
# --------------------------------------------------------------------------

def _failure_explorer(records) -> None:
    found = failures(records)
    if not found:
        st.success("No failures in this run.")
        return

    st.subheader("What actually went wrong")
    st.caption("{} failures. The cause is read off the evidence — what the "
               "context held — never assumed.".format(len(found)))

    counts = {}
    for item in found:
        counts[item["cause"]] = counts.get(item["cause"], 0) + 1
    cause_frame = pd.DataFrame([
        {"cause": CAUSES[c][0], "n": n} for c, n in
        sorted(counts.items(), key=lambda kv: -kv[1])
    ])
    chart = alt.Chart(cause_frame).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("n:Q", title="failures"),
        y=alt.Y("cause:N", sort="-x", title=None),
        color=alt.Color("n:Q", legend=None, scale=alt.Scale(scheme="oranges")),
        tooltip=["cause", "n"],
    ).properties(height=44 * len(cause_frame))
    st.altair_chart(chart, use_container_width=True)
    st.divider()

    left, right = st.columns(2)
    backends = sorted({i["record"]["backend"] for i in found})
    pick_backend = left.selectbox("Memory system", backends)
    causes_here = sorted({i["cause"] for i in found
                          if i["record"]["backend"] == pick_backend})
    pick_cause = right.selectbox("Cause", causes_here,
                                 format_func=lambda c: CAUSES[c][0])

    subset = [i for i in found if i["record"]["backend"] == pick_backend
              and i["cause"] == pick_cause]
    labels = {"{} — {}".format(i["record"]["probe"]["probe_id"],
                               i["record"]["probe"]["question"][:70]): i
              for i in subset}
    chosen = labels[st.selectbox("Failed question ({})".format(len(subset)), list(labels))]
    _failure_detail(chosen)


def _failure_detail(item) -> None:
    record, cause = item["record"], item["cause"]
    probe = record["probe"]
    summary, action = CAUSES[cause]

    st.markdown(
        '<div class="card"><h4>Fact history</h4><div class="ctx">{}</div></div>'.format(
            _fact_history(probe)), unsafe_allow_html=True)

    cols = st.columns(2)
    cols[0].markdown(
        '<div class="card"><h4>Question asked</h4><div class="answer">{}</div>'
        '<div class="meta">{} · asked after session {}</div></div>'.format(
            probe["question"], probe.get("ask_date") or "", probe.get("ask_after_session")),
        unsafe_allow_html=True)
    cols[1].markdown(
        '<div class="card"><h4>Expected answer</h4>'
        '<div class="answer"><span class="current">{}</span></div></div>'.format(
            probe.get("expected_answer") or "—"), unsafe_allow_html=True)

    st.markdown(
        '<div class="card"><h4>What the assistant answered</h4>'
        '<div class="answer">{}</div></div>'.format(record.get("answer") or "—"),
        unsafe_allow_html=True)

    context = "\n".join(record.get("retrieval_context") or []) or "(empty context)"
    st.markdown(
        '<div class="card"><h4>Evidence — what the model was told</h4>'
        '<div class="ctx">{}</div></div>'.format(
            highlight(context, probe.get("superseded_values"),
                      probe.get("expected_answer"))), unsafe_allow_html=True)

    st.markdown(
        '<div class="card" style="border-color:{}"><h4 style="color:{}">Diagnosis</h4>'
        '<div class="answer">{}</div>'
        '<div class="answer" style="margin-top:.5rem"><b>Suggested next action.</b> {}</div>'
        '</div>'.format(WARN, WARN, summary, action), unsafe_allow_html=True)


def _fact_history(probe) -> str:
    """The timeline this question sits on, from its own answer key."""
    lines = []
    for value in probe.get("superseded_values") or []:
        lines.append('<span class="stale">{}</span>  — superseded'.format(value))
    expected = probe.get("expected_answer")
    if expected:
        lines.append('<span class="current">{}</span>  — current at {}'.format(
            expected, probe.get("ask_date") or "ask time"))
    if not lines:
        lines.append("(control — this fact never changes)")
    return "<br>".join(lines)


# --------------------------------------------------------------------------
# 3. recommendation
# --------------------------------------------------------------------------

def _recommendation(records, board) -> None:
    scored = [r for r in board if r["answered"] and r["consistency_rate"] is not None]
    if not scored:
        st.info("Nothing scored in this run.")
        return
    best = max(scored, key=lambda r: r["consistency_rate"])
    cheapest = min(scored, key=lambda r: r["avg_tokens_in"] or 0)

    st.subheader("What we would change")
    cols = st.columns(2)
    cols[0].markdown(
        '<div class="card" style="border-color:{}"><h4 style="color:{}">Most reliable'
        '</h4><div class="big">{}</div><div class="answer">{} of {} correct · '
        '{:.0f} tokens per query</div></div>'.format(
            GOOD, GOOD, best["backend"], best["correct"], best["answered"],
            best["avg_tokens_in"] or 0), unsafe_allow_html=True)
    cols[1].markdown(
        '<div class="card" style="border-color:{}"><h4 style="color:{}">Cheapest'
        '</h4><div class="big">{}</div><div class="answer">{} of {} correct · '
        '{:.0f} tokens per query</div></div>'.format(
            WARN, WARN, cheapest["backend"], cheapest["correct"],
            cheapest["answered"], cheapest["avg_tokens_in"] or 0),
        unsafe_allow_html=True)

    if best["backend"] != cheapest["backend"]:
        extra = (best["avg_tokens_in"] or 0) - (cheapest["avg_tokens_in"] or 0)
        gain = (best["correct"] or 0) - (cheapest["correct"] or 0)
        st.markdown(
            '<div class="card"><h4>The trade</h4><div class="answer">'
            'Moving from <b>{}</b> to <b>{}</b> buys <b>{} more correct answers</b> '
            'per 100 questions, for <b>{:.0f} more input tokens</b> per query. '
            'Whether that is worth it is a decision for the team that owns the '
            'risk — but it should be made with these numbers in front of them.'
            '</div></div>'.format(cheapest["backend"], best["backend"], gain, extra),
            unsafe_allow_html=True)

    found = failures(records)
    counts = {}
    for item in found:
        counts[item["cause"]] = counts.get(item["cause"], 0) + 1
    if counts:
        top_cause = max(counts, key=counts.get)
        st.markdown(
            '<div class="card"><h4>Dominant failure mode</h4><div class="answer">'
            '<b>{} of {} failures</b>: {}</div>'
            '<div class="answer" style="margin-top:.5rem">{}</div></div>'.format(
                counts[top_cause], len(found), CAUSES[top_cause][0],
                CAUSES[top_cause][1]), unsafe_allow_html=True)

    st.markdown(
        '<div class="card"><h4>Free fix, measured</h4><div class="answer">'
        'Keeping “If facts conflict, prefer the most recent” in the system prompt '
        'is worth <b>1–4 points</b> of accuracy across every memory system we '
        'tested, and costs nothing. Compare the <code>.recency-on</code> and '
        '<code>.recency-off</code> runs in the sidebar.</div></div>',
        unsafe_allow_html=True)

    st.caption("Re-run after any change to the memory layer and keep this as a "
               "regression test: `python scripts/run_benchmark.py --mode live`.")


# --------------------------------------------------------------------------
# 4. side-by-side demo
# --------------------------------------------------------------------------

def _side_by_side(records, all_probes, shown, live) -> None:
    labels = {"{}  —  {}".format(p["probe_id"], p["question"]): p for p in all_probes}
    chosen = labels[st.selectbox("Probe", list(labels))]

    if live and st.button("Re-run live against the orchestrator"):
        _rerun_live(chosen)

    rows = [r for r in columns_for(records, chosen["probe_id"]) if r["backend"] in shown]
    if not rows:
        st.info("No backends selected.")
        return
    for column, record in zip(st.columns(len(rows)), rows):
        with column:
            _column(record, chosen)


def _column(record, probe) -> None:
    label, colour = STATUS_LABEL.get(record["status"], (record["status"], MUTED))
    st.markdown(
        '<div class="card"><h4>{}</h4><span class="pill" style="color:{}">{}</span>'
        '</div>'.format(record["backend"], colour, label), unsafe_allow_html=True)

    if record["status"] != "ok":
        st.markdown('<div class="meta">{}</div>'.format(record.get("error") or ""),
                    unsafe_allow_html=True)
        return

    context = "\n".join(record["retrieval_context"] or [])
    st.markdown(
        '<div class="card"><h4>What the model was told</h4><div class="ctx">{}</div></div>'
        .format(highlight(context, probe.get("superseded_values"),
                          probe.get("expected_answer"))), unsafe_allow_html=True)

    stale = has_stale(record)
    scores = record["scores"] or {}
    consistency_block = scores.get("consistency") or {}
    consistency = consistency_block.get("value")
    if consistency_block.get("abstained"):
        verdict = ("COULD NOT ANSWER — fact missing", "#f5a524")
    elif stale:
        verdict = ("STALE FACT REACHED THE MODEL", "#e5484d")
    elif stale is False:
        verdict = ("context current", "#30a46c")
    elif not (record["probe"] or {}).get("superseded_values"):
        verdict = ("control — nothing to supersede", MUTED)
    else:
        verdict = ("unknown", MUTED)

    st.markdown(
        '<div class="card"><h4>Answer</h4><div class="answer">{}</div>'
        '<div class="verdict" style="color:{}">{}</div>'
        '<div class="meta">{} tok in · {} · {} ms · consistency {}</div></div>'
        .format(record["answer"], verdict[1], verdict[0],
                scores.get("tokens_in") or 0, _usd(scores.get("cost_usd")),
                _num(scores.get("latency_ms")),
                "—" if consistency is None else consistency),
        unsafe_allow_html=True)


def _rot(records, shown) -> None:
    rot = {b: v for b, v in stale_by_phase(records).items() if b in shown}
    if not rot:
        st.info("No scored contexts yet.")
        return
    phases = sorted({p for series in rot.values() for p in series})
    frame = pd.DataFrame(
        {b: [series.get(p) for p in phases] for b, series in rot.items()},
        index=["phase {}".format(p) for p in phases])
    st.bar_chart(frame, stack=False, y_label="stale context rate", x_label="")
    st.caption("Share of injected contexts containing the superseded fact. A "
               "backend sitting at zero has no visible bar.")


def _rerun_live(probe) -> None:
    import httpx
    try:
        response = httpx.post("{}/ask".format(ORCHESTRATOR), timeout=60, json={
            "session_id": "client_acme", "question": probe["question"],
            "probe_id": probe["probe_id"], "fact_ids": probe.get("fact_ids") or [],
            "expected_answer": probe.get("expected_answer"),
            "superseded_values": probe.get("superseded_values") or [],
            "phase": probe.get("phase"), "tier": probe.get("tier"),
        })
        response.raise_for_status()
        st.success("Re-ran live; the log now has a fresh row.")
        st.rerun()
    except Exception as exc:
        st.error("Orchestrator unreachable at {} ({}). The replay is unaffected."
                 .format(ORCHESTRATOR, type(exc).__name__))


def _pct(value):
    return "—" if value is None else "{:.0f}%".format(value * 100)


def _num(value):
    return "—" if value is None else "{:.1f}".format(value)


def _usd(value):
    return "—" if value is None else "${:.4f}".format(value)


if __name__ == "__main__":
    main()
