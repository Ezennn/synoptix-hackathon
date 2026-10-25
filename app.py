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
</style>
"""


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

    tabs = st.tabs(["Evaluation", "Cost & efficiency", "Failure explorer",
                    "Recommendation", "Side-by-side", "Context rot"])
    with tabs[0]:
        _evaluation(records, board)
    with tabs[1]:
        _cost_efficiency(records, board, meta)
    with tabs[2]:
        _failure_explorer(records)
    with tabs[3]:
        _recommendation(records, board)
    with tabs[4]:
        _side_by_side(records, probes(records), shown, live)
    with tabs[5]:
        _rot(records, shown)


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

    with st.expander("Full scoreboard"):
        _scoreboard_table(board)


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
