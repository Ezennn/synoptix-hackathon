"""RegWatch AI — demo UI. Agent B.

Replays results/run_log.jsonl. Nothing on the Demo tab touches the network, so
the stage demo cannot fail on a cold API. The one live path is the "re-run
live" button, which is opt-in and degrades to a warning if the orchestrator is
not running.

    streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ui.replay import (  # noqa: E402
    DEFAULT_LOG, backend_order, columns_for, has_stale, highlight, load_log,
    probes, scoreboard, stale_by_phase,
)

ORCHESTRATOR = "http://127.0.0.1:8077"

STATUS_LABEL = {
    "ok": ("answered", "#30a46c"),
    "unavailable": ("not available", "#8b8d98"),
    "not_implemented": ("spike pending", "#8b8d98"),
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
</style>
"""


def main() -> None:
    st.set_page_config(page_title="RegWatch AI", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    records = load_log()
    if not records:
        st.title("RegWatch AI")
        st.warning(
            "No run log at `{}`. Generate a rehearsal log with "
            "`python scripts/seed_demo_log.py`, or run the real sweep.".format(DEFAULT_LOG)
        )
        return

    st.title("RegWatch AI")
    st.caption("Everyone tests the model. Nobody tests what the model was told.")

    all_probes = probes(records)
    answered = {r["backend"] for r in records if r["status"] == "ok"}
    default = [b for b in backend_order() if b in answered][:5] or backend_order()[:5]

    with st.sidebar:
        st.subheader("Columns")
        shown = st.multiselect("Backends", backend_order(), default=default)
        st.caption("{} probes · {} rows".format(len(all_probes), len(records)))
        live = st.toggle("Enable live re-run", value=False,
                         help="Calls the orchestrator. Off for the stage demo.")

    demo_tab, board_tab, rot_tab = st.tabs(["Demo", "Scoreboard", "Context rot"])

    with demo_tab:
        _demo(records, all_probes, shown, live)
    with board_tab:
        _scoreboard(records)
    with rot_tab:
        _rot(records, shown)


def _demo(records, all_probes, shown, live) -> None:
    labels = {"{}  —  {}".format(p["probe_id"], p["question"]): p for p in all_probes}
    chosen = labels[st.selectbox("Probe", list(labels))]

    st.markdown(
        '<div class="card"><h4>Client timeline</h4>'
        '<div class="ctx">Jan 2026 &nbsp;<span class="stale">crypto-heavy, aggressive</span>'
        '<br>Mar 2026 &nbsp;<span class="current">fully divested, conservative</span>'
        '</div></div>', unsafe_allow_html=True)

    if live:
        if st.button("Re-run live against the orchestrator"):
            _rerun_live(chosen)

    rows = [r for r in columns_for(records, chosen["probe_id"]) if r["backend"] in shown]
    if not rows:
        st.info("No backends selected.")
        return

    for column, record in zip(st.columns(len(rows)), rows):
        with column:
            _column(record, chosen)


def _column(record, probe) -> None:
    label, colour = STATUS_LABEL.get(record["status"], (record["status"], "#8b8d98"))
    st.markdown(
        '<div class="card"><h4>{}</h4>'
        '<span class="pill" style="color:{}">{}</span></div>'.format(
            record["backend"], colour, label),
        unsafe_allow_html=True)

    if record["status"] != "ok":
        st.markdown('<div class="meta">{}</div>'.format(record.get("error") or ""),
                    unsafe_allow_html=True)
        return

    context = "\n".join(record["retrieval_context"] or [])
    st.markdown(
        '<div class="card"><h4>What the model was told</h4><div class="ctx">{}</div></div>'
        .format(highlight(context, probe.get("superseded_answer"),
                          probe.get("expected_answer"))),
        unsafe_allow_html=True)

    stale = has_stale(record)
    verdict = ("STALE FACT REACHED THE MODEL", "#e5484d") if stale else \
              ("context current", "#30a46c") if stale is False else ("unknown", "#8b8d98")
    scores = record["scores"] or {}
    consistency = (scores.get("consistency") or {}).get("value")
    st.markdown(
        '<div class="card"><h4>Answer</h4><div class="answer">{}</div>'
        '<div class="verdict" style="color:{}">{}</div>'
        '<div class="meta">{} tok in · ${:.5f} · {:.0f} ms · consistency {}</div></div>'
        .format(record["answer"], verdict[1], verdict[0],
                scores.get("tokens_in", 0), scores.get("cost_usd", 0.0),
                scores.get("latency_ms", 0.0),
                "—" if consistency is None else consistency),
        unsafe_allow_html=True)


def _scoreboard(records) -> None:
    rows = []
    for row in scoreboard(records):
        rows.append({
            "backend": row["backend"],
            "answered": "{}/{}".format(row["answered"], row["probes"]),
            "stale context": _pct(row["stale_context_rate"]),
            "consistency": _pct(row["consistency_rate"]),
            "avg tokens in": _num(row["avg_tokens_in"]),
            "total cost $": "${:.5f}".format(row["total_cost_usd"]),
            "avg latency ms": _num(row["avg_latency_ms"]),
        })
    # Every cell is a string: a column mixing floats with the "—" placeholder
    # fails Arrow serialisation.
    st.dataframe(rows, width="stretch", hide_index=True)
    st.caption(
        "Stale context is deterministic: did the superseded phrase appear in the "
        "injected text. Consistency comes from Agent A's DAG metric and shows — "
        "until that scorer is on this machine. Context precision, faithfulness and "
        "recall are a separate offline pass (evaluator.score_offline)."
    )


def _rot(records, shown) -> None:
    rot = {b: v for b, v in stale_by_phase(records).items() if b in shown}
    if not rot:
        st.info("No scored contexts yet.")
        return
    import pandas as pd

    phases = sorted({p for series in rot.values() for p in series})
    frame = pd.DataFrame(
        {b: [series.get(p) for p in phases] for b, series in rot.items()},
        index=["phase {}".format(p) for p in phases],
    )
    st.bar_chart(frame, stack=False, y_label="stale context rate", x_label="")
    st.caption("Share of injected contexts containing the superseded fact. A backend "
               "sitting at zero has no visible bar — read it off the scoreboard.")


def _rerun_live(probe) -> None:
    import httpx
    try:
        response = httpx.post("{}/ask".format(ORCHESTRATOR), timeout=60, json={
            "session_id": "client_acme", "question": probe["question"],
            "probe_id": probe["probe_id"], "fact_id": probe.get("fact_id"),
            "expected_answer": probe.get("expected_answer"),
            "superseded_answer": probe.get("superseded_answer"),
            "phase": probe.get("phase"),
        })
        response.raise_for_status()
        st.success("Re-ran live; the log now has a fresh row.")
        st.rerun()
    except Exception as exc:
        st.error("Orchestrator unreachable at {} ({}). The replay below is "
                 "unaffected.".format(ORCHESTRATOR, type(exc).__name__))


def _pct(value):
    return "—" if value is None else "{:.0f}%".format(value * 100)


def _num(value):
    return "—" if value is None else "{:.1f}".format(value)


if __name__ == "__main__":
    main()
