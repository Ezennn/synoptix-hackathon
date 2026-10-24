"""RegWatch AI — demo UI. Agent B.

Replays results/run_log.jsonl. Nothing on the Demo tab touches the network, so
the stage demo cannot fail on a cold API. The one live path is the "re-run
live" button, which is opt-in and degrades to a warning if the orchestrator is
not running.

    streamlit run app.py
"""
from __future__ import annotations

import json
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


def _available_logs():
    """Every run log in results/, newest-looking first but with the default
    log pinned to the top so the stage demo opens on it."""
    logs = sorted(DEFAULT_LOG.parent.glob("run_log*.jsonl"))
    logs.sort(key=lambda p: (p != DEFAULT_LOG, p.name))
    return logs


def _log_summary(path) -> str:
    """Provenance from the first row — mode, model and which A/B arm.

    Reads one line, not the whole file: the point is to make it impossible to
    demo the wrong arm by accident.
    """
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


def main() -> None:
    st.set_page_config(page_title="RegWatch AI", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    logs = _available_logs()
    if not logs:
        st.title("RegWatch AI")
        st.warning(
            "No run log in `{}`. Generate a rehearsal log with "
            "`python scripts/seed_demo_log.py`, or run "
            "`python scripts/run_benchmark.py`.".format(DEFAULT_LOG.parent)
        )
        return

    with st.sidebar:
        st.subheader("Run")
        chosen = st.selectbox(
            "Which run", logs, format_func=lambda p: p.stem.replace("run_log", "") or "current",
            help="Each log is one benchmark run. The caption below names the "
                 "mode, model and prompt arm it was produced with.")
        st.caption(_log_summary(chosen))

    records = load_log(chosen)
    if not records:
        st.title("RegWatch AI")
        st.warning("`{}` is empty.".format(chosen))
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
        .format(highlight(context, probe.get("superseded_values"),
                          probe.get("expected_answer"))),
        unsafe_allow_html=True)

    stale = has_stale(record)
    scores = record["scores"] or {}
    consistency_block = scores.get("consistency") or {}
    consistency = consistency_block.get("value")
    # An abstention is its own outcome: the fact was lost, not contradicted.
    if consistency_block.get("abstained"):
        verdict = ("COULD NOT ANSWER — fact missing", "#f5a524")
    elif stale:
        verdict = ("STALE FACT REACHED THE MODEL", "#e5484d")
    elif stale is False:
        verdict = ("context current", "#30a46c")
    elif not (record["probe"] or {}).get("superseded_values"):
        # A control probe: the fact never changed, so there is nothing stale to
        # find. Saying "unknown" here reads as a bug rather than a clean result.
        verdict = ("control — nothing to supersede", "#8b8d98")
    else:
        verdict = ("unknown", "#8b8d98")
    st.markdown(
        '<div class="card"><h4>Answer</h4><div class="answer">{}</div>'
        '<div class="verdict" style="color:{}">{}</div>'
        '<div class="meta">{} tok in · {} · {} ms · consistency {}</div></div>'
        .format(record["answer"], verdict[1], verdict[0],
                scores.get("tokens_in") or 0,
                _usd(scores.get("cost_usd")),
                _num(scores.get("latency_ms")),
                "—" if consistency is None else consistency),
        unsafe_allow_html=True)


def _scoreboard(records) -> None:
    rows = []
    for row in scoreboard(records):
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
    # Every cell is a string: a column mixing floats with the "—" placeholder
    # fails Arrow serialisation.
    st.dataframe(rows, width="stretch", hide_index=True)
    st.caption(
        "Stale context is deterministic: did the superseded phrase appear in the "
        "injected text. Consistency is Agent A's DAG metric over the model's own "
        "answer; an answer that declines to answer scores 0 and is counted as an "
        "abstention, not a wrong answer. Cost shows — when the model's token rate "
        "is not configured (REGWATCH_PRICE_IN_PER_M / _OUT_PER_M), rather than "
        "reporting a figure we cannot stand behind."
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
            "probe_id": probe["probe_id"], "fact_ids": probe.get("fact_ids") or [],
            "expected_answer": probe.get("expected_answer"),
            "superseded_values": probe.get("superseded_values") or [],
            "phase": probe.get("phase"), "tier": probe.get("tier"),
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


def _usd(value):
    """Blank rather than $0.00 when the model's price is unknown."""
    return "—" if value is None else "${:.4f}".format(value)


if __name__ == "__main__":
    main()
