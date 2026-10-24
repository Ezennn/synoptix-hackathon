"""Deterministic data + answer-key generator (Agent A, PLAN step 2).

Reads ``data/facts.yaml`` (the temporal answer key) and emits:

  data/sessions/session_{n}.json   one transcript per simulated session
  data/probes.json                 100 probe questions, each tagged with the
                                   fact(s) it tests and the EXPECTED CURRENT
                                   answer at ask-time.

Everything is deterministic (fixed seed) so a rerun reproduces byte-identical
output — required for a replayable demo and for comparing backends fairly.

------------------------------------------------------------------------------
PROBE JSON SCHEMA  (published here — Agent B's evaluator reads this contract)
------------------------------------------------------------------------------
data/probes.json is: {"probes": [Probe, ...]}  where each Probe is:

  {
    "probe_id":          "p0001",
    "phase":             1|2|3|4,            # which stress phase this exercises
    "tier":              "baseline|direct|partial|transitive|reinstatement",
    "question":          "What is Acme's current crypto risk appetite?",
    "fact_ids":          ["crypto_appetite_v2"],   # fact(s) under test
    "ask_after_session": 2,                   # probe is asked after this session
    "ask_date":          "2026-03-20",        # the 'now' the answer is judged at
    "expected_answer":   "conservative",      # CURRENT value at ask_date
    "superseded_values": ["aggressive"],      # values that MUST NOT be asserted
    "multi_hop":         false,               # true => needs >1 timeline/fact
    "notes":             "direct overwrite; recency check"
  }

Axis-1 (DAG state consistency) uses: fact_ids, expected_answer,
superseded_values, ask_date. Axes 2/3 use the probe's question against the
backend's retrieval_context. B must not recompute expected answers — they come
from here.
------------------------------------------------------------------------------

Python target: 3.9.
"""
from __future__ import annotations

import json
import random
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FACTS_PATH = DATA / "facts.yaml"
SESSIONS_DIR = DATA / "sessions"
PROBES_PATH = DATA / "probes.json"

SEED = 42
N_PROBES = 100

SESSION_DATES = {1: "2026-01-10", 2: "2026-03-15", 3: "2026-04-20", 4: "2026-06-01"}


def _parse_date(s) -> Optional[date]:
    """Accept None, an ISO string, or a datetime.date (PyYAML auto-parses
    unquoted ISO dates into date objects)."""
    if s is None:
        return None
    if isinstance(s, date):
        return s
    y, m, d = (int(x) for x in str(s).split("-"))
    return date(y, m, d)


def load_facts() -> List[Dict]:
    with FACTS_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)["facts"]


def current_value_at(facts: List[Dict], entity: str, attribute: str, at: date):
    """Return (value, fact_id, superseded_values) current at ``at``.

    This is the ground truth the metric trusts. ``superseded_values`` are the
    values that WERE true for this (entity, attribute) but are not at ``at``.
    """
    current_val = None
    current_id = None
    superseded = []
    for f in facts:
        if f["entity"] != entity or f["attribute"] != attribute:
            continue
        vf = _parse_date(f["valid_from"])
        vu = _parse_date(f["valid_until"])
        is_current = vf is not None and vf <= at and (vu is None or at < vu)
        if is_current:
            current_val = str(f["value"])
            current_id = f["id"]
        else:
            # true at some other time => a superseded/future value to reject now
            superseded.append(str(f["value"]))
    # don't list the current value as superseded even if it recurs
    superseded = sorted({v for v in superseded if v != current_val})
    return current_val, current_id, superseded


# ---- statement phrasing (human-ish transcript lines) ----------------------
def _statement_for(fact: Dict) -> str:
    e, a, v = fact["entity"], fact["attribute"], fact["value"]
    phrasings = {
        "legal_name": "For the record, the client's legal name is {v}.",
        "base_currency": "All reporting for this client is in {v}.",
        "risk_appetite_crypto": "The client's risk appetite for crypto assets is now {v}.",
        "risk_tier": "The {e} is classified risk tier: {v}.",
        "holds_position": "{e} currently holds the {v} position.",
        "ownership_status": "Note: {e} has been {v} as of this session.",
        "has_crypto_exposure": "(derived) crypto exposure at group level: {v}.",
        "tax_domicile": "The client's tax domicile is now {v}.",
        "version": "Regulatory update — suitability policy is now {v}.",
    }
    tmpl = phrasings.get(a, "{e} {a} = {v}.")
    return tmpl.format(e=e.replace("_", " "), a=a, v=v)


# ---- Phase-4 plausible-but-stale distractors ------------------------------
def _distractor_for(fact: Dict, rng: random.Random) -> str:
    """A long, topically-relevant note that discusses the OLD value at length.

    This is the hard noise from the README: not random JSON, but a detailed
    meeting note about a superseded fact, so precision/faithfulness are stressed
    by a distractor that is temporally wrong but semantically on-topic.
    """
    e = fact["entity"].replace("_", " ")
    v = fact["value"]
    filler = rng.choice(
        [
            "The committee spent considerable time on this in the prior quarter.",
            "Historical minutes record extensive debate before this was set.",
            "This was reaffirmed in several earlier calls with the relationship team.",
        ]
    )
    return (
        "MEETING NOTE (archival): In an earlier session the {e} position was "
        "discussed at length. At that time the recorded value was '{v}', and "
        "the advisory team built the model portfolio around it. {f} Attendees "
        "reviewed exposure, liquidity and reporting implications of '{v}' in "
        "detail across roughly forty minutes of discussion."
    ).format(e=e, v=v, f=filler)


def build_sessions(facts: List[Dict]) -> Dict[int, Dict]:
    sessions: Dict[int, Dict] = {
        n: {"session": n, "date": SESSION_DATES[n], "statements": [], "distractors": []}
        for n in SESSION_DATES
    }
    for f in facts:
        s = int(f["stated_in_session"])
        sessions[s]["statements"].append(
            {"fact_id": f["id"], "text": _statement_for(f), "tier": f["tier"]}
        )
    return sessions


def add_phase4_distractors(sessions: Dict[int, Dict], facts: List[Dict], rng: random.Random):
    """Seed each session after the first with stale distractors (Phase 4)."""
    superseded = [f for f in facts if f.get("valid_until") is not None]
    for n in sorted(sessions)[1:]:
        picks = rng.sample(superseded, k=min(3, len(superseded)))
        for f in picks:
            sessions[n]["distractors"].append(
                {"about_fact_id": f["id"], "text": _distractor_for(f, rng)}
            )


# ---- probe generation ------------------------------------------------------
_QUESTION_TEMPLATES = {
    "risk_appetite_crypto": "What is {name}'s current risk appetite for crypto assets?",
    "risk_tier": "What is the current risk tier of the {entity}?",
    "tax_domicile": "What is {name}'s current tax domicile?",
    "has_crypto_exposure": "Does {name} currently have any crypto exposure? Answer yes or no.",
    "ownership_status": "What is the current ownership status of {entity}?",
    "legal_name": "What is {name}'s full legal name?",
    "base_currency": "What base currency is used for {name}?",
    "version": "Which version of the suitability policy is currently in force?",
}

_MULTIHOP_TEMPLATE = (
    "Given {name}'s current crypto risk tier and the suitability policy "
    "currently in force, is holding crypto compliant? Answer yes or no and name "
    "the governing policy version."
)


def _phase_for_probe(tier: str, multi_hop: bool) -> int:
    if multi_hop:
        return 3
    if tier == "baseline":
        return 1
    return 2  # mutations are probed in/after phase 2


def build_probes(facts: List[Dict], rng: random.Random) -> List[Dict]:
    # unique (entity, attribute) pairs we can probe
    pairs = []
    seen = set()
    for f in facts:
        k = (f["entity"], f["attribute"])
        if k not in seen:
            seen.add(k)
            pairs.append((f["entity"], f["attribute"], f["tier"]))

    probes: List[Dict] = []
    pid = 1

    def emit(entity, attribute, tier, ask_after, multi_hop=False):
        nonlocal pid
        at = _parse_date(SESSION_DATES[ask_after])
        # ask a few days after the session so same-day edge cases are explicit
        from datetime import timedelta

        ask_dt = at
        val, fid, superseded = current_value_at(facts, entity, attribute, ask_dt)
        if val is None:
            return  # nothing current at this time for this pair; skip
        name = "Acme Capital Partners LLP"
        if multi_hop:
            question = _MULTIHOP_TEMPLATE.format(name=name)
            # compliance answer derived from crypto tier + policy version
            tier_val, _, _ = current_value_at(
                facts, "client_acme", "risk_appetite_crypto", ask_dt
            )
            pol_val, pol_id, _ = current_value_at(
                facts, "policy_mifid_suitability", "version", ask_dt
            )
            compliant = "no" if (tier_val or "").lower() == "conservative" else "yes"
            expected = "%s (%s)" % (compliant, pol_val)
            fact_ids = ["crypto_appetite_v2" if compliant == "no" else "crypto_appetite_v1", pol_id]
            superseded_vals = []
        else:
            tmpl = _QUESTION_TEMPLATES.get(attribute, "What is the current {attribute} of {entity}?")
            question = tmpl.format(name=name, entity=entity.replace("_", " "), attribute=attribute)
            expected = val
            fact_ids = [fid]
            superseded_vals = superseded

        probes.append(
            {
                "probe_id": "p%04d" % pid,
                "phase": 3 if multi_hop else _phase_for_probe(tier, False),
                "tier": tier,
                "question": question,
                "fact_ids": fact_ids,
                "ask_after_session": ask_after,
                "ask_date": ask_dt.isoformat(),
                "expected_answer": expected,
                "superseded_values": superseded_vals,
                "multi_hop": multi_hop,
                "notes": _note_for(tier, multi_hop),
            }
        )
        pid += 1

    # Probe every pair at every session >= its first appearance, across tiers,
    # so there is a natural spread over phases and ask-times.
    for entity, attribute, tier in pairs:
        first = min(
            int(f["stated_in_session"])
            for f in facts
            if f["entity"] == entity and f["attribute"] == attribute
        )
        for s in range(first, 5):
            emit(entity, attribute, tier, s)

    # Multi-hop compliance probes (Phase 3), asked at S3 and S4.
    for s in (3, 4):
        emit("client_acme", "risk_appetite_crypto", "transitive", s, multi_hop=True)

    # Pad deterministically to exactly N_PROBES by resampling existing pairs at
    # varied ask-times (keeps the answer key correct; just more coverage).
    base = list(probes)
    while len(probes) < N_PROBES and base:
        src = base[rng.randrange(len(base))]
        s = rng.randint(1, 4)
        # reuse emit path to recompute a correct expected answer at the new time
        ent_attr = None
        for f in facts:
            if f["id"] in src["fact_ids"]:
                ent_attr = (f["entity"], f["attribute"], f["tier"])
                break
        if ent_attr is None:
            base.remove(src)
            continue
        emit(ent_attr[0], ent_attr[1], ent_attr[2], s, multi_hop=src["multi_hop"])

    return probes[:N_PROBES]


def _note_for(tier: str, multi_hop: bool) -> str:
    if multi_hop:
        return "phase-3 multi-hop over client timeline + regulation timeline"
    return {
        "baseline": "control; value never changes",
        "direct": "direct overwrite; recency check",
        "partial": "scoped overwrite; must not invalidate sibling entity",
        "transitive": "invalidated by inference, not direct contradiction",
        "reinstatement": "A/B/A; must track history, not just final value",
    }.get(tier, "")


def main() -> None:
    rng = random.Random(SEED)
    facts = load_facts()

    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = build_sessions(facts)
    add_phase4_distractors(sessions, facts, rng)
    for n, sess in sessions.items():
        (SESSIONS_DIR / ("session_%d.json" % n)).write_text(
            json.dumps(sess, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    probes = build_probes(facts, rng)
    PROBES_PATH.write_text(
        json.dumps({"probes": probes}, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Summary to stdout
    by_phase: Dict[int, int] = {}
    for p in probes:
        by_phase[p["phase"]] = by_phase.get(p["phase"], 0) + 1
    print("generated %d sessions, %d probes" % (len(sessions), len(probes)))
    print("probes by phase:", dict(sorted(by_phase.items())))
    print("wrote:", PROBES_PATH)


if __name__ == "__main__":
    main()
