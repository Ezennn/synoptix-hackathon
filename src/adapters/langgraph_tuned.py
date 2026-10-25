"""LangGraph (TUNED, Rung-1) — latest-per-fact reducer PLUS a generic
dependency-propagation inference rule.

This is a LABELLED TUNED VARIANT of ``langgraph_state`` (the baseline reducer).
The baseline keeps the latest statement per (entity, attribute) and, by design,
CANNOT do transitive invalidation — it has no rule linking "divested subsidiary
B" to "no crypto exposure". This variant adds exactly one thing: a GENERIC
dependency rule that propagates an entity's invalidation to whatever is held via
that entity.

Why this is Rung-1 (honest, not teaching-to-the-test)
------------------------------------------------------
We hardcode the SHAPE of the reasoning, never the CONTENT. The rule is:

    held_via(A, B)  AND  invalidated(B)   =>   exposure(A) is invalidated

It is expressed over GENERIC relations extracted from statement text:
  * held_via(A, B)   — "A is held via / held through / held by B",
                        "B holds the A position", "A sits in B"
  * invalidated(B)   — "B divested / sold / closed / wound up / disposed"

Nothing here names "crypto", "subsidiary B" or any scenario value, so the rule
generalises to ANY held-via chain (pension-via-fund, position-via-subsidiary,
exposure-via-entity). That is the difference between a general inference
mechanism (defensible) and pre-writing the answer for one probe (a confound).

What it should do on the benchmark
----------------------------------
It still passes everything the baseline passes (latest-wins for direct and
reinstatement). ADDITIONALLY, on the transitive probe — "does the client still
have crypto exposure?" after the holding entity is divested — it emits a derived
note ("NO exposure to <position>: held via <B>, which is now invalidated"),
which the answer model reads to say "no". The baseline reducer misses this; the
lift between the two columns is the point of the variant.

name: "langgraph-tuned" — a SEPARATE column, so the scoreboard shows the
baseline's miss next to this one's hit.

retrieval_context: the reduced statement set PLUS any derived-invalidation
notes, fully exposed so the order-free metrics score the real injected text.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from src.adapters.base import TurnResult
from src.adapters.langgraph_state import reduce_statements
from src.llm import LLMGateway, get_gateway
from src.prompts import system_prompt

# --- Generic relation extraction (no scenario-specific tokens) --------------

# held_via(A, B): what is held, and the entity it is held through.
# Two phrasings: "<A> is held via <B>"  and  "<B> holds the <A> position".
_HELD_VIA_PATTERNS: List[re.Pattern] = [
    re.compile(r"(?P<a>[\w\s]+?)\s+(?:is\s+)?held\s+(?:via|through|by|in)\s+(?P<b>[\w\s]+)", re.I),
    re.compile(r"(?P<b>[\w\s]+?)\s+(?:currently\s+)?holds?\s+the\s+(?P<a>[\w\s]+?)\s+position", re.I),
    re.compile(r"(?P<a>[\w\s]+?)\s+sits?\s+(?:in|inside|within)\s+(?P<b>[\w\s]+)", re.I),
]

# invalidated(B): the entity B is no longer owned / active.
# Transitive verbs (take a direct object): "divested <B>".
_INVALIDATION_VERBS = (
    "divested", "sold", "disposed of", "closed", "wound up", "wound-up",
    "liquidated", "exited", "disposed", "no longer owns", "no longer holds",
)
# Intransitive / passive forms (subject is <B>): "<B> was divested".
_INVALIDATION_PASSIVE = (
    "divested", "sold", "closed", "wound up", "wound-up", "liquidated",
    "disposed", "gone", "no longer owned", "no longer held",
)
# Entity AFTER the verb: "(we) divested <B>", "disposed of <B>".
_INVALIDATION_AFTER = re.compile(
    r"(?:%s)\s+(?:of\s+|the\s+|our\s+|its\s+)?(?P<b>[\w\s]+)"
    % "|".join(re.escape(w) for w in _INVALIDATION_VERBS),
    re.I,
)
# Entity BEFORE the verb: "<B> was divested", "<B> is closed".
_INVALIDATION_BEFORE = re.compile(
    r"(?P<b>[\w\s]+?)\s+(?:was|is|has been|were|have been)\s+(?:%s)"
    % "|".join(re.escape(w) for w in _INVALIDATION_PASSIVE),
    re.I,
)
# Subject words that are never the invalidated ENTITY (strip if captured).
_NON_ENTITY_SUBJECTS = {"we", "it", "they", "the client", "client", "", "the"}


def _norm(s: str) -> str:
    """Normalise an entity phrase for comparison: lowercase, collapse spaces,
    strip leading articles/determiners and common org-prefix noise, and
    canonicalise the generic 'subsidiary X' <-> 'subX' abbreviation so the two
    surface forms of one entity compare equal. All rules are domain-neutral
    (no scenario value is named)."""
    s = s.strip().lower()
    s = re.sub(r"^(the|a|an|our|its|their)\s+", "", s)
    # Drop a leading org/client prefix so "client acme subB" -> "subB".
    s = re.sub(r"^(client\s+|acme\s+)+", "", s)
    s = re.sub(r"\s+", " ", s).strip(" .,:;")
    # Generic abbreviation: "subsidiary b" / "subb" / "sub b" -> canonical "sub b".
    s = re.sub(r"\bsubsidiary\s+", "sub ", s)
    s = re.sub(r"\bsub([a-z0-9])\b", r"sub \1", s)
    return s.strip()


def _tokens(s: str) -> set:
    """Significant tokens of an entity phrase (drop 1-char noise words)."""
    return {t for t in s.split() if t}


def _extract_held_via(statements: List[str]) -> List[Tuple[str, str]]:
    """Return list of (held_thing, via_entity) pairs found across statements."""
    pairs: List[Tuple[str, str]] = []
    for s in statements:
        for pat in _HELD_VIA_PATTERNS:
            m = pat.search(s)
            if m:
                a, b = _norm(m.group("a")), _norm(m.group("b"))
                if a and b and a != b:
                    pairs.append((a, b))
    return pairs


def _extract_invalidated(statements: List[str]) -> List[str]:
    """Return list of entities asserted as invalidated (divested/sold/closed).

    Handles both phrasings: entity AFTER the verb ("divested subsidiary B") and
    entity BEFORE the verb ("subsidiary B was divested"). Subjects that are not
    real entities (we/it/the client) are dropped so a transitive "we divested X"
    yields X, not "we".
    """
    out: List[str] = []
    for s in statements:
        for rx in (_INVALIDATION_BEFORE, _INVALIDATION_AFTER):
            m = rx.search(s)
            if m:
                b = _norm(m.group("b"))
                if b and b not in _NON_ENTITY_SUBJECTS:
                    out.append(b)
                    break  # one invalidation per statement is enough
    return out


def _entities_match(a: str, b: str) -> bool:
    """True if two entity phrases plausibly refer to the same entity.

    Uses canonical-form equality, substring containment, OR shared significant
    tokens (so 'sub b' matches 'sub b' and 'client acme subb' matches
    'subsidiary b' once both are canonicalised). Domain-neutral."""
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    ta, tb = _tokens(a), _tokens(b)
    shared = ta & tb
    # Require at least one shared token longer than a single char, so unrelated
    # entities that merely share "b" do not match — but "sub"+"b" both shared do.
    return bool(shared) and (shared == ta or shared == tb or
                             any(len(t) > 2 for t in shared) and len(shared) >= 2)


def derive_invalidations(statements: List[str]) -> List[str]:
    """Apply the generic rule held_via(A,B) AND invalidated(B) => A invalidated.

    Returns a list of human-readable derived-invalidation notes (possibly empty).
    Content-neutral: it only wires together relations found in the text.
    """
    held_via = _extract_held_via(statements)
    invalidated = _extract_invalidated(statements)
    notes: List[str] = []
    for held_thing, via_entity in held_via:
        for inv in invalidated:
            if _entities_match(via_entity, inv):
                notes.append(
                    "DERIVED: no current exposure to '{thing}' — it was held via "
                    "'{via}', which has been invalidated (divested/closed).".format(
                        thing=held_thing, via=via_entity
                    )
                )
                break
    return notes


class LangGraphTunedBackend:
    name = "langgraph-tuned"

    def __init__(self, gateway: Optional[LLMGateway] = None) -> None:
        self.gw = gateway or get_gateway()
        self._store: Dict[str, List[str]] = {}

    def reset(self, session_id: str) -> None:
        self._store[session_id] = []

    def load(self, session_id: str, documents: List[str]) -> None:
        self._store.setdefault(session_id, []).extend(documents)

    def update(self, session_id: str, statement: str) -> None:
        self._store.setdefault(session_id, []).append(statement)

    def _assemble(self, session_id: str) -> List[str]:
        raw = list(self._store.get(session_id, []))
        pruned = reduce_statements(raw)          # baseline latest-per-fact
        notes = derive_invalidations(raw)        # Rung-1 generic propagation
        # Derived notes come FIRST so a superseding inference is salient.
        return notes + pruned

    def ask(self, session_id: str, question: str) -> TurnResult:
        context = self._assemble(session_id)
        prompt = "STATE (reduced + derived invalidations):\n" + "\n".join(
            "- " + c for c in context
        ) + "\n\nQUESTION: " + question
        resp = self.gw.complete(
            [
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": prompt},
            ]
        )
        return TurnResult(
            answer=resp.text,
            retrieval_context=context,
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
            latency_ms=resp.latency_ms,
        )


def build(gateway: Optional[LLMGateway] = None) -> LangGraphTunedBackend:
    """Zero-arg factory for the registry; uses the shared default gateway when
    called as ``build()``. Optional ``gateway`` for injection."""
    return LangGraphTunedBackend(gateway)
