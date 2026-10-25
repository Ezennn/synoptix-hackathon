"""Backend plugin registry. Agent B.

Backends are declared here by name and imported lazily, so a machine without
zep-python or letta installed can still run the service — those columns simply
report `unavailable` instead of crashing the process. Agent A's engine must
stay importable without B's heavy dependencies; that is what `requires` and the
deferred import below protect.

Adding a backend is one entry in SPECS.
"""
from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .adapters.base import MemoryBackend


@dataclass
class BackendSpec:
    name: str
    module: str                       # dotted path, imported only when used
    factory: str                      # zero-arg callable in that module
    requires: Tuple[str, ...] = ()    # importable module names it needs
    owner: str = "B"
    note: str = ""
    _instance: Optional[MemoryBackend] = field(default=None, repr=False)

    def missing_requirements(self) -> List[str]:
        return [m for m in self.requires if importlib.util.find_spec(m) is None]

    def available(self) -> bool:
        if self.missing_requirements():
            return False
        return importlib.util.find_spec(self.module) is not None

    def get(self) -> MemoryBackend:
        """Import and instantiate on first use. Raises if unavailable."""
        if self._instance is None:
            module = importlib.import_module(self.module)
            self._instance = getattr(module, self.factory)()
        return self._instance


SPECS: List[BackendSpec] = [
    BackendSpec("stub_current", "src.adapters.stub", "build_current",
                note="fixture: always surfaces the live fact"),
    BackendSpec("stub_stale", "src.adapters.stub", "build_stale",
                note="fixture: always surfaces the superseded fact"),
    # Agent A's three in-process backends. Declared now so the UI reserves their
    # columns; they report unavailable until A pushes them.
    BackendSpec("sanity", "src.adapters.sanity", "build", owner="A",
                note="full history in context (ceiling)"),
    BackendSpec("litellm_trim", "src.adapters.litellm_trim", "build", owner="A",
                note="hard token-budget trimming"),
    BackendSpec("langgraph_state", "src.adapters.langgraph_state", "build", owner="A",
                note="checkpointer + summarising reducer"),
    BackendSpec("langgraph_tuned", "src.adapters.langgraph_tuned", "build", owner="A",
                note="TUNED (Rung-1): reducer + generic held_via dependency-propagation inference"),
    # B's hosted integrations. Gated on the hour-0 spikes.
    BackendSpec("zep_graph", "src.adapters.zep_graph", "build",
                requires=("zep_cloud",),
                note="temporal knowledge graph; showcase for contradiction resolution"),
    BackendSpec("letta_paging", "src.adapters.letta_paging", "build",
                requires=("letta_client",),
                note="self-editing core memory; needs Python 3.11+"),
]

_BY_NAME: Dict[str, BackendSpec] = {s.name: s for s in SPECS}


def get_spec(name: str) -> BackendSpec:
    if name not in _BY_NAME:
        raise KeyError("unknown backend: {}".format(name))
    return _BY_NAME[name]


def available_specs() -> List[BackendSpec]:
    return [s for s in SPECS if s.available()]


def status() -> List[dict]:
    """What every declared backend is doing — drives the UI's column states."""
    out = []
    for s in SPECS:
        missing = s.missing_requirements()
        out.append({
            "name": s.name,
            "owner": s.owner,
            "available": s.available(),
            "missing_requirements": missing,
            "reason": _reason(s, missing),
            "note": s.note,
        })
    return out


def _reason(spec: BackendSpec, missing: List[str]) -> str:
    if missing:
        return "not installed: {}".format(", ".join(missing))
    if importlib.util.find_spec(spec.module) is None:
        return "adapter not written yet (owner {})".format(spec.owner)
    return "ready"
