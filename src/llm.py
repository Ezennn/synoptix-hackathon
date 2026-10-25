"""Cached LLM gateway — the single path every adapter calls the model through.

Why this exists (PLAN.md, non-negotiables):
- Token counts and latency are measured HERE, once, so cost (axis 5) and
  retrieval latency (axis 4) are comparable across every backend. An adapter
  never estimates them.
- The model and generation params are held constant across adapters, so the
  only thing that differs between runs is which memory layer assembled the
  prompt.
- Responses are cached on disk keyed by (model, messages, params). A rerun of
  the whole benchmark is then free and deterministic — important for a demo and
  for CI on a memory-tight host.

Two modes, chosen by ``REGWATCH_LLM_MODE`` (or the ``mode`` arg):

- ``"stub"`` (DEFAULT): no network, no API key, no heavy deps. Returns
  deterministic canned text derived from the prompt. This is the day-one
  unblock for Agent B and the mode tests run in. The stub still reports
  plausible token counts and real wall-clock latency, so the full pipeline
  (generate -> adapters -> evaluator -> scoreboard) runs end to end offline.
- ``"live"``: routes through LiteLLM to a real model. LiteLLM is imported
  lazily so the engine stays importable on a box that never installed it.

Python target: 3.9 (``from __future__ import annotations`` + typing generics).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# One cache dir for the whole project; override with REGWATCH_CACHE_DIR.
_DEFAULT_CACHE_DIR = Path(
    os.environ.get("REGWATCH_CACHE_DIR", Path(__file__).resolve().parent.parent / "data" / "cache")
)
_DEFAULT_MODEL = os.environ.get("REGWATCH_MODEL", "gpt-4o-mini")


@dataclass
class LLMResponse:
    """What the gateway returns for one completion.

    Adapters forward these numbers straight into ``TurnResult`` — they do not
    recompute them.
    """

    text: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cached: bool = False
    model: str = _DEFAULT_MODEL


def _approx_tokens(text: str) -> int:
    """Cheap, deterministic token estimate (~4 chars/token).

    Used only by the stub and as a fallback when a live provider omits usage.
    Good enough for relative cost comparison; the live path prefers real usage.
    """
    if not text:
        return 0
    return max(1, round(len(text) / 4))


class LLMGateway:
    """Cached, instrumented single entry point to the model.

    Usage (every adapter):

        gw = LLMGateway()                      # stub by default
        resp = gw.complete(messages=[{"role": "user", "content": prompt}])
        # -> resp.text, resp.tokens_in, resp.tokens_out, resp.latency_ms
    """

    def __init__(
        self,
        model: Optional[str] = None,
        mode: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        temperature: float = 0.0,
        max_retries: int = 3,
    ) -> None:
        self.model = model or _DEFAULT_MODEL
        self.mode = (mode or os.environ.get("REGWATCH_LLM_MODE", "stub")).lower()
        self.temperature = temperature
        self.max_retries = max_retries
        self.cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Cleared on the first provider refusal; see _live_complete.
        self._send_temperature = True
        # Cache status of the most recent call, plus running tallies. The
        # frozen contract has no place for this, and a cached turn returns in
        # ~0 ms, so without it a warm-cache run silently mixes real timings
        # with cache reads and reports the average as if it were measured.
        # The runner reads `last_cached` straight after each ask().
        self.last_cached = False
        self.calls = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def usage(self) -> Dict[str, int]:
        """Call tallies since this gateway was created."""
        return {"calls": self.calls, "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses}

    # ---- cache ---------------------------------------------------------
    def _cache_key(self, messages: List[Dict[str, str]], max_tokens: int) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": max_tokens,
                "mode": self.mode,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / (key + ".json")

    def _read_cache(self, key: str) -> Optional[Dict]:
        p = self._cache_path(key)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                return None
        return None

    def _write_cache(self, key: str, data: Dict) -> None:
        try:
            self._cache_path(key).write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8"
            )
        except OSError:
            pass  # cache is an optimization, never fatal

    # ---- public API ----------------------------------------------------
    def complete(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
    ) -> LLMResponse:
        """Return a completion for ``messages``, caching by content.

        A cache hit still reports real (near-zero) latency and ``cached=True``
        so the evaluator can tell a cached turn from a live one.
        """
        key = self._cache_key(messages, max_tokens)
        hit = self._read_cache(key)
        start = time.perf_counter()
        if hit is not None:
            latency_ms = (time.perf_counter() - start) * 1000.0
            self.last_cached = True
            self.cache_hits += 1
            self.calls += 1
            return LLMResponse(
                text=hit["text"],
                tokens_in=hit["tokens_in"],
                tokens_out=hit["tokens_out"],
                latency_ms=latency_ms,
                cached=True,
                model=self.model,
            )

        if self.mode == "stub":
            text, t_in, t_out = self._stub_complete(messages, max_tokens)
        elif self.mode == "live":
            text, t_in, t_out = self._live_complete(messages, max_tokens)
        else:
            raise ValueError(
                "REGWATCH_LLM_MODE must be 'stub' or 'live', got %r" % self.mode
            )

        latency_ms = (time.perf_counter() - start) * 1000.0
        self._write_cache(key, {"text": text, "tokens_in": t_in, "tokens_out": t_out})
        self.last_cached = False
        self.cache_misses += 1
        self.calls += 1
        return LLMResponse(
            text=text,
            tokens_in=t_in,
            tokens_out=t_out,
            latency_ms=latency_ms,
            cached=False,
            model=self.model,
        )

    # ---- stub ----------------------------------------------------------
    def _stub_complete(self, messages, max_tokens):
        """Deterministic, offline answer.

        The stub is intentionally literal: it echoes the most recent
        non-system line it can find that looks like a stated fact, so a
        recency-biased answer is the DEFAULT. That makes the stub a useful
        fixture — a backend that does nothing clever will trivially "pass"
        recency and trivially "fail" transitive invalidation, which is exactly
        the behaviour the metrics must distinguish. Real differentiation comes
        in ``live`` mode; the stub just keeps the pipeline runnable offline.
        """
        prompt = "\n".join(m.get("content", "") for m in messages)
        t_in = _approx_tokens(prompt)
        # Echo last substantive line as the "answer".
        lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
        answer = lines[-1] if lines else ""
        answer = "[stub] " + answer[: max_tokens * 4]
        return answer, t_in, _approx_tokens(answer)

    # ---- live ----------------------------------------------------------
    def _live_complete(self, messages, max_tokens):
        """Route through LiteLLM with retries. LiteLLM imported lazily."""
        try:
            import litellm  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "live mode needs litellm: pip install litellm, or use stub mode"
            ) from exc

        last_exc = None
        for attempt in range(self.max_retries):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                }
                # Newer models reject `temperature` outright ("`temperature` is
                # deprecated for this model"). Send it only while the model
                # accepts it; once refused, drop it for the rest of the run so
                # every later call does not burn its retries on the same error.
                if self._send_temperature:
                    kwargs["temperature"] = self.temperature
                resp = litellm.completion(**kwargs)
                text = resp["choices"][0]["message"]["content"] or ""
                usage = resp.get("usage") or {}
                t_in = int(usage.get("prompt_tokens") or _approx_tokens(
                    "\n".join(m.get("content", "") for m in messages)
                ))
                t_out = int(usage.get("completion_tokens") or _approx_tokens(text))
                return text, t_in, t_out
            except Exception as exc:  # noqa: BLE001 — retry any provider error
                last_exc = exc
                if self._send_temperature and "temperature" in str(exc).lower():
                    # Retry immediately without it rather than sleeping.
                    self._send_temperature = False
                    continue
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError("LLM call failed after retries: %s" % last_exc)


# Module-level convenience singleton for adapters that just want the default.
_default_gateway: Optional[LLMGateway] = None


def get_gateway() -> LLMGateway:
    """Shared default gateway (stub mode unless REGWATCH_LLM_MODE=live)."""
    global _default_gateway
    if _default_gateway is None:
        _default_gateway = LLMGateway()
    return _default_gateway
