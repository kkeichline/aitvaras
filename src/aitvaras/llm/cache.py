"""On-disk response cache. What makes reruns reproducible at all.

LLMs are not seedable, so "same seed, same transcript" is only achievable by not
calling the model twice for the same input. This cache is what turns the
determinism claim in the phase 1 done-criteria from a wish into a property.

It is a *correctness* mechanism first and a cost mechanism second. Two things
follow from that:

* **The key must cover everything that could change the output.** Model, system
  prompt, every message, max_tokens, temperature, and any extra parameter. A key
  that missed one would serve a stale response for a different request, and the
  run would look deterministic while being wrong -- the worst available failure.
* **A cache hit is recorded in the transcript** (``Usage.cached``). A reader can
  then tell which steps cost money and which were replayed, instead of having to
  trust that the cache was warm.

Entries are plain JSON files, one per call, named by hash. Deliberately not a
database: a corrupted entry can be deleted by hand, and a diff shows what
changed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from aitvaras.llm.client import LLMClient, LLMResponse, Message


def cache_key(
    messages: list[Message],
    *,
    model: str,
    system: str | None,
    max_tokens: int,
    temperature: float | None,
    extra: dict[str, Any],
) -> str:
    """Hash of every input that could change the response.

    ``sort_keys=True`` is load-bearing: Python dict ordering would otherwise
    make the same logical request hash differently across runs, and the cache
    would silently never hit.
    """
    payload = json.dumps(
        {
            "model": model,
            "system": system,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "extra": {k: str(v) for k, v in sorted(extra.items())},
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class CachingLLMClient:
    """Wraps any ``LLMClient`` with a content-addressed on-disk cache."""

    def __init__(
        self,
        inner: LLMClient,
        root: Path | str = ".llm_cache",
        *,
        read_only: bool = False,
    ) -> None:
        self._inner = inner
        self._root = Path(root)
        self._read_only = read_only
        self.hits = 0
        self.misses = 0

    def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        key = cache_key(
            messages,
            model=model,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            extra=kwargs,
        )
        path = self._root / f"{key}.json"

        if path.is_file():
            try:
                cached = LLMResponse.model_validate_json(path.read_text())
            except Exception:
                # A corrupt entry must not be fatal. Fall through and re-fetch;
                # a cache is an optimisation, never a source of truth.
                path.unlink(missing_ok=True)
            else:
                self.hits += 1
                # Mark the replay so the transcript records that this step cost
                # nothing, rather than implying it was paid for twice.
                return cached.model_copy(
                    update={"usage": cached.usage.model_copy(update={"cached": True})}
                )

        if self._read_only:
            raise CacheMiss(
                f"no cached response for {model} (key {key[:12]}) and the client "
                "is read-only; run with caching enabled to populate it"
            )

        self.misses += 1
        response = self._inner.complete(
            messages,
            model=model,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        self._root.mkdir(parents=True, exist_ok=True)
        path.write_text(response.model_dump_json(indent=2))
        return response


class CacheMiss(RuntimeError):
    """Raised by a read-only cache when an entry is absent.

    Useful in CI and in reproduction runs: it turns "quietly spent money" into a
    loud failure, which is the behaviour you want when a run is supposed to be a
    pure replay.
    """
