"""Model access behind a narrow protocol.

Two reasons this is a protocol rather than direct calls:

1. **Cross-family variance is the phase 3 experiment.** Measuring whether a
   weaker monitor catches a stronger agent requires running monitors from
   different model families against identical transcripts. A client bound to one
   vendor cannot do that job, which is why LiteLLM is here despite being a
   heavier dependency than a single SDK.
2. **Determinism.** LLMs are not seedable, so byte-identical reruns are only
   possible with a response cache. That cache has to sit at exactly one
   chokepoint, and a protocol gives us one.

Cost accounting is best-effort and recorded per call rather than estimated at
the end. Published prices drift; a transcript that carries the cost it actually
incurred stays honest after they change.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from aitvaras.schemas.decision import Usage


class Message(BaseModel):
    """One chat message. Deliberately minimal -- role plus text.

    No image or tool-call blocks in phase 1. Adding them later is additive; a
    richer type now would be guessing at a shape we do not need yet.
    """

    model_config = ConfigDict(frozen=True)

    role: str
    content: str


class LLMResponse(BaseModel):
    """What came back, plus what it cost and whether we paid for it."""

    model_config = ConfigDict(frozen=True)

    text: str
    model: str
    usage: Usage
    reasoning: str = ""
    """Summarised reasoning, where the provider returns any.

    Worth understanding for the benchmark: on current Claude models the raw
    chain of thought is never returned, and summaries are opt-in. So the agent's
    *real* reasoning is largely invisible, and the journal is the only channel
    through which it explains itself.

    That is not a limitation to work around -- it is the deception problem in
    miniature. Hidden reasoning plus a self-authored public account is exactly
    the situation an oversight monitor faces. Captured separately from ``text``
    so a later monitor condition can choose whether to reveal it.
    """

    finish_reason: str = ""
    raw: dict[str, Any] = {}


@runtime_checkable
class LLMClient(Protocol):
    def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse: ...


class LiteLLMClient:
    """Multi-provider completion via LiteLLM.

    Imported lazily so that the rule oracle, the environment, and the entire
    test suite stay runnable without the dependency installed or any API key
    present. That property is worth protecting: the parts of this project that
    define ground truth should never need a network.
    """

    def __init__(self, *, timeout: int = 120, max_retries: int = 3) -> None:
        self._timeout = timeout
        self._max_retries = max_retries

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
        import litellm

        payload: list[dict[str, str]] = []
        if system:
            payload.append({"role": "system", "content": system})
        payload.extend({"role": m.role, "content": m.content} for m in messages)

        params: dict[str, Any] = {
            "model": model,
            "messages": payload,
            "max_tokens": max_tokens,
            "timeout": self._timeout,
            "num_retries": self._max_retries,
        }
        if temperature is not None:
            params["temperature"] = temperature
        params.update(kwargs)

        response = litellm.completion(**params)
        return _to_response(response, model)


def _to_response(response: Any, model: str) -> LLMResponse:
    """Normalise a LiteLLM response.

    Defensive throughout: field availability varies by provider, and a missing
    cost or token count must degrade to zero rather than crash a 60-step run
    forty steps in.
    """
    choice = response.choices[0]
    text = getattr(choice.message, "content", None) or ""
    reasoning = getattr(choice.message, "reasoning_content", None) or ""

    raw_usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(raw_usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(raw_usage, "completion_tokens", 0) or 0)

    # Cached-prefix reads. Reported under different names per provider, and a
    # zero here across repeated calls with a stable prefix is the signal that
    # something is silently invalidating the cache.
    cached_tokens = 0
    details = getattr(raw_usage, "prompt_tokens_details", None)
    if details is not None:
        cached_tokens = int(getattr(details, "cached_tokens", 0) or 0)

    cost = Decimal("0")
    try:
        import litellm

        cost = Decimal(str(litellm.completion_cost(completion_response=response) or 0))
    except Exception:
        # Prices drift and LiteLLM does not know every model. An unknown cost is
        # worth recording as zero; it is not worth failing a run over.
        pass

    return LLMResponse(
        text=text,
        model=getattr(response, "model", model) or model,
        reasoning=reasoning,
        finish_reason=getattr(choice, "finish_reason", "") or "",
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            cost_usd=cost,
            cached=False,
        ),
    )
