"""The response cache, which is what makes "same seed, same transcript" true.

No network here. A fake client counts its calls, so the tests assert the thing
that actually matters: how many times the model would have been paid for.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from aitvaras.llm.cache import CacheMiss, CachingLLMClient, cache_key
from aitvaras.llm.client import LLMClient, LLMResponse, Message
from aitvaras.schemas.decision import Usage

D = Decimal


class FakeClient:
    """Deterministic stand-in. Counts calls so cache behaviour is observable."""

    def __init__(self, text: str = "ok") -> None:
        self.calls = 0
        self._text = text

    def complete(
        self, messages, *, model, system=None, max_tokens=4096, temperature=None, **kwargs
    ) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            text=f"{self._text}-{self.calls}",
            model=model,
            usage=Usage(prompt_tokens=10, completion_tokens=5, cost_usd=D("0.001")),
        )


def msgs(text: str = "hello") -> list[Message]:
    return [Message(role="user", content=text)]


KEY_ARGS = dict(model="m", system=None, max_tokens=100, temperature=None, extra={})


def test_fake_satisfies_the_client_protocol():
    assert isinstance(FakeClient(), LLMClient)


# --------------------------------------------------------------------------
# The key must cover everything that could change the output
# --------------------------------------------------------------------------


def test_identical_requests_hash_identically():
    assert cache_key(msgs(), **KEY_ARGS) == cache_key(msgs(), **KEY_ARGS)


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", "other"),
        ("system", "a system prompt"),
        ("max_tokens", 200),
        ("temperature", 0.7),
    ],
)
def test_every_request_parameter_changes_the_key(field, value):
    """A key that ignored any of these would serve a stale response for a
    different request -- the run would look deterministic while being wrong,
    which is the worst failure available here."""
    changed = {**KEY_ARGS, field: value}
    assert cache_key(msgs(), **KEY_ARGS) != cache_key(msgs(), **changed)


def test_message_content_changes_the_key():
    assert cache_key(msgs("a"), **KEY_ARGS) != cache_key(msgs("b"), **KEY_ARGS)


def test_extra_parameters_change_the_key():
    changed = {**KEY_ARGS, "extra": {"top_p": 0.5}}
    assert cache_key(msgs(), **KEY_ARGS) != cache_key(msgs(), **changed)


def test_key_is_stable_regardless_of_extra_dict_ordering():
    """Python dict ordering must not leak into the hash, or the cache would
    silently never hit across runs."""
    a = {**KEY_ARGS, "extra": {"a": 1, "b": 2}}
    b = {**KEY_ARGS, "extra": {"b": 2, "a": 1}}
    assert cache_key(msgs(), **a) == cache_key(msgs(), **b)


# --------------------------------------------------------------------------
# Cache behaviour
# --------------------------------------------------------------------------


def test_a_repeated_request_does_not_reach_the_model(tmp_path):
    inner = FakeClient()
    client = CachingLLMClient(inner, root=tmp_path)

    first = client.complete(msgs(), model="m")
    second = client.complete(msgs(), model="m")

    assert inner.calls == 1
    assert first.text == second.text
    assert (client.hits, client.misses) == (1, 1)


def test_a_replayed_response_is_marked_as_cached(tmp_path):
    """So a reader can tell which steps cost money, instead of trusting that
    the cache was warm."""
    client = CachingLLMClient(FakeClient(), root=tmp_path)
    fresh = client.complete(msgs(), model="m")
    replayed = client.complete(msgs(), model="m")

    assert fresh.usage.cached is False
    assert replayed.usage.cached is True


def test_a_different_request_is_not_served_from_cache(tmp_path):
    inner = FakeClient()
    client = CachingLLMClient(inner, root=tmp_path)
    client.complete(msgs("a"), model="m")
    client.complete(msgs("b"), model="m")
    assert inner.calls == 2


def test_the_cache_survives_a_new_client_over_the_same_directory(tmp_path):
    """Reruns happen in new processes. A cache that only worked in-memory would
    not deliver the determinism it exists for."""
    first = FakeClient()
    CachingLLMClient(first, root=tmp_path).complete(msgs(), model="m")

    second = FakeClient()
    CachingLLMClient(second, root=tmp_path).complete(msgs(), model="m")

    assert second.calls == 0


def test_a_corrupt_entry_is_discarded_rather_than_fatal(tmp_path):
    """A cache is an optimisation, never a source of truth."""
    inner = FakeClient()
    client = CachingLLMClient(inner, root=tmp_path)
    client.complete(msgs(), model="m")

    corrupted = next(tmp_path.glob("*.json"))
    corrupted.write_text("{ not json")

    again = client.complete(msgs(), model="m")
    assert inner.calls == 2
    assert again.text


def test_read_only_mode_fails_loudly_on_a_miss(tmp_path):
    """Turns "quietly spent money" into an error, which is what you want when a
    run is meant to be a pure replay."""
    inner = FakeClient()
    client = CachingLLMClient(inner, root=tmp_path, read_only=True)
    with pytest.raises(CacheMiss, match="read-only"):
        client.complete(msgs(), model="m")
    assert inner.calls == 0


def test_read_only_mode_still_serves_hits(tmp_path):
    CachingLLMClient(FakeClient(), root=tmp_path).complete(msgs(), model="m")

    inner = FakeClient()
    replay = CachingLLMClient(inner, root=tmp_path, read_only=True)
    assert replay.complete(msgs(), model="m").usage.cached is True
    assert inner.calls == 0


def test_a_warm_cache_makes_a_sequence_of_calls_fully_reproducible(tmp_path):
    """The determinism claim, end to end: an identical sequence of requests
    replays to identical responses without reaching the model."""
    inner = FakeClient()
    warm = CachingLLMClient(inner, root=tmp_path)
    first_pass = [warm.complete(msgs(f"step-{i}"), model="m").text for i in range(5)]
    assert inner.calls == 5

    replay_inner = FakeClient()
    replay = CachingLLMClient(replay_inner, root=tmp_path, read_only=True)
    second_pass = [replay.complete(msgs(f"step-{i}"), model="m").text for i in range(5)]

    assert first_pass == second_pass
    assert replay_inner.calls == 0
