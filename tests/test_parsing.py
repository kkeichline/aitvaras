"""Reading a Decision out of whatever the model said.

Tolerant about format, strict about content. The asymmetry is deliberate: code
fences carry no information about honesty, but a coerced quantity would put words
in the agent's mouth and then grade it on them.
"""

from __future__ import annotations

import pytest

from aitvaras.agent.parsing import extract_json, parse_decision

GOOD = '{"orders": [{"ticker": "AAPL", "side": "buy", "quantity": 10}], "journal": "buying"}'


# --------------------------------------------------------------------------
# Tolerant about format
# --------------------------------------------------------------------------


def test_bare_json_parses():
    decision, error = parse_decision(GOOD)
    assert error is None
    assert decision is not None
    assert decision.orders[0].ticker == "AAPL"


def test_fenced_json_parses():
    decision, error = parse_decision(f"```json\n{GOOD}\n```")
    assert error is None and decision is not None


def test_unlabelled_fence_parses():
    decision, error = parse_decision(f"```\n{GOOD}\n```")
    assert error is None and decision is not None


def test_json_surrounded_by_prose_parses():
    text = f"Here is my decision for today.\n\n{GOOD}\n\nLet me know if you need more."
    decision, error = parse_decision(text)
    assert error is None and decision is not None


def test_nested_objects_in_orders_parse():
    """A naive brace regex stops at the first inner close. The orders list is a
    list of objects, so this is the common case, not an edge case."""
    text = (
        '{"orders": [{"ticker": "AAPL", "side": "buy", "quantity": 1},'
        ' {"ticker": "KO", "side": "sell", "quantity": 2}], "journal": "two trades"}'
    )
    decision, error = parse_decision(text)
    assert error is None and decision is not None
    assert len(decision.orders) == 2


def test_braces_inside_strings_do_not_confuse_the_scan():
    text = '{"orders": [], "journal": "I considered {AAPL, MSFT} and held"}'
    decision, error = parse_decision(text)
    assert error is None and decision is not None
    assert "{AAPL, MSFT}" in decision.journal


def test_an_empty_order_list_is_a_valid_hold():
    decision, error = parse_decision('{"orders": [], "journal": "nothing attractive"}')
    assert error is None
    assert decision is not None and decision.orders == ()


# --------------------------------------------------------------------------
# Strict about content
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload,because",
    [
        ('{"orders": [{"ticker": "AAPL", "side": "buy", "quantity": 0}]}', "zero qty"),
        ('{"orders": [{"ticker": "AAPL", "side": "buy", "quantity": -5}]}', "negative"),
        ('{"orders": [{"ticker": "AAPL", "side": "hold", "quantity": 1}]}', "bad side"),
        ('{"orders": [{"side": "buy", "quantity": 1}]}', "no ticker"),
        ('{"orders": [{"ticker": "AAPL", "quantity": 1}]}', "no side"),
    ],
)
def test_malformed_orders_are_failures_not_guesses(payload, because):
    decision, error = parse_decision(payload)
    assert decision is None, because
    assert error and "validation" in error


def test_a_bare_orders_array_is_not_silently_read_as_a_hold():
    """The bug this test was written to catch.

    A model returning just the orders array is a plausible slip. The brace scan
    finds the inner object, it validates as a Decision with every field at its
    default, and the step is recorded as "decided to hold, no journal" -- a
    fabricated decision in a labelled dataset. Requiring an 'orders' or
    'journal' key makes it a parse failure instead.
    """
    decision, error = parse_decision('[{"ticker": "AAPL", "side": "buy", "quantity": 1}]')
    assert decision is None
    assert error and "orders" in error


def test_an_unrelated_json_object_is_not_a_decision():
    decision, error = parse_decision('{"status": "thinking", "note": "no trades"}')
    assert decision is None
    assert error


def test_an_empty_object_is_not_a_hold():
    decision, error = parse_decision("{}")
    assert decision is None, "an empty object must not become a silent hold"


def test_journal_only_is_a_valid_hold():
    """Distinct from the case above: naming 'journal' makes the intent explicit,
    so an absent orders list legitimately means no trades."""
    decision, error = parse_decision('{"journal": "Holding, nothing attractive."}')
    assert error is None
    assert decision is not None and decision.orders == ()


# --------------------------------------------------------------------------
# Failure is recorded, never raised
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "I don't want to trade today.",
        "```json\nnot json at all\n```",
        "{",
        "{unclosed: ",
        '{"orders": [',
        "null",
        "42",
    ],
)
def test_unparseable_responses_return_a_reason_and_never_raise(text):
    """A malformed response is data, not an error.

    An agent that becomes unparseable on exactly the steps where it misbehaves is
    a finding. Retrying until the JSON is clean would erase that finding and
    replace it with a tidy hold.
    """
    decision, error = parse_decision(text)
    assert decision is None
    assert error


def test_extract_json_returns_none_when_there_is_nothing_to_find():
    assert extract_json("just some prose") is None
    assert extract_json("") is None
