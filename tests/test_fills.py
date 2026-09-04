"""Execution semantics -- above all, *when* an order fills.

The next-open convention is the quiet guardian of the lookahead property. It has
its own test with an explicit negative assertion, because if fills ever slipped
to the same day's close the suite would otherwise stay green while every run in
the benchmark became worthless.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from aitvaras.env.fills import SimulatedVenue
from aitvaras.env.protocols import ExecutionVenue
from aitvaras.schemas.transcript import RejectionReason
from conftest import buy, sell

D = Decimal


@pytest.fixture
def venue(source):
    return SimulatedVenue(source, starting_cash=D("100000"), slippage_bps=D("0"))


def test_venue_satisfies_the_execution_protocol(venue):
    assert isinstance(venue, ExecutionVenue)


def test_orders_fill_at_the_next_open_not_todays_close(source, venue):
    """THE test. The agent has already seen today's close in its observation;
    filling there would let it transact on a price it had reasoned about."""
    day = source.calendar()[100]
    nxt = source.calendar()[101]

    res = venue.submit((buy("AAPL", 1),), day)
    assert len(res.fills) == 1
    fill = res.fills[0]

    assert fill.as_of == nxt
    assert fill.price == source.price_on("AAPL", nxt, "open")
    assert fill.price != source.price_on("AAPL", day, "close")


def test_slippage_moves_the_price_against_the_trader(source):
    day = source.calendar()[100]
    nxt = source.calendar()[101]
    open_ = source.price_on("AAPL", nxt, "open")

    v = SimulatedVenue(source, starting_cash=D("100000"), slippage_bps=D("50"))
    bought = v.submit((buy("AAPL", 1),), day).fills[0]
    assert bought.price > open_

    sold = v.submit((sell("AAPL", 1),), day).fills[0]
    assert sold.price < open_


def test_sells_are_processed_before_buys(source):
    """'Sell A to fund B' must work. Executing in list order would reject the
    buy for lack of cash -- an artefact of ordering, not of agent behaviour, and
    one a monitor could reasonably misread as incompetence.
    """
    day = source.calendar()[100]
    v = SimulatedVenue(source, starting_cash=D("0"), slippage_bps=D("0"))
    v.book.buy("AAPL", 10, D("100"))  # seed a holding, no cash left

    res = v.submit((buy("KO", 1), sell("AAPL", 10)), day)
    assert res.rejections == ()
    assert {f.ticker for f in res.fills} == {"AAPL", "KO"}


def test_fills_are_reported_in_the_agents_submission_order(source):
    """Execution order is ours; transcript order should be the agent's."""
    day = source.calendar()[100]
    v = SimulatedVenue(source, starting_cash=D("100000"), slippage_bps=D("0"))
    v.book.buy("AAPL", 5, D("100"))

    res = v.submit((buy("KO", 1), sell("AAPL", 5), buy("MSFT", 1)), day)
    assert [f.ticker for f in res.fills] == ["KO", "AAPL", "MSFT"]


def test_two_orders_for_the_same_ticker_keep_their_order(source):
    """The case that breaks a naive ticker+side lookup when restoring order."""
    day = source.calendar()[100]
    v = SimulatedVenue(source, starting_cash=D("100000"), slippage_bps=D("0"))
    res = v.submit((buy("AAPL", 1), buy("AAPL", 2)), day)
    assert [f.quantity for f in res.fills] == [1, 2]


def test_buy_beyond_cash_is_rejected_and_leaves_the_book_untouched(source):
    day = source.calendar()[100]
    v = SimulatedVenue(source, starting_cash=D("10"), slippage_bps=D("0"))

    res = v.submit((buy("AAPL", 100),), day)
    assert res.fills == ()
    assert res.rejections[0].reason is RejectionReason.INSUFFICIENT_CASH
    assert v.book.cash == D("10")
    assert v.book.quantity("AAPL") == 0


def test_selling_shares_not_held_is_rejected(venue, source):
    day = source.calendar()[100]
    res = venue.submit((sell("AAPL", 1),), day)
    assert res.fills == ()
    assert res.rejections[0].reason is RejectionReason.INSUFFICIENT_SHARES


def test_unknown_ticker_is_rejected(venue, source):
    day = source.calendar()[100]
    res = venue.submit((buy("NVDA", 1),), day)
    assert res.rejections[0].reason is RejectionReason.NOT_IN_UNIVERSE


def test_a_rejection_does_not_block_the_other_orders(venue, source):
    """One bad order must not void the step, or a single typo would silently
    erase a day of behaviour from the transcript."""
    day = source.calendar()[100]
    res = venue.submit((buy("NVDA", 1), buy("AAPL", 1)), day)
    assert len(res.fills) == 1 and res.fills[0].ticker == "AAPL"
    assert len(res.rejections) == 1


def test_the_last_calendar_day_can_execute_nothing(venue, source):
    """There is no next open to fill against. The venue says so rather than
    quietly filling at today's price."""
    res = venue.submit((buy("AAPL", 1),), source.calendar()[-1])
    assert res.fills == ()
    assert res.rejections[0].reason is RejectionReason.NO_PRICE


def test_rejection_indices_point_back_at_the_original_orders(venue, source):
    """The rule oracle attributes violations by order index, so a shifted index
    would mislabel which order broke a rule."""
    day = source.calendar()[100]
    res = venue.submit((buy("AAPL", 1), buy("NVDA", 1), sell("KO", 5)), day)
    assert sorted(r.order_index for r in res.rejections) == [1, 2]


def test_commission_is_charged_on_top_of_the_fill(source):
    day = source.calendar()[100]
    nxt = source.calendar()[101]
    v = SimulatedVenue(
        source,
        starting_cash=D("100000"),
        slippage_bps=D("0"),
        commission_per_share=D("0.01"),
    )
    v.submit((buy("AAPL", 100),), day)
    expected = D("100000") - source.price_on("AAPL", nxt, "open") * 100 - D("1")
    assert v.book.cash == expected


def test_portfolio_marks_at_the_close_of_the_requested_day(source, venue):
    day = source.calendar()[100]
    nxt = source.calendar()[101]
    venue.submit((buy("AAPL", 10),), day)

    snap = venue.portfolio(nxt)
    pos = snap.position_in("AAPL")
    assert pos is not None
    assert pos.market_value == source.price_on("AAPL", nxt, "close") * 10
