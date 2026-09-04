"""Accounting invariants.

Accounting bugs do not raise -- they quietly produce a plausible wrong number.
Since realised P&L drives the cooldown rule and equity drives the daily-loss
halt, a silent error here becomes wrong ground truth in phase 2 and a wrong
catch rate in phase 3, with nothing along the way to signal it.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from aitvaras.env.portfolio import Book

D = Decimal
DAY = dt.date(2023, 6, 1)


def test_buy_moves_cash_and_creates_a_position():
    b = Book(D("1000"))
    b.buy("AAPL", 5, D("100"))
    assert b.cash == D("500")
    assert b.quantity("AAPL") == 5
    assert b.avg_cost("AAPL") == D("100")


def test_average_cost_blends_across_buys():
    """5 @ 100 then 5 @ 200 is 10 @ 150, not 10 @ 200 or two separate lots."""
    b = Book(D("10000"))
    b.buy("AAPL", 5, D("100"))
    b.buy("AAPL", 5, D("200"))
    assert b.quantity("AAPL") == 10
    assert b.avg_cost("AAPL") == D("150")


def test_sell_at_a_profit_realises_gain_and_leaves_basis_alone():
    b = Book(D("10000"))
    b.buy("AAPL", 10, D("100"))
    realized = b.sell("AAPL", 4, D("130"))
    assert realized == D("120")  # (130 - 100) * 4
    assert b.quantity("AAPL") == 6
    assert b.avg_cost("AAPL") == D("100")  # unchanged by a sale
    assert b.realized_pnl_today == D("120")


def test_sell_at_a_loss_realises_a_negative_number():
    """This is what starts a cooldown, so the sign matters more than usual."""
    b = Book(D("10000"))
    b.buy("XOM", 10, D("100"))
    assert b.sell("XOM", 10, D("90")) == D("-100")


def test_partial_sale_at_a_loss_still_realises_a_loss():
    """The cooldown rule triggers on any realised loss, including partial --
    otherwise 'sell 99%, keep one share' walks straight through it."""
    b = Book(D("10000"))
    b.buy("XOM", 10, D("100"))
    assert b.sell("XOM", 1, D("90")) < 0
    assert b.quantity("XOM") == 9


def test_closing_flat_drops_the_basis_so_re_entry_starts_clean():
    """Without this, re-buying after a full exit blends into the cost of a
    position that no longer exists, and every later realised P&L is wrong."""
    b = Book(D("10000"))
    b.buy("AAPL", 5, D("100"))
    b.sell("AAPL", 5, D("120"))
    assert b.quantity("AAPL") == 0
    b.buy("AAPL", 5, D("200"))
    assert b.avg_cost("AAPL") == D("200")


def test_cannot_sell_more_than_held():
    b = Book(D("10000"))
    b.buy("AAPL", 5, D("100"))
    with pytest.raises(ValueError, match="only 5 held"):
        b.sell("AAPL", 6, D("100"))


def test_realised_pnl_accumulates_within_a_day_and_resets_on_start_day():
    b = Book(D("10000"))
    b.buy("AAPL", 10, D("100"))
    b.sell("AAPL", 5, D("110"))
    b.sell("AAPL", 5, D("90"))
    assert b.realized_pnl_today == D("0")  # +50 then -50
    b.start_day()
    assert b.realized_pnl_today == D("0")


def test_snapshot_values_positions_at_the_supplied_marks():
    b = Book(D("10000"))
    b.buy("AAPL", 10, D("100"))
    snap = b.snapshot(DAY, {"AAPL": D("120")})
    assert snap.cash == D("9000")
    assert snap.equity == D("10200")  # 9000 cash + 10 * 120
    pos = snap.position_in("AAPL")
    assert pos is not None
    assert pos.unrealized_pnl == D("200")


def test_snapshot_refuses_to_guess_a_missing_mark():
    """Falling back to average cost would understate a drawdown, which would
    silently suppress the daily-loss halt. A rule that fails to fire is far
    harder to notice than one that fires wrongly."""
    b = Book(D("10000"))
    b.buy("AAPL", 1, D("100"))
    with pytest.raises(KeyError, match="no mark"):
        b.snapshot(DAY, {})


def test_round_trip_at_the_same_price_conserves_equity():
    """The sanity check that catches sign errors and double-counting."""
    b = Book(D("10000"))
    b.buy("AAPL", 10, D("100"))
    b.sell("AAPL", 10, D("100"))
    assert b.cash == D("10000")
    assert b.snapshot(DAY, {}).equity == D("10000")


def test_ticker_case_does_not_create_two_positions():
    b = Book(D("10000"))
    b.buy("aapl", 5, D("100"))
    b.buy("AAPL", 5, D("100"))
    assert b.quantity("AAPL") == 10
    assert b.tickers() == ("AAPL",)
