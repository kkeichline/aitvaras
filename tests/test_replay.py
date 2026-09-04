"""The replay source's own invariants -- the ones the lookahead guard assumes."""

from __future__ import annotations

import datetime as dt

import pytest

from aitvaras.env.protocols import MarketDataSource
from aitvaras.env.replay import ReplayDataSource
from conftest import PRICES


def test_replay_satisfies_the_market_data_protocol(source):
    """If this breaks, the broker swap in phase 4 breaks with it."""
    assert isinstance(source, MarketDataSource)


def test_calendar_is_sorted_and_unique(source):
    cal = source.calendar()
    assert list(cal) == sorted(cal)
    assert len(set(cal)) == len(cal)


def test_calendar_is_the_intersection_not_the_union(source):
    """Every ticker must have a bar on every calendar date.

    A union calendar would give the agent ragged windows, and 'N trading days'
    would silently mean something different per ticker -- quietly changing how
    strong the cooldown rule is.
    """
    for day in (source.calendar()[0], source.calendar()[-1]):
        windows = source.observe(day, lookback=1)
        assert set(windows) == set(source.universe())
        assert all(w.bars[-1].date == day for w in windows.values())


def test_lookback_limits_window_length(source):
    day = source.calendar()[300]
    for w in source.observe(day, lookback=20).values():
        assert len(w.bars) == 20


def test_window_is_short_at_the_start_of_history(source):
    """No padding, no wrap-around. Early steps legitimately see less."""
    day = source.calendar()[2]
    for w in source.observe(day, lookback=60).values():
        assert len(w.bars) == 3


def test_next_trading_day_counts_trading_days_not_calendar_days(source):
    """The cooldown rule depends on this: a weekend must not shorten it."""
    cal = source.calendar()
    friday = next(d for d in cal if d.weekday() == 4)
    nxt = source.next_trading_day(friday, 1)
    assert nxt is not None
    assert (nxt - friday).days > 1  # skipped the weekend
    assert nxt.weekday() < 5


def test_next_trading_day_returns_none_past_the_end(source):
    assert source.next_trading_day(source.calendar()[-1], 1) is None


def test_price_on_requires_an_exact_date(source):
    day = source.calendar()[100]
    assert source.price_on("AAPL", day, "open") > 0
    with pytest.raises(KeyError):
        source.price_on("AAPL", day + dt.timedelta(days=1000))


def test_unknown_ticker_is_rejected_at_construction():
    with pytest.raises(FileNotFoundError):
        ReplayDataSource(PRICES, tickers=("NOSUCH",))


def test_missing_directory_is_rejected():
    with pytest.raises(FileNotFoundError):
        ReplayDataSource(PRICES / "nope")
