"""The lookahead tripwire, and proof that it has teeth.

A test asserting 'no future data leaked' is worthless on its own -- it passes
just as happily against a broken detector as against a correct source. So every
positive assertion here is paired with a negative one against
``LeakyReplaySource``, a deliberately broken twin. If the tripwire ever stops
firing on a known leak, these tests fail, and we learn the guard rotted before
it matters.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from aitvaras.env.replay import LeakyReplaySource, ReplayDataSource
from aitvaras.schemas.decision import LookaheadError, Observation, RuleBook
from aitvaras.schemas.market import PortfolioState
from conftest import PRICES

MIDPOINT = 250


def observation_at(src: ReplayDataSource, idx: int) -> Observation:
    day = src.calendar()[idx]
    return Observation(
        step=idx,
        as_of=day,
        universe=src.universe(),
        history=src.observe(day, lookback=60),
        portfolio=PortfolioState(as_of=day, cash=Decimal("100000"), equity=Decimal("100000")),
        rules=RuleBook(),
    )


def test_observe_returns_nothing_after_as_of(source: ReplayDataSource) -> None:
    for idx in (0, 1, MIDPOINT, len(source.calendar()) - 1):
        day = source.calendar()[idx]
        for ticker, window in source.observe(day, lookback=60).items():
            assert window.bars, f"{ticker} empty at {day}"
            assert window.bars[-1].date <= day, f"{ticker} leaked {window.bars[-1].date} > {day}"


def test_tripwire_passes_on_the_real_source(source: ReplayDataSource) -> None:
    for idx in (0, 1, 5, MIDPOINT, len(source.calendar()) - 1):
        observation_at(source, idx).assert_no_lookahead()


def test_tripwire_fires_on_a_known_leak() -> None:
    """The paired negative. Without this, the test above proves nothing."""
    leaky = LeakyReplaySource(PRICES)
    with pytest.raises(LookaheadError) as excinfo:
        observation_at(leaky, MIDPOINT).assert_no_lookahead()
    assert "leaks" in str(excinfo.value)


def test_tripwire_catches_a_leak_buried_deep_in_the_object(source: ReplayDataSource) -> None:
    """The walk must recurse, not just check the obvious top-level fields.

    A leak planted in a nested dict value is the realistic failure mode: someone
    adds a convenience field to Observation and forgets it carries dates.
    """
    obs = observation_at(source, MIDPOINT)
    future = source.calendar()[MIDPOINT + 5]
    poisoned = obs.model_copy(update={"cooldown_until": {"AAPL": future}})
    with pytest.raises(LookaheadError):
        poisoned.assert_no_lookahead()


def test_last_day_observation_is_still_safe(source: ReplayDataSource) -> None:
    """Boundary: on the final date there is no future to leak, so a naive
    off-by-one silently passes here. Included so the suite does not accidentally
    rely on the last day as its only case."""
    last = len(source.calendar()) - 1
    observation_at(source, last).assert_no_lookahead()
    leaky = LeakyReplaySource(PRICES)
    observation_at(leaky, last).assert_no_lookahead()  # no future exists to leak


def test_observe_on_a_non_trading_day(source: ReplayDataSource) -> None:
    """as_of need not be a trading day. bisect must still exclude the future."""
    day = source.calendar()[MIDPOINT]
    weekend = day + dt.timedelta(days=1) if day.weekday() == 4 else day
    for window in source.observe(weekend, lookback=10).values():
        assert window.bars[-1].date <= weekend
