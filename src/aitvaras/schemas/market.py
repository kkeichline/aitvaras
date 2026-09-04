"""What the agent is allowed to know about the world.

The central safety property of this package is *negative*: an ``Observation``
must never contain information dated after ``as_of``. Every other guarantee in
the benchmark rests on that, because an agent that can see the future is not
misbehaving in an interesting way -- it is just cheating at a game we
accidentally rigged, and the monitor's job becomes meaningless.

We enforce this two ways, deliberately redundant:

1. *Structurally* -- ``ReplayDataSource`` owns the full price history as a
   private attribute and hands out slices. The agent is never given a reference
   it could walk back to future rows. This is the real defence.
2. *By assertion* -- ``Observation.assert_no_lookahead()`` walks the serialized
   object and fails if any date exceeds ``as_of``. This is the tripwire that
   catches a future refactor quietly breaking (1).
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Bar(BaseModel):
    """One daily OHLCV candle. The atom of the whole environment.

    Daily bars are a deliberate simplification: they make one agent turn equal
    one trading day, which keeps transcripts short enough for a monitor to read
    in a single context window. Intraday data would multiply transcript length
    by ~400x and buy us nothing for an oversight question.
    """

    model_config = ConfigDict(frozen=True)

    date: dt.date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_ohlc_coherent(self) -> Bar:
        if not (self.low <= self.open <= self.high and self.low <= self.close <= self.high):
            raise ValueError(f"incoherent OHLC on {self.date}: {self!r}")
        return self


class PriceWindow(BaseModel):
    """A contiguous run of bars for one ticker, ending at the observation date.

    Invariant, checked here rather than trusted: bars are sorted ascending and
    contain no duplicate dates. A silently mis-sorted window would let the agent
    read a future close as if it were the latest one -- a lookahead leak that
    would not trip any date-based check, because all the dates are legal.
    """

    model_config = ConfigDict(frozen=True)

    ticker: str
    bars: tuple[Bar, ...]

    @model_validator(mode="after")
    def _check_sorted_unique(self) -> PriceWindow:
        dates = [b.date for b in self.bars]
        if dates != sorted(dates):
            raise ValueError(f"{self.ticker}: bars not in ascending date order")
        if len(set(dates)) != len(dates):
            raise ValueError(f"{self.ticker}: duplicate bar dates")
        return self

    @property
    def latest(self) -> Bar:
        return self.bars[-1]


class Position(BaseModel):
    """A holding in one ticker, marked to the most recent close.

    Long-only in phase 1. Shorting doubles the rule surface (borrow, margin,
    unbounded loss) and adds nothing to the oversight question, so
    ``quantity`` is constrained non-negative and we revisit only if a red-team
    vector actually needs it.
    """

    model_config = ConfigDict(frozen=True)

    ticker: str
    quantity: int = Field(ge=0)
    avg_cost: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal


class PortfolioState(BaseModel):
    """The agent's book at a point in time.

    ``equity`` is cash plus the marked value of every position -- the single
    number the score is computed from, and therefore the number the agent is
    under pressure to move.
    """

    model_config = ConfigDict(frozen=True)

    as_of: dt.date
    cash: Decimal
    equity: Decimal
    positions: tuple[Position, ...] = ()
    realized_pnl_today: Decimal = Decimal("0")

    def position_in(self, ticker: str) -> Position | None:
        return next((p for p in self.positions if p.ticker == ticker), None)

    def quantity_in(self, ticker: str) -> int:
        p = self.position_in(ticker)
        return p.quantity if p else 0
