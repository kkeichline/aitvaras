"""Replay historical daily bars, one step at a time, without leaking the future.

The defence is structural, not procedural. ``ReplayDataSource`` holds the full
history in ``self._bars``, a private attribute, and every public method returns
a *copy of a slice*. The engine holds the source; the agent holds only the
``Observation`` the engine built. There is no reference path from the agent to
a future bar -- not one that is checked and rejected, one that does not exist.

``LeakyReplaySource`` at the bottom is a deliberately broken twin used only by
the test suite. A lookahead test that has never actually caught a leak proves
nothing about its own sensitivity, so we keep a known leak around to prove the
tripwire fires.
"""

from __future__ import annotations

import csv
import datetime as dt
from bisect import bisect_right
from decimal import Decimal
from pathlib import Path

from aitvaras.schemas.market import Bar, PriceWindow


class ReplayDataSource:
    """Lookahead-safe replay over per-ticker CSV files.

    Expects ``<root>/<TICKER>.csv`` with header ``date,open,high,low,close,volume``.
    """

    def __init__(self, root: Path | str, tickers: tuple[str, ...] | None = None) -> None:
        self._root = Path(root)
        if not self._root.is_dir():
            raise FileNotFoundError(f"price data directory not found: {self._root}")

        discovered = tuple(sorted(p.stem.upper() for p in self._root.glob("*.csv")))
        self._universe = tuple(t.upper() for t in tickers) if tickers else discovered
        missing = [t for t in self._universe if not (self._root / f"{t}.csv").is_file()]
        if missing:
            raise FileNotFoundError(f"no CSV for tickers: {missing} under {self._root}")

        # Private. Nothing outside this class ever receives this dict.
        self._bars: dict[str, list[Bar]] = {t: self._load(t) for t in self._universe}
        self._dates: dict[str, list[dt.date]] = {
            t: [b.date for b in bars] for t, bars in self._bars.items()
        }

        # The trading calendar is the intersection: a date is tradable only if
        # every ticker has a bar for it. Using the union would hand the agent
        # windows with ragged endpoints, and 'N trading days' would mean a
        # different thing per ticker -- quietly changing the cooldown rule.
        common = set.intersection(*(set(d) for d in self._dates.values()))
        self._calendar: tuple[dt.date, ...] = tuple(sorted(common))
        if not self._calendar:
            raise ValueError("no dates common to all tickers; cannot build a calendar")

    def _load(self, ticker: str) -> list[Bar]:
        path = self._root / f"{ticker}.csv"
        bars: list[Bar] = []
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                bars.append(
                    Bar(
                        date=dt.date.fromisoformat(row["date"]),
                        open=Decimal(row["open"]),
                        high=Decimal(row["high"]),
                        low=Decimal(row["low"]),
                        close=Decimal(row["close"]),
                        volume=int(row["volume"]),
                    )
                )
        bars.sort(key=lambda b: b.date)
        return bars

    # --- MarketDataSource ---

    def calendar(self) -> tuple[dt.date, ...]:
        return self._calendar

    def universe(self) -> tuple[str, ...]:
        return self._universe

    def observe(self, as_of: dt.date, lookback: int = 60) -> dict[str, PriceWindow]:
        """Bars up to and including ``as_of``, at most ``lookback`` of them.

        ``bisect_right`` gives the index one past the last bar dated <= as_of,
        so the slice can never include a future bar even when ``as_of`` is not
        itself a trading day.
        """
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        out: dict[str, PriceWindow] = {}
        for ticker in self._universe:
            hi = bisect_right(self._dates[ticker], as_of)
            window = self._bars[ticker][max(0, hi - lookback) : hi]
            if window:
                out[ticker] = PriceWindow(ticker=ticker, bars=tuple(window))
        return out

    def price_on(self, ticker: str, day: dt.date, field: str = "open") -> Decimal:
        """Exact-date price lookup for the fill simulator.

        Deliberately *not* lookahead-safe: the venue needs the next open to fill
        today's order. That privilege is confined to this method, which the
        agent path never calls.
        """
        ticker = ticker.upper()
        idx = bisect_right(self._dates[ticker], day) - 1
        if idx < 0 or self._dates[ticker][idx] != day:
            raise KeyError(f"no bar for {ticker} on {day}")
        return getattr(self._bars[ticker][idx], field)

    def next_trading_day(self, day: dt.date, offset: int = 1) -> dt.date | None:
        """Calendar arithmetic in trading days, not calendar days."""
        idx = bisect_right(self._calendar, day) - 1
        if idx < 0 or self._calendar[idx] != day:
            idx = bisect_right(self._calendar, day) - 1
        target = idx + offset
        if 0 <= target < len(self._calendar):
            return self._calendar[target]
        return None


class LeakyReplaySource(ReplayDataSource):
    """Deliberately broken. Test fixture only -- never importable into a run.

    Returns one bar *past* ``as_of``, the classic off-by-one that an
    ``<=``/``<`` slip would produce. Its whole job is to make the lookahead
    tripwire fail, proving the tripwire has teeth.
    """

    def observe(self, as_of: dt.date, lookback: int = 60) -> dict[str, PriceWindow]:
        out: dict[str, PriceWindow] = {}
        for ticker in self._universe:
            hi = bisect_right(self._dates[ticker], as_of) + 1  # <-- the bug
            window = self._bars[ticker][max(0, hi - lookback) : hi]
            if window:
                out[ticker] = PriceWindow(ticker=ticker, bars=tuple(window))
        return out
