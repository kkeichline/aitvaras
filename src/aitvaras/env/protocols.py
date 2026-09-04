"""The swap boundary. Two protocols, and that is the whole of it.

Migrating from replayed CSVs to a broker's paper-trading API means writing new
implementations of these two protocols and changing one line of config. It must
not touch ``rules``, ``agent``, ``journal``, or ``monitor``. If a future change
makes those packages care which implementation is behind the protocol, the
boundary has leaked and the migration will be a rewrite.

A concrete test of whether the boundary still holds: grep the repo for
``ReplayDataSource`` outside ``env/`` and ``tests/``. There should be no hits.
"""

from __future__ import annotations

import datetime as dt
from typing import Protocol, runtime_checkable

from aitvaras.schemas.decision import Order
from aitvaras.schemas.market import PortfolioState, PriceWindow
from aitvaras.schemas.transcript import Fill


@runtime_checkable
class MarketDataSource(Protocol):
    """Supplies price history, and is solely responsible for not leaking the future.

    Implementations must guarantee: ``observe(as_of, ...)`` returns no bar dated
    after ``as_of``. Everything else in the codebase is allowed to trust this,
    which is precisely why it is asserted redundantly at the Observation layer.
    """

    def calendar(self) -> tuple[dt.date, ...]:
        """Every tradable date, ascending. Defines what 'N trading days' means
        for the cooldown rule -- calendar days would silently change the rule's
        strength across holidays."""
        ...

    def universe(self) -> tuple[str, ...]: ...

    def observe(self, as_of: dt.date, lookback: int) -> dict[str, PriceWindow]:
        """History up to and including ``as_of``, at most ``lookback`` bars."""
        ...

    def price_on(self, ticker: str, day: dt.date, field: str = "open") -> object:
        """Single price lookup, for the fill simulator. Separate from
        ``observe`` because the venue legitimately needs the *next* day's open
        to fill today's order -- a lookahead the agent must never have."""
        ...


@runtime_checkable
class ExecutionVenue(Protocol):
    """Turns orders into fills and owns position accounting.

    The venue is the only component permitted to see a price the agent has not
    been shown, because filling today's market order requires tomorrow's open.
    Keeping that privilege here, behind the protocol, is what stops it leaking
    into the agent's observation.
    """

    def submit(self, orders: tuple[Order, ...], as_of: dt.date) -> tuple[Fill, ...]: ...

    def portfolio(self, as_of: dt.date) -> PortfolioState: ...
