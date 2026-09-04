"""Position and cash accounting.

A mutable ``Book`` that the venue mutates, which emits immutable
``PortfolioState`` snapshots for the transcript. The split matters: the
transcript must hold a *value*, not a live handle. If a ``Step`` held a
reference to the book, every step in a finished run would show the final
portfolio, and both the rule oracle and the monitor would be reading a
portfolio that had not happened yet -- a lookahead leak by aliasing rather than
by dates, which no date-based tripwire would catch.

Accounting choices, all deliberately the boring option:

* **Average cost basis.** Not FIFO, not tax lots. Realised P&L on a sale is
  ``(price - avg_cost) * quantity``, and the average cost of the remainder is
  unchanged. FIFO would give different realised numbers and therefore different
  cooldown triggers, for no gain to the oversight question.
* **Long only.** ``quantity`` never goes negative; a sale beyond the holding is
  rejected rather than flipped short. Shorting would add borrow, margin, and
  unbounded loss to the rule surface and change nothing about monitoring.
* **Integer shares.** No fractional quantities, so a position either exists or
  does not and there is no dust to reason about.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

from aitvaras.schemas.market import PortfolioState, Position

ZERO = Decimal("0")


class Book:
    """Mutable position and cash ledger. Owned by the venue, never by the agent."""

    def __init__(self, starting_cash: Decimal) -> None:
        self.cash: Decimal = Decimal(starting_cash)
        self._qty: dict[str, int] = {}
        self._avg_cost: dict[str, Decimal] = {}
        self.realized_pnl_today: Decimal = ZERO

    # --- queries ---

    def quantity(self, ticker: str) -> int:
        return self._qty.get(ticker.upper(), 0)

    def avg_cost(self, ticker: str) -> Decimal:
        return self._avg_cost.get(ticker.upper(), ZERO)

    def tickers(self) -> tuple[str, ...]:
        return tuple(sorted(t for t, q in self._qty.items() if q > 0))

    # --- mutations ---

    def start_day(self) -> None:
        """Reset the daily realised-P&L accumulator.

        Called by the engine at the top of each step. Kept explicit rather than
        inferred from a date change, so that the daily-loss rule's notion of
        'today' is a decision the engine makes rather than an accident of when
        a fill happened to be recorded.
        """
        self.realized_pnl_today = ZERO

    def buy(self, ticker: str, quantity: int, price: Decimal) -> None:
        """Add shares at ``price``, blending into the average cost."""
        ticker = ticker.upper()
        cost = Decimal(quantity) * price
        held = self._qty.get(ticker, 0)
        prior = self._avg_cost.get(ticker, ZERO)

        self._avg_cost[ticker] = (prior * held + cost) / Decimal(held + quantity)
        self._qty[ticker] = held + quantity
        self.cash -= cost

    def sell(self, ticker: str, quantity: int, price: Decimal) -> Decimal:
        """Remove shares at ``price``. Returns realised P&L for this sale.

        The return value drives the cooldown rule, which is why it is realised
        P&L rather than proceeds: a sale at a profit starts no cooldown, a sale
        at a loss does, and 'loss' here means against average cost.
        """
        ticker = ticker.upper()
        held = self._qty.get(ticker, 0)
        if quantity > held:
            raise ValueError(f"cannot sell {quantity} {ticker}; only {held} held")

        basis = self._avg_cost.get(ticker, ZERO)
        realized = (price - basis) * Decimal(quantity)

        self._qty[ticker] = held - quantity
        if self._qty[ticker] == 0:
            # Drop the basis when flat, so a later re-entry starts clean rather
            # than blending into the cost of a position that no longer exists.
            self._qty.pop(ticker)
            self._avg_cost.pop(ticker, None)

        self.cash += Decimal(quantity) * price
        self.realized_pnl_today += realized
        return realized

    # --- snapshot ---

    def snapshot(self, as_of: dt.date, marks: dict[str, Decimal]) -> PortfolioState:
        """Freeze the book, valuing positions at ``marks``.

        Raises on a missing mark rather than falling back to average cost.
        A silently stale mark would understate a drawdown, which would quietly
        suppress the daily-loss halt -- a rule failing to fire is far harder to
        notice than one firing wrongly.
        """
        positions: list[Position] = []
        for ticker in self.tickers():
            qty = self._qty[ticker]
            if ticker not in marks:
                raise KeyError(f"no mark for held position {ticker} on {as_of}")
            mark = marks[ticker]
            basis = self._avg_cost[ticker]
            positions.append(
                Position(
                    ticker=ticker,
                    quantity=qty,
                    avg_cost=basis,
                    market_value=Decimal(qty) * mark,
                    unrealized_pnl=(mark - basis) * Decimal(qty),
                )
            )

        return PortfolioState(
            as_of=as_of,
            cash=self.cash,
            equity=self.cash + sum((p.market_value for p in positions), ZERO),
            positions=tuple(positions),
            realized_pnl_today=self.realized_pnl_today,
        )
