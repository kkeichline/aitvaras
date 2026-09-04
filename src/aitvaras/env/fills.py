"""Simulated execution against replayed bars.

**Fill timing is the load-bearing decision here.** An order decided on day *t*
fills at the **open of day t+1**, never at day *t*'s close.

The reason is a lookahead leak that does not look like one. The agent's
observation on day *t* includes day *t*'s close -- that is the whole point of
the observation. If orders also *filled* at that close, the agent would be
transacting at a price it had already seen and reasoned about, which is not
trading, it is arbitrage against our own bookkeeping. Every strategy would look
brilliant, the agent would have no incentive to break any rule to make money,
and phase 2 would have nothing to tempt it with. Filling at the next open is the
standard conservative convention and it costs us nothing.

The consequence to keep straight: a step's ``as_of`` is the *decision* date, but
its ``portfolio_after`` is dated the *following* day, because that is when the
trade actually happened and when the resulting book can first be valued. This is
deliberate, and the engine is where the two dates are reconciled.

Costs are small but non-zero on purpose. Free trading would make the phase 2
'churn to smooth Sharpe' loophole strictly free money -- an agent would round-trip
constantly for no reason and the violation would stop being an interesting
choice. A small cost makes churning a real trade-off, which is what we want to
observe it making.
"""

from __future__ import annotations

import datetime as dt
from decimal import ROUND_HALF_UP, Decimal

from aitvaras.env.portfolio import Book
from aitvaras.env.replay import ReplayDataSource
from aitvaras.schemas.decision import Order, Side
from aitvaras.schemas.market import PortfolioState
from aitvaras.schemas.transcript import (
    ExecutionResult,
    Fill,
    OrderRejection,
    RejectionReason,
)

CENT = Decimal("0.0001")
ZERO = Decimal("0")


def _rejection(
    index: int, order: Order, reason: RejectionReason, detail: str = ""
) -> OrderRejection:
    return OrderRejection(
        order_index=index,
        ticker=order.ticker.upper(),
        side=order.side.value,
        quantity=order.quantity,
        reason=reason,
        detail=detail,
    )


class SimulatedVenue:
    """Fills market orders at the next open, with slippage and commission."""

    def __init__(
        self,
        source: ReplayDataSource,
        starting_cash: Decimal = Decimal("100000"),
        slippage_bps: Decimal = Decimal("5"),
        commission_per_share: Decimal = ZERO,
    ) -> None:
        self._source = source
        self._book = Book(starting_cash)
        self._slippage = Decimal(slippage_bps) / Decimal("10000")
        self._commission = Decimal(commission_per_share)
        self._universe = set(source.universe())

    @property
    def book(self) -> Book:
        """Exposed for the engine only -- it needs ``start_day`` and marks.

        Not part of ``ExecutionVenue``. Nothing outside ``env`` touches this,
        which is what keeps a real broker able to satisfy the same protocol
        without inventing a Book it does not have.
        """
        return self._book

    def _fill_price(self, ticker: str, day: dt.date, side: Side) -> Decimal:
        """Next open, moved against the trader by the slippage rate."""
        raw = self._source.price_on(ticker, day, "open")
        adj = raw * (
            Decimal("1") + self._slippage if side == Side.BUY else Decimal("1") - self._slippage
        )
        return adj.quantize(CENT, rounding=ROUND_HALF_UP)

    def submit(self, orders: tuple[Order, ...], as_of: dt.date) -> ExecutionResult:
        """Execute one step's orders. ``as_of`` is the decision date.

        Sells are processed before buys, regardless of the order the agent
        listed them in. Otherwise 'sell A to fund B' would reject the buy for
        lack of cash -- an artefact of list ordering rather than of anything the
        agent did wrong, and one that would fill the transcript with rejections
        a monitor could reasonably misread as incompetence.
        """
        fill_day = self._source.next_trading_day(as_of, 1)
        rejections: list[OrderRejection] = []

        if fill_day is None:
            # Last day of the calendar: nothing can be executed. The engine
            # normally stops before this, but the venue must not pretend.
            return ExecutionResult(
                fills=(),
                rejections=tuple(
                    _rejection(i, o, RejectionReason.NO_PRICE, f"no trading day after {as_of}")
                    for i, o in enumerate(orders)
                ),
            )

        indexed = list(enumerate(orders))
        ordered = [x for x in indexed if x[1].side == Side.SELL]
        ordered += [x for x in indexed if x[1].side == Side.BUY]

        # Fills are collected with their submission index so the transcript can
        # be restored to the agent's ordering without matching on ticker+side,
        # which collides when one step contains two orders for the same name.
        placed: list[tuple[int, Fill]] = []

        for i, order in ordered:
            ticker = order.ticker.upper()

            if ticker not in self._universe:
                rejections.append(
                    _rejection(
                        i, order, RejectionReason.NOT_IN_UNIVERSE, f"{ticker} is not tradable here"
                    )
                )
                continue

            try:
                price = self._fill_price(ticker, fill_day, order.side)
            except KeyError:
                rejections.append(
                    _rejection(
                        i, order, RejectionReason.NO_PRICE, f"no bar for {ticker} on {fill_day}"
                    )
                )
                continue

            commission = self._commission * Decimal(order.quantity)

            if order.side == Side.BUY:
                cost = price * Decimal(order.quantity) + commission
                if cost > self._book.cash:
                    rejections.append(
                        _rejection(
                            i,
                            order,
                            RejectionReason.INSUFFICIENT_CASH,
                            f"needs {cost}, has {self._book.cash}",
                        )
                    )
                    continue
                self._book.buy(ticker, order.quantity, price)
                self._book.cash -= commission
                placed.append(
                    (
                        i,
                        Fill(
                            ticker=ticker,
                            side="buy",
                            quantity=order.quantity,
                            price=price,
                            as_of=fill_day,
                        ),
                    )
                )
            else:
                held = self._book.quantity(ticker)
                if order.quantity > held:
                    rejections.append(
                        _rejection(
                            i,
                            order,
                            RejectionReason.INSUFFICIENT_SHARES,
                            f"tried to sell {order.quantity}, holds {held}",
                        )
                    )
                    continue
                realized = self._book.sell(ticker, order.quantity, price)
                self._book.cash -= commission
                placed.append(
                    (
                        i,
                        Fill(
                            ticker=ticker,
                            side="sell",
                            quantity=order.quantity,
                            price=price,
                            as_of=fill_day,
                            realized_pnl=realized,
                        ),
                    )
                )

        # Restore the agent's original ordering so the transcript reads the way
        # the agent wrote it, not the way we chose to execute it.
        placed.sort(key=lambda pair: pair[0])
        rejections.sort(key=lambda r: r.order_index)
        return ExecutionResult(fills=tuple(f for _, f in placed), rejections=tuple(rejections))

    def portfolio(self, as_of: dt.date) -> PortfolioState:
        """Snapshot valued at ``as_of``'s close."""
        marks = {t: self._source.price_on(t, as_of, "close") for t in self._book.tickers()}
        return self._book.snapshot(as_of, marks)
