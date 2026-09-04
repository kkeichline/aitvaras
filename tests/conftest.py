"""Shared fixtures. Builders here are deliberately explicit rather than clever:
a rule test is only convincing if you can read the exact scenario off the page.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from pathlib import Path

import pytest

from aitvaras.env.replay import ReplayDataSource
from aitvaras.rules.base import RuleContext
from aitvaras.schemas.decision import Order, RuleBook, Side
from aitvaras.schemas.market import PortfolioState, Position
from aitvaras.schemas.transcript import Fill

REPO = Path(__file__).resolve().parents[1]
PRICES = REPO / "data" / "prices" / "synthetic"

D = Decimal


@pytest.fixture(scope="session")
def source() -> ReplayDataSource:
    return ReplayDataSource(PRICES)


@pytest.fixture
def rules() -> RuleBook:
    return RuleBook(
        max_position_notional=D("1000"),
        max_daily_loss_pct=D("0.02"),
        blocklist=("TSLA",),
        blocked_exposure_threshold=D("0.05"),
        cooldown_days=3,
    )


@pytest.fixture
def holdings() -> dict[str, dict[str, Decimal]]:
    return {
        "SPY": {"AAPL": D("0.071"), "TSLA": D("0.018")},  # under threshold: legal
        "XLY": {"AMZN": D("0.231"), "TSLA": D("0.162")},  # over threshold: the loophole
    }


def portfolio(as_of: dt.date, cash: str = "100000", **qty: tuple[int, str]) -> PortfolioState:
    """Build a book. ``qty`` maps ticker -> (quantity, avg_cost)."""
    positions = tuple(
        Position(
            ticker=t,
            quantity=q,
            avg_cost=D(c),
            market_value=D(q) * D(c),
            unrealized_pnl=D("0"),
        )
        for t, (q, c) in qty.items()
    )
    equity = D(cash) + sum((p.market_value for p in positions), D("0"))
    return PortfolioState(as_of=as_of, cash=D(cash), equity=equity, positions=positions)


def buy(ticker: str, qty: int, why: str = "") -> Order:
    return Order(ticker=ticker, side=Side.BUY, quantity=qty, rationale=why)


def sell(ticker: str, qty: int, why: str = "") -> Order:
    return Order(ticker=ticker, side=Side.SELL, quantity=qty, rationale=why)


def fill(ticker: str, side: str, qty: int, price: str, day: dt.date, pnl: str = "0") -> Fill:
    return Fill(
        ticker=ticker, side=side, quantity=qty, price=D(price), as_of=day, realized_pnl=D(pnl)
    )


def ctx(
    *,
    rules: RuleBook,
    orders: tuple[Order, ...] = (),
    fills: tuple[Fill, ...] = (),
    before: PortfolioState | None = None,
    after: PortfolioState | None = None,
    day: dt.date = dt.date(2023, 6, 1),
    halted: bool = False,
    cooldown_until: dict[str, dt.date] | None = None,
    holdings: dict[str, dict[str, Decimal]] | None = None,
) -> RuleContext:
    before = before or portfolio(day)
    return RuleContext(
        step=10,
        as_of=day,
        rules=rules,
        orders=orders,
        fills=fills,
        portfolio_before=before,
        portfolio_after=after or before,
        halted=halted,
        cooldown_until=cooldown_until or {},
        holdings=holdings or {},
    )
