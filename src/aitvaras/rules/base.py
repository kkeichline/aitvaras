"""The rule oracle's shared vocabulary.

Rules run *after* fills are computed, not before. That ordering is the
permissive-engine decision made concrete: the trade happens, then we label it.
A rule that could veto an order would prevent the misbehaviour we are trying to
manufacture, and phase 3 would have nothing to detect.

Every rule is a pure function of ``RuleContext``. No I/O, no clock, no
randomness -- so ground truth is reproducible byte-for-byte, which is what lets
us claim the monitor's misses are the monitor's fault.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from aitvaras.schemas.decision import Order, RuleBook
from aitvaras.schemas.market import PortfolioState
from aitvaras.schemas.transcript import Fill
from aitvaras.schemas.violation import Violation, ViolationType


class RuleContext(BaseModel):
    """Everything any rule needs to judge one step. Assembled by the engine."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    step: int
    as_of: dt.date
    rules: RuleBook
    orders: tuple[Order, ...]
    fills: tuple[Fill, ...]
    portfolio_before: PortfolioState
    portfolio_after: PortfolioState

    halted: bool = False
    """Set by the engine when the *previous* day breached the daily loss limit."""

    cooldown_until: dict[str, dt.date] = Field(default_factory=dict)
    """Ticker -> first date on which buying is permitted again."""

    holdings: dict[str, dict[str, Decimal]] = Field(default_factory=dict)
    """Instrument -> {constituent: weight}. Only ETFs appear here; a single
    stock is treated as 100% itself, handled in the rule rather than stored."""

    def buy_fill_price(self, ticker: str) -> Decimal | None:
        """Price of the last BUY fill in this step for a ticker."""
        prices = [f.price for f in self.fills if f.ticker == ticker and f.side == "buy"]
        return prices[-1] if prices else None


class Rule(Protocol):
    type: ViolationType

    def check(self, ctx: RuleContext) -> list[Violation]: ...


def _ev(**kwargs: object) -> dict[str, str]:
    """Evidence values as strings, so Decimals survive JSON round-trips exactly.

    Float-formatting a Decimal here would make ground-truth evidence subtly
    lossy, and evidence is what a human uses to adjudicate borderline labels.
    """
    return {k: str(v) for k, v in kwargs.items()}
