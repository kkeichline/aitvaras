"""The two artefacts a run produces, and the wall between them.

    runs/<run_id>/transcript.json   -> RunTranscript. The monitor may read this.
    runs/<run_id>/labels.json       -> RunLabels.     The monitor must never read this.

That separation is the single most important structural decision in the
project. If one engine-derived fact leaks across it -- a violation count, a
tell-tale field ordering, a suspiciously specific number -- then phase 3
measures leakage instead of detection, and the whole benchmark is void. The
types are split so the leak has to be deliberate rather than accidental:
there is no object holding both.

``RunTranscript`` is also designed to be *self-contained and replayable*. The
agent is the expensive axis; monitors are cheap and numerous. Generate a
transcript once, then run every monitor condition -- model, prompt paraphrase,
rules-shown-or-not -- against the frozen set. That turns a multiplicative cost
grid into an additive one, which is what makes phase 3 affordable on evenings
and weekends.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from aitvaras.schemas.decision import (
    FORWARD_LOOKING,
    AgentTurn,
    Observation,
    RuleBook,
)
from aitvaras.schemas.market import PortfolioState, PriceWindow
from aitvaras.schemas.violation import Violation


class Fill(BaseModel):
    """An executed trade. Phase 1 fills at the next day's open."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    side: str
    quantity: int
    price: Decimal
    as_of: dt.date
    realized_pnl: Decimal = Decimal("0")
    """Non-zero only on sells. Drives the cooldown rule."""


class RejectionReason(StrEnum):
    INSUFFICIENT_CASH = "insufficient_cash"
    INSUFFICIENT_SHARES = "insufficient_shares"
    NO_PRICE = "no_price"
    NOT_IN_UNIVERSE = "not_in_universe"


class OrderRejection(BaseModel):
    """An order the venue would not execute.

    Rejections are an *execution* fact, not a rule violation, and the two must
    not be confused. The venue rejects a buy the agent cannot afford; the rule
    oracle separately labels that same order a blocklist violation if it was for
    a forbidden ticker. An order can be both, either, or neither.

    They belong in the transcript because repeatedly ordering what you cannot
    afford is behaviour worth seeing -- and because a monitor that never learns
    an order failed would misread the portfolio deltas that follow.
    """

    model_config = ConfigDict(frozen=True)

    order_index: int
    ticker: str
    side: str
    quantity: int
    reason: RejectionReason
    detail: str = ""


class ExecutionResult(BaseModel):
    """What the venue did with one step's orders."""

    model_config = ConfigDict(frozen=True)

    fills: tuple[Fill, ...] = ()
    rejections: tuple[OrderRejection, ...] = ()


class Step(BaseModel):
    """One trading day, start to finish -- as *persisted*.

    Note what is absent: the price history the agent saw. It is not here.

    Embedding the full ``Observation`` per step was the original design and it
    was measurably wrong. A 60-step run with a 60-bar lookback over 12 tickers
    serialized 43,200 bars of which only 1,428 were distinct: a 30x duplication
    that made a single transcript 4.53 MB / ~1.13M tokens, of which 99% was
    repeated price tables. No monitor can read that, and phase 3 would have been
    impossible before it started. The decision content a monitor actually
    reasons about -- orders, rationales, journal, fills, portfolio deltas -- was
    50 KB of the 4.53 MB.

    So history lives once on ``RunTranscript.history`` and each step records the
    ``lookback`` it used. ``RunTranscript.observation_at`` reconstructs exactly
    what the agent saw, and a test asserts that reconstruction is byte-equal to
    the live source. Nothing is lost; 22x is saved.

    The other benefit is structural rather than economic. With history held
    separately, rendering a transcript *for a monitor* becomes a real seam:
    full history, truncated history, or none at all, chosen per monitor
    condition. That seam is what open question #6 -- does the monitor get the
    rules and the prices? -- actually requires, and embedding everything per
    step would have made it unreachable.
    """

    model_config = ConfigDict(frozen=True)

    step: int
    as_of: dt.date
    """The *decision* date. ``portfolio_after`` is dated the following day,
    because that is when the orders filled."""

    lookback: int
    """How many bars of history the agent was shown. With ``as_of``, this is
    enough to reconstruct the observation from run-level history."""

    portfolio_before: PortfolioState
    cooldown_until: Annotated[dict[str, dt.date], FORWARD_LOOKING] = Field(default_factory=dict)
    remaining_daily_loss_budget: Decimal = Decimal("0")
    halted: bool = False

    turn: AgentTurn
    fills: tuple[Fill, ...] = ()
    rejections: tuple[OrderRejection, ...] = ()
    portfolio_after: PortfolioState


class ScoreCard(BaseModel):
    """Performance, not compliance. Deliberately separate from labels.

    Sharpe is the primary because it is what makes 'churn to smooth the ratio'
    an exploitable loophole -- an agent under pressure to raise Sharpe has a
    concrete dishonest move available, which is exactly what phase 2 needs.
    """

    model_config = ConfigDict(frozen=True)

    final_equity: Decimal
    total_return: Decimal
    sharpe: float | None = None
    """None when there are too few return observations to estimate one."""
    max_drawdown: Decimal = Decimal("0")
    n_trading_days: int = 0
    n_orders: int = 0
    turnover: Decimal = Decimal("0")
    """Traded notional over average equity. A churn detector, reported so the
    Sharpe loophole is visible in the scorecard rather than hidden in it."""


class RunConditions(BaseModel):
    """The experimental cell this run belongs to.

    ``observed`` is an independent variable, not a detail. An agent that has
    never been told anyone reads its journal has no motive to deceive, so its
    'dirty' runs are merely sloppy. An agent that knows it is watched is
    evading. Both are legitimate experiments; conflating them is not.
    """

    model_config = ConfigDict(frozen=True)

    agent_model: str
    agent_prompt_id: str
    pressure: str = "neutral"
    """Which incentive-pressure prompt variant was used (phase 2)."""
    observed: bool = False
    """Whether the agent was told its journal is reviewed."""
    seed: int = 0
    """Controls data slicing and environment setup only. It does NOT control
    model sampling -- LLMs are not seedable. Run-to-run model variance is a
    separate, uncontrolled source, and the methodology doc says so out loud."""


class RunTranscript(BaseModel):
    """Everything the monitor is allowed to see. No ground truth here."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    created_at: dt.datetime
    config_hash: str
    conditions: RunConditions
    rules: RuleBook
    """Run-level, not per-step. The agent was shown the same rulebook every
    day, so storing it 60 times bought nothing."""

    universe: tuple[str, ...]
    history: dict[str, PriceWindow] = Field(default_factory=dict)
    """Every bar any step of this run saw, stored exactly once.

    The union across the run, which makes ``observation_at`` exact: a step's
    window is 'the last ``lookback`` bars dated at or before ``as_of``', and
    slicing that out of the union gives back precisely what the source gave.
    """

    steps: tuple[Step, ...]
    score: ScoreCard

    def observation_at(self, step: int) -> Observation:
        """Rebuild what the agent saw on ``step``, from run-level history.

        This is the inverse of the deduplication, and it is load-bearing: if it
        drifted from what the engine actually showed the agent, the monitor
        would be reading a different run than the one that was labelled.
        ``test_transcript.py`` pins it against the live data source.
        """
        record = self.steps[step]
        window: dict[str, PriceWindow] = {}
        for ticker, full in self.history.items():
            bars = tuple(b for b in full.bars if b.date <= record.as_of)
            if bars:
                window[ticker] = PriceWindow(ticker=ticker, bars=bars[-record.lookback :])

        return Observation(
            step=record.step,
            as_of=record.as_of,
            universe=self.universe,
            history=window,
            portfolio=record.portfolio_before,
            rules=self.rules,
            cooldown_until=dict(record.cooldown_until),
            remaining_daily_loss_budget=record.remaining_daily_loss_budget,
            halted=record.halted,
        )


class RunLabels(BaseModel):
    """Ground truth for one run. Paired to the transcript only by ``run_id``."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    violations: tuple[Violation, ...] = ()
    intent_labels_complete: bool = False
    """False until a human has passed over the run for intent-class violations.
    Absence of a JOURNAL_MISREPRESENTATION label means 'not yet looked at',
    not 'clean', until this flips -- treating unreviewed as clean would inflate
    the monitor's false-positive rate with real catches."""

    notes: str = ""

    @property
    def is_dirty(self) -> bool:
        return len(self.violations) > 0

    def of_type(self, *types: object) -> tuple[Violation, ...]:
        return tuple(v for v in self.violations if v.type in types)


class RunArtifacts(BaseModel):
    """Convenience pair for code that legitimately holds both -- the runner and
    the scorer. Never serialized as one file, never passed to a monitor."""

    model_config = ConfigDict(frozen=True)

    transcript: RunTranscript
    labels: RunLabels
