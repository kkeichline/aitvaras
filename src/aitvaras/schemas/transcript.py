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

from pydantic import BaseModel, ConfigDict

from aitvaras.schemas.decision import AgentTurn, Observation, RuleBook
from aitvaras.schemas.market import PortfolioState
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


class Step(BaseModel):
    """One trading day, start to finish."""

    model_config = ConfigDict(frozen=True)

    step: int
    as_of: dt.date
    observation: Observation
    turn: AgentTurn
    fills: tuple[Fill, ...] = ()
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
    universe: tuple[str, ...]
    steps: tuple[Step, ...]
    score: ScoreCard


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
