"""The step loop. Where the environment, the agent, and the oracle meet.

The engine owns the two pieces of cross-step state that no single component can
compute on its own:

* **the daily-loss halt** -- whether yesterday's move breached the limit, which
  makes today buy-restricted;
* **cooldowns** -- when each ticker becomes buyable again after a realised loss.

Both are derived here and *handed to the agent in its observation*. That is a
deliberate choice with teeth: withholding them would let us score violations the
agent had no way to avoid, and we would be measuring its bookkeeping rather than
its honesty.

The timeline for one step, which is worth reading once carefully:

    day[t]      observation built and marked at day[t] close
                agent decides
    day[t+1]    orders fill at the open
                book re-marked at day[t+1] close  -> portfolio_after
                rules evaluated, halt and cooldowns updated for the next step

So ``Step.as_of`` is the *decision* date while ``Step.portfolio_after`` is dated
the following day. The asymmetry is real, not a bug: it is what keeps the agent
from ever trading on a price it has already seen.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import uuid
from decimal import Decimal

from aitvaras.agent.protocols import Agent
from aitvaras.env.fills import SimulatedVenue
from aitvaras.env.replay import ReplayDataSource
from aitvaras.rules.base import RuleContext
from aitvaras.rules.registry import DEFAULT_RULES, Rule, evaluate
from aitvaras.schemas.decision import Observation, RuleBook
from aitvaras.schemas.market import Bar, PriceWindow
from aitvaras.schemas.transcript import (
    RunArtifacts,
    RunConditions,
    RunLabels,
    RunTranscript,
    Step,
)
from aitvaras.schemas.violation import Violation
from aitvaras.scoring import score_run

ZERO = Decimal("0")


class Engine:
    """Runs one agent over one slice of the calendar and emits both artefacts."""

    def __init__(
        self,
        source: ReplayDataSource,
        venue: SimulatedVenue,
        rulebook: RuleBook,
        holdings: dict[str, dict[str, Decimal]] | None = None,
        rules: tuple[Rule, ...] = DEFAULT_RULES,
        warmup: int = 60,
    ) -> None:
        self._source = source
        self._venue = venue
        self._rulebook = rulebook
        self._holdings = holdings or {}
        self._rules = rules
        self._warmup = warmup

    def run(
        self,
        agent: Agent,
        conditions: RunConditions,
        start: int = 0,
        n_steps: int = 60,
    ) -> RunArtifacts:
        calendar = self._source.calendar()

        # Stop one short of the end: the final day has no next open to fill
        # against, so a step there could decide but never execute.
        last_decidable = len(calendar) - 1
        if start >= last_decidable:
            raise ValueError(f"start={start} leaves no decidable days")
        end = min(start + n_steps, last_decidable)

        starting_equity = self._venue.portfolio(calendar[start]).equity

        steps: list[Step] = []
        violations: list[Violation] = []
        cooldown_until: dict[str, dt.date] = {}
        halted = False

        # Every distinct bar any step sees, accumulated once. Keyed by date so
        # the 30x duplication that came from embedding history per step collapses
        # to the union. See the Step docstring for the measurements.
        seen_bars: dict[str, dict[dt.date, Bar]] = {}

        for i, index in enumerate(range(start, end)):
            day = calendar[index]
            fill_day = calendar[index + 1]

            self._venue.book.start_day()
            before = self._venue.portfolio(day)

            # The window is CUMULATIVE: warmup bars before the run started, plus
            # every day since. Two reasons, and neither is about the prompt.
            #
            # Fairness: the cooldown rule requires remembering a realised loss
            # from several days ago. A window that slid that loss out of view
            # would score the agent for forgetting something we hid from it,
            # measuring our bookkeeping rather than its honesty.
            #
            # Fidelity: the agent's context grows over a run (a sliding prompt
            # would change its prefix every step and defeat provider-side
            # caching entirely). If the engine kept a fixed window while the
            # agent saw a growing one, observation_at() would reconstruct a
            # narrower context than the agent actually had, and the monitor
            # would be reading a run that never happened.
            #
            # This costs nothing in the transcript: history is stored once at
            # run level, so a cumulative window is the same bytes as a fixed one.
            lookback = self._warmup + i

            obs = Observation(
                step=i,
                as_of=day,
                universe=self._source.universe(),
                history=self._source.observe(day, lookback),
                portfolio=before,
                rules=self._rulebook,
                cooldown_until=dict(cooldown_until),
                # How much equity may be lost before the halt trips. Stated so
                # the agent can comply rather than having to reverse-engineer it.
                remaining_daily_loss_budget=(before.equity * self._rulebook.max_daily_loss_pct),
                halted=halted,
            )
            # Cheap insurance, run on every step rather than in tests only.
            # The cost is a walk over ~700 bars; the thing it prevents is an
            # entire benchmark quietly measuring nothing.
            obs.assert_no_lookahead()

            for ticker, window in obs.history.items():
                bucket = seen_bars.setdefault(ticker, {})
                for bar in window.bars:
                    bucket[bar.date] = bar

            turn = agent.decide(obs)
            result = self._venue.submit(turn.orders, day)
            after = self._venue.portfolio(fill_day)

            ctx = RuleContext(
                step=i,
                as_of=day,
                rules=self._rulebook,
                orders=turn.orders,
                fills=result.fills,
                portfolio_before=before,
                portfolio_after=after,
                halted=halted,
                cooldown_until=dict(cooldown_until),
                holdings=self._holdings,
            )
            violations.extend(evaluate(ctx, self._rules))

            steps.append(
                Step(
                    step=i,
                    as_of=day,
                    lookback=lookback,
                    portfolio_before=before,
                    cooldown_until=dict(cooldown_until),
                    remaining_daily_loss_budget=obs.remaining_daily_loss_budget,
                    halted=halted,
                    turn=turn,
                    fills=result.fills,
                    rejections=result.rejections,
                    portfolio_after=after,
                )
            )

            # --- cross-step state for the next iteration ---

            # A realised loss starts the clock from the day it was realised,
            # counted in trading days so a weekend cannot shorten it.
            for fill in result.fills:
                if fill.realized_pnl < ZERO:
                    release = self._source.next_trading_day(fill_day, self._rulebook.cooldown_days)
                    if release is not None:
                        cooldown_until[fill.ticker] = release

            halted = self._breached(before.equity, after.equity)

        # Generated once: this id is the ONLY thing pairing a transcript with its
        # labels. Calling the generator twice produced two different uuids and
        # silently unpaired every run.
        run_id = conditions_run_id(conditions)

        return RunArtifacts(
            transcript=RunTranscript(
                run_id=run_id,
                created_at=dt.datetime.now(dt.UTC),
                config_hash=self._config_hash(start, end),
                conditions=conditions,
                rules=self._rulebook,
                universe=self._source.universe(),
                history={
                    ticker: PriceWindow(
                        ticker=ticker,
                        bars=tuple(bars[d] for d in sorted(bars)),
                    )
                    for ticker, bars in sorted(seen_bars.items())
                },
                steps=tuple(steps),
                score=score_run(tuple(steps), starting_equity),
            ),
            labels=RunLabels(
                run_id=run_id,
                violations=tuple(violations),
                # Mechanical labels are complete the moment the run ends; intent
                # labels are not, and must not be mistaken for "clean".
                intent_labels_complete=False,
            ),
        )

    def _breached(self, before: Decimal, after: Decimal) -> bool:
        """Did the move from ``before`` to ``after`` breach the daily loss limit?"""
        if before <= ZERO:
            return False
        return (after - before) / before <= -self._rulebook.max_daily_loss_pct

    def _config_hash(self, start: int, end: int) -> str:
        """Fingerprint of everything that would change a run's meaning.

        Two runs with the same config_hash and the same agent output must be
        identical. If they are not, something unrecorded is varying, and every
        cross-seed comparison in phase 3 is suspect.
        """
        payload = json.dumps(
            {
                "rules": self._rulebook.model_dump(mode="json"),
                "universe": list(self._source.universe()),
                "holdings": {
                    k: {c: str(w) for c, w in v.items()} for k, v in sorted(self._holdings.items())
                },
                "warmup": self._warmup,
                "start": start,
                "end": end,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def conditions_run_id(conditions: RunConditions) -> str:
    """Stable-ish id: readable prefix plus a short random suffix.

    Deliberately not a pure hash of the conditions -- repeat runs of the same
    cell are the point of the variance work, so they must not collide.
    """
    stem = f"{conditions.agent_model}-{conditions.agent_prompt_id}-s{conditions.seed}"
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    return f"{stem}-{uuid.uuid4().hex[:8]}"
