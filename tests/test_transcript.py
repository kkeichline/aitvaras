"""The deduplication contract.

History is stored once per run and each step records the lookback it used, so
``observation_at`` has to rebuild exactly what the agent saw. If it drifts, the
monitor reads a different run than the one that was labelled, and every phase 3
number is measured against the wrong thing -- silently, because both sides would
still be internally consistent.

These tests also guard the size win. It was a 22x reduction won by changing the
schema; without a budget assertion, someone re-embedding an Observation "for
convenience" would give it all back with no test going red.
"""

from __future__ import annotations

import json
from decimal import Decimal

from aitvaras.agent.scripted import CallbackAgent, NoopAgent
from aitvaras.env.engine import Engine
from aitvaras.env.fills import SimulatedVenue
from aitvaras.env.replay import ReplayDataSource
from aitvaras.schemas.decision import Decision, Order, RuleBook
from aitvaras.schemas.transcript import RunConditions, RunTranscript
from conftest import PRICES

D = Decimal
CONDITIONS = RunConditions(agent_model="scripted", agent_prompt_id="none", seed=0)
LOOKBACK = 60


def run(start: int = 100, n_steps: int = 30, lookback: int = LOOKBACK):
    source = ReplayDataSource(PRICES)
    engine = Engine(
        source,
        SimulatedVenue(source, D("100000")),
        RuleBook(),
        lookback=lookback,
    )
    art = engine.run(NoopAgent(), CONDITIONS, start=start, n_steps=n_steps)
    return source, art.transcript


# --------------------------------------------------------------------------
# Reconstruction must be exact
# --------------------------------------------------------------------------


def test_reconstruction_matches_the_live_source_on_every_step():
    """The load-bearing test for the whole deduplication."""
    source, t = run()
    for i, step in enumerate(t.steps):
        expected = source.observe(step.as_of, LOOKBACK)
        actual = t.observation_at(i).history
        assert actual == expected, f"step {i} ({step.as_of}) reconstructed differently"


def test_reconstruction_is_exact_at_the_start_of_history():
    """Early steps saw a *short* window -- fewer bars than lookback, because
    history had not accumulated yet. Slicing the last N out of the run-level
    union must reproduce that shortfall rather than back-filling from bars the
    agent never saw."""
    source, t = run(start=0, n_steps=10)
    for i, step in enumerate(t.steps):
        expected = source.observe(step.as_of, LOOKBACK)
        assert t.observation_at(i).history == expected
        assert len(t.observation_at(i).history["AAPL"].bars) == i + 1


def test_reconstruction_survives_a_json_round_trip():
    """Persistence must be lossless. This is the path a monitor will take:
    read the file from disk, rebuild, reason about it."""
    source, t = run(n_steps=10)
    revived = RunTranscript.model_validate_json(t.model_dump_json())
    for i in range(len(t.steps)):
        assert revived.observation_at(i).history == t.observation_at(i).history
        assert revived.observation_at(i) == t.observation_at(i)


def test_reconstruction_preserves_the_agents_own_state():
    """History is not the only thing rebuilt -- cooldowns, the halt flag and the
    loss budget must come back too, or the monitor cannot tell whether the agent
    knew it was constrained."""
    source = ReplayDataSource(PRICES)
    engine = Engine(source, SimulatedVenue(source, D("100000"), slippage_bps=D("1000")), RuleBook())
    art = engine.run(
        CallbackAgent(
            lambda o: (
                Decision(orders=(Order(ticker="AAPL", side="buy", quantity=10),))
                if o.step == 0
                else Decision(orders=(Order(ticker="AAPL", side="sell", quantity=10),))
                if o.step == 1
                else Decision(orders=())
            )
        ),
        CONDITIONS,
        start=0,
        n_steps=8,
    )
    obs = art.transcript.observation_at(4)
    assert "AAPL" in obs.cooldown_until
    assert obs.rules == art.transcript.rules
    assert obs.universe == art.transcript.universe


# --------------------------------------------------------------------------
# The size win, guarded
# --------------------------------------------------------------------------


def test_history_is_stored_exactly_once():
    _, t = run(n_steps=30)
    stored = sum(len(w.bars) for w in t.history.values())
    dates = {b.date for w in t.history.values() for b in w.bars}
    assert stored == len(dates) * len(t.universe), "history contains duplicate bars"


def test_no_step_carries_its_own_price_history():
    """The regression guard. Re-embedding an Observation in Step would restore
    the 30x duplication and break phase 3 again."""
    _, t = run(n_steps=5)
    blob = json.loads(t.model_dump_json())
    for step in blob["steps"]:
        assert "history" not in step
        assert "observation" not in step
        assert "rules" not in step, "rulebook is run-level; 60 copies bought nothing"


def test_a_sixty_step_transcript_stays_small_enough_to_read():
    """A budget, not a benchmark.

    Before deduplication this was 4.53 MB / ~1.13M tokens, which no monitor can
    read in one context. 500 KB leaves generous headroom over the ~200 KB this
    currently produces while still failing loudly if per-step history returns.
    """
    _, t = run(start=100, n_steps=60, lookback=60)
    size = len(t.model_dump_json())
    assert size < 500_000, f"transcript grew to {size / 1e6:.2f} MB"


def test_the_decision_content_is_a_small_fraction_of_the_whole():
    """What a monitor reasons about -- orders, journal, fills, portfolio deltas
    -- should dominate the *meaning* of the transcript even though prices
    dominate its bytes. Recorded as a measurement, not an aspiration: it is what
    makes a monitor-facing rendering that omits history a sensible thing to try.
    """
    _, t = run(start=100, n_steps=60)
    whole = len(t.model_dump_json())
    history_only = len(
        json.dumps(
            {k: v.model_dump(mode="json") for k, v in t.history.items()},
            separators=(",", ":"),
        )
    )
    assert history_only / whole > 0.5, "expected prices to still dominate the bytes"
