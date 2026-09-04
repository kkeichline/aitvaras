"""Deterministic agents with no model behind them.

These are not toys. They are how the environment gets tested without spending
money or inheriting model nondeterminism, and how phase 2 will construct
*known-dirty* runs whose violations are placed exactly where we want them --
a scripted agent that buys a blocklisted ticker on step 7 gives us a run whose
ground truth we know independently of the oracle, which is the only way to test
that the oracle itself is right.
"""

from __future__ import annotations

from collections.abc import Callable

from aitvaras.schemas.decision import AgentTurn, Decision, Observation


class NoopAgent:
    """Holds forever. The clean baseline.

    Useful precisely because it should produce zero violations: if a run of this
    agent ever gets a label, the oracle has a false positive, and every catch
    rate measured against it would be inflated.
    """

    def decide(self, obs: Observation) -> AgentTurn:
        return AgentTurn(
            raw_response='{"orders": [], "journal": "hold"}',
            decision=Decision(orders=(), journal="No action taken."),
        )


class ScriptedAgent:
    """Replays a fixed list of decisions, one per step, then holds."""

    def __init__(self, script: list[Decision]) -> None:
        self._script = script

    def decide(self, obs: Observation) -> AgentTurn:
        decision = (
            self._script[obs.step]
            if obs.step < len(self._script)
            else Decision(orders=(), journal="Script exhausted; holding.")
        )
        return AgentTurn(raw_response=decision.model_dump_json(), decision=decision)


class CallbackAgent:
    """Builds each decision from the observation via a plain function.

    For tests that need to react to state -- 'buy whatever is cheapest', 'sell
    everything on step 5' -- without hand-writing a script of the right length.
    """

    def __init__(self, fn: Callable[[Observation], Decision]) -> None:
        self._fn = fn

    def decide(self, obs: Observation) -> AgentTurn:
        decision = self._fn(obs)
        return AgentTurn(raw_response=decision.model_dump_json(), decision=decision)
