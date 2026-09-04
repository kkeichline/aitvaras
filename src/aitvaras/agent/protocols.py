"""What the engine requires of an agent.

One method. Everything the agent knows arrives in the ``Observation``; anything
it wants to do leaves in the ``AgentTurn``. There is no other channel -- no
shared object, no callback, no handle on the venue -- which is what makes the
lookahead guarantee checkable rather than merely intended.

``tool_calls`` already exists on ``AgentTurn`` and is empty in phase 1. When the
agent gains tools in phase 4, this signature does not change and transcripts
generated now stay replayable against later monitors.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from aitvaras.schemas.decision import AgentTurn, Observation


@runtime_checkable
class Agent(Protocol):
    def decide(self, obs: Observation) -> AgentTurn: ...
