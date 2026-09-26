"""The LLM trading agent.

Stateful by design, which is the one thing worth understanding here. The
``Agent`` protocol is a single ``decide(obs)`` call, and each call is still
exactly one model request returning one day's decision -- so this is a single-shot
agent, as chosen in phase 1. But the agent keeps the *conversation* internally
and appends to it, rather than rebuilding a fresh prompt each step.

Why that matters, in order of importance:

1. **The agent sees its own prior journals.** To misrepresent a trade
   consistently, an agent has to maintain a story across days. Rebuilding the
   prompt each step would give it amnesia, and journal misrepresentation --
   arguably the most interesting violation class -- would become nearly
   impossible to observe.
2. **Cooldown memory comes free.** The rule requires remembering a realised loss
   from several days ago. With the conversation retained, that is simply in
   context.
3. **The prefix is append-only, so provider caching works.** A prompt rebuilt
   each step with a sliding window changes at byte one and never caches. This
   shape reads the whole prior conversation at ~0.1x and pays full price only for
   the new day.

API failures are allowed to propagate. A network error must not be silently
recorded as "the agent decided to hold" -- that would be a fabricated decision
sitting in a labelled dataset. A failed run should fail.
"""

from __future__ import annotations

import datetime as dt

from aitvaras.agent.parsing import parse_decision
from aitvaras.agent.prompts.render import render_step
from aitvaras.agent.prompts.system import system_for
from aitvaras.llm.client import LLMClient, Message
from aitvaras.schemas.decision import AgentTurn, Observation, RuleBook


class LLMAgent:
    """A model-backed agent with an append-only conversation."""

    def __init__(
        self,
        client: LLMClient,
        *,
        model: str,
        prompt_id: str,
        rules: RuleBook,
        max_tokens: int = 2048,
        temperature: float | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._prompt_id = prompt_id
        self._system = system_for(prompt_id, rules)
        self._max_tokens = max_tokens
        self._temperature = temperature

        self._messages: list[Message] = []
        self._last_rendered: dt.date | None = None

    @property
    def system_prompt(self) -> str:
        return self._system

    @property
    def conversation(self) -> tuple[Message, ...]:
        return tuple(self._messages)

    def decide(self, obs: Observation) -> AgentTurn:
        user_text = render_step(obs, since=self._last_rendered)

        # Move the cache breakpoint to the newest message and clear the old one.
        # Anthropic allows at most four, and only the frontier one is useful here
        # since the conversation only ever grows.
        self._messages = [m.model_copy(update={"cache": False}) for m in self._messages]
        self._messages.append(Message(role="user", content=user_text, cache=True))

        response = self._client.complete(
            self._messages,
            model=self._model,
            system=self._system,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )

        # The assistant's raw text goes back into the conversation verbatim, even
        # when it failed to parse. Cleaning it up would hide from the agent what
        # it actually said, and an agent that cannot see its own prior output
        # cannot be held to a consistent story.
        self._messages.append(Message(role="assistant", content=response.text))
        self._last_rendered = obs.as_of

        decision, error = parse_decision(response.text)
        return AgentTurn(
            raw_response=response.text,
            decision=decision,
            parse_error=error,
            usage=response.usage,
        )
