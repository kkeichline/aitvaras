"""The default rule set, in one place.

Ordering is stable so that ``labels.json`` is byte-reproducible across runs --
a diff in ground truth should always mean a real change, never a set-iteration
accident.
"""

from __future__ import annotations

from aitvaras.rules.base import Rule, RuleContext
from aitvaras.rules.mechanical import (
    BlocklistRule,
    CooldownRule,
    MaxDailyLossRule,
    PositionSizeCapRule,
)
from aitvaras.schemas.violation import Violation

DEFAULT_RULES: tuple[Rule, ...] = (
    PositionSizeCapRule(),
    MaxDailyLossRule(),
    BlocklistRule(),
    CooldownRule(),
)


def evaluate(ctx: RuleContext, rules: tuple[Rule, ...] = DEFAULT_RULES) -> tuple[Violation, ...]:
    """Run every rule over one step. Rules never short-circuit each other:
    one order can violate several rules at once, and collapsing that to a single
    label would make the per-violation-type breakdown in phase 3 wrong."""
    out: list[Violation] = []
    for rule in rules:
        out.extend(rule.check(ctx))
    return tuple(out)
