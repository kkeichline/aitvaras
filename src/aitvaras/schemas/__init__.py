"""Typed contract shared by every other package.

``schemas`` imports nothing from its sibling packages. That is what keeps the
agent/environment boundary real rather than aspirational: everything depends on
schemas, schemas depends on nobody, so swapping the data or execution layer
cannot ripple into the agent, the rules, the journal, or the monitor.
"""

from aitvaras.schemas.decision import (
    AgentTurn,
    Decision,
    LookaheadError,
    Observation,
    Order,
    RuleBook,
    Side,
    ToolCall,
    Usage,
)
from aitvaras.schemas.market import Bar, PortfolioState, Position, PriceWindow
from aitvaras.schemas.transcript import (
    ExecutionResult,
    Fill,
    OrderRejection,
    RejectionReason,
    RunArtifacts,
    RunConditions,
    RunLabels,
    RunTranscript,
    ScoreCard,
    Step,
)
from aitvaras.schemas.violation import (
    MECHANICAL,
    Detectability,
    EnforcementMode,
    Violation,
    ViolationType,
)

__all__ = [
    "MECHANICAL",
    "RejectionReason",
    "OrderRejection",
    "ExecutionResult",
    "AgentTurn",
    "Bar",
    "Decision",
    "Detectability",
    "EnforcementMode",
    "Fill",
    "LookaheadError",
    "Observation",
    "Order",
    "PortfolioState",
    "Position",
    "PriceWindow",
    "RuleBook",
    "RunArtifacts",
    "RunConditions",
    "RunLabels",
    "RunTranscript",
    "ScoreCard",
    "Side",
    "Step",
    "ToolCall",
    "Usage",
    "Violation",
    "ViolationType",
]
