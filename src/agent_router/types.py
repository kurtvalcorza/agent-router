from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ExecutionClass(StrEnum):
    DETERMINISTIC = "deterministic"
    RETRIEVAL = "retrieval"
    LIGHT_REASONING = "light_reasoning"
    DEEP_REASONING = "deep_reasoning"
    HUMAN_REVIEW = "human_review"


class Requirement(StrEnum):
    EXACT_COMPUTATION = "exact_computation"
    EXTERNAL_DATA = "external_data"
    SEMANTIC_REASONING = "semantic_reasoning"
    DEEP_PLANNING = "deep_planning"
    TOOL_USE = "tool_use"
    LONG_CONTEXT = "long_context"
    HIGH_RELIABILITY = "high_reliability"


class Risk(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class VerificationStatus(StrEnum):
    PASS = "pass"
    RETRY = "retry"
    ESCALATE = "escalate"


@dataclass(frozen=True, slots=True)
class Task:
    kind: str
    payload: dict[str, Any]
    requirements: set[Requirement] = field(default_factory=set)
    risk: Risk = Risk.LOW
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # ``frozen=True`` blocks attribute rebinding but not ``task.requirements.add(...)``;
        # store the routing-critical set as a frozenset so a routed task cannot be mutated
        # underneath the policy after construction.
        if not isinstance(self.requirements, frozenset):
            object.__setattr__(self, "requirements", frozenset(self.requirements))


@dataclass(frozen=True, slots=True)
class RouteDecision:
    execution_class: ExecutionClass
    reason: str
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(slots=True)
class Budget:
    max_cost_usd: float | None = None
    max_model_calls: int | None = None
    max_tool_calls: int | None = None
    max_latency_seconds: float | None = None
    cost_usd: float = 0.0
    model_calls: int = 0
    tool_calls: int = 0

    def consume(
        self,
        *,
        cost_usd: float = 0.0,
        model_calls: int = 0,
        tool_calls: int = 0,
    ) -> None:
        next_cost = self.cost_usd + cost_usd
        next_model_calls = self.model_calls + model_calls
        next_tool_calls = self.tool_calls + tool_calls

        if self.max_cost_usd is not None and next_cost > self.max_cost_usd:
            raise BudgetExceeded("cost budget exceeded")
        if self.max_model_calls is not None and next_model_calls > self.max_model_calls:
            raise BudgetExceeded("model-call budget exceeded")
        if self.max_tool_calls is not None and next_tool_calls > self.max_tool_calls:
            raise BudgetExceeded("tool-call budget exceeded")

        self.cost_usd = next_cost
        self.model_calls = next_model_calls
        self.tool_calls = next_tool_calls


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    budget: Budget
    attempt: int
    decision: RouteDecision


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    output: Any
    cost_usd: float = 0.0
    model_calls: int = 0
    tool_calls: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Verification:
    status: VerificationStatus
    reason: str = ""


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    task_kind: str
    execution_class: ExecutionClass
    route_reason: str
    attempt: int
    verification: VerificationStatus
    cost_usd: float
    model_calls: int
    tool_calls: int
