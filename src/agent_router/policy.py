from __future__ import annotations

from dataclasses import dataclass

from .types import ExecutionClass, Requirement, Risk, RouteDecision, Task


@dataclass(frozen=True, slots=True)
class RoutingPolicy:
    """Deterministic policy gates before any learned/model-based routing."""

    high_risk_requires_human: bool = False

    def route(self, task: Task) -> RouteDecision:
        requirements = task.requirements

        if task.risk is Risk.HIGH and self.high_risk_requires_human:
            return RouteDecision(
                ExecutionClass.HUMAN_REVIEW,
                "policy requires human review for high-risk tasks",
            )

        if Requirement.EXACT_COMPUTATION in requirements:
            return RouteDecision(
                ExecutionClass.DETERMINISTIC,
                "task requires exact computation",
            )

        needs_deep_reasoning = (
            task.risk is Risk.HIGH
            or Requirement.DEEP_PLANNING in requirements
            or Requirement.HIGH_RELIABILITY in requirements
            or Requirement.LONG_CONTEXT in requirements
        )

        if (
            Requirement.EXTERNAL_DATA in requirements
            and Requirement.SEMANTIC_REASONING not in requirements
            and not needs_deep_reasoning
        ):
            return RouteDecision(
                ExecutionClass.RETRIEVAL,
                "external data can be retrieved without semantic reasoning",
            )

        if needs_deep_reasoning:
            return RouteDecision(
                ExecutionClass.DEEP_REASONING,
                "task requires high-capability reasoning",
            )

        if Requirement.SEMANTIC_REASONING in requirements:
            return RouteDecision(
                ExecutionClass.LIGHT_REASONING,
                "task requires bounded semantic reasoning",
            )

        if Requirement.TOOL_USE in requirements:
            return RouteDecision(
                ExecutionClass.DETERMINISTIC,
                "task can begin with deterministic tool execution",
            )

        return RouteDecision(
            ExecutionClass.DETERMINISTIC,
            "no reasoning requirement declared",
        )


def next_execution_class(current: ExecutionClass) -> ExecutionClass:
    escalation_order = {
        ExecutionClass.DETERMINISTIC: ExecutionClass.LIGHT_REASONING,
        ExecutionClass.RETRIEVAL: ExecutionClass.LIGHT_REASONING,
        ExecutionClass.LIGHT_REASONING: ExecutionClass.DEEP_REASONING,
        ExecutionClass.DEEP_REASONING: ExecutionClass.HUMAN_REVIEW,
        ExecutionClass.HUMAN_REVIEW: ExecutionClass.HUMAN_REVIEW,
    }
    return escalation_order[current]
