from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .types import Budget, Requirement, Risk, Task


class PolicyMode(str, Enum):
    ECONOMY = "economy"
    BALANCED = "balanced"
    QUALITY = "quality"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class AdaptivePolicy:
    mode: PolicyMode = PolicyMode.BALANCED

    def reliability_floor(self, task: Task) -> float:
        base = {
            PolicyMode.ECONOMY: 0.70,
            PolicyMode.BALANCED: 0.82,
            PolicyMode.QUALITY: 0.90,
            PolicyMode.CRITICAL: 0.97,
        }[self.mode]

        if task.risk is Risk.HIGH:
            base = max(base, 0.97)
        elif task.risk is Risk.MEDIUM:
            base = max(base, 0.88)

        if Requirement.HIGH_RELIABILITY in task.requirements:
            base = max(base, 0.95)

        return min(base, 1.0)

    def cost_pressure(self, budget: Budget | None) -> float:
        if budget is None or budget.max_cost_usd is None:
            return 0.0
        if budget.max_cost_usd <= 0.01:
            return 1.0
        if budget.max_cost_usd <= 0.05:
            return 0.7
        if budget.max_cost_usd <= 0.20:
            return 0.4
        return 0.1

    def allow_human_review(self, task: Task) -> bool:
        return self.mode is PolicyMode.CRITICAL or task.risk is Risk.HIGH

    def model_selection_weights(self, budget: Budget | None = None) -> tuple[float, float]:
        """Return (cost_weight, reliability_weight) for model ranking."""
        pressure = self.cost_pressure(budget)
        if self.mode is PolicyMode.ECONOMY:
            return 1.0 + pressure, 0.2
        if self.mode is PolicyMode.BALANCED:
            return 0.8 + pressure * 0.5, 0.6
        if self.mode is PolicyMode.QUALITY:
            return 0.4 + pressure * 0.3, 1.0
        return 0.2 + pressure * 0.2, 1.5
