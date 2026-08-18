from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .types import Requirement, Risk, Task


class PolicyMode(StrEnum):
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

    def allow_human_review(self, task: Task) -> bool:
        return self.mode is PolicyMode.CRITICAL or task.risk is Risk.HIGH
