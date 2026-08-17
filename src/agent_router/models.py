from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .types import ExecutionClass, Requirement, Task


@dataclass(frozen=True, slots=True)
class ModelProfile:
    name: str
    provider: str
    execution_classes: set[ExecutionClass]
    capabilities: set[Requirement] = field(default_factory=set)
    context_window: int | None = None
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    reliability: float = 1.0
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError("reliability must be between 0 and 1")
        if self.input_cost_per_million < 0 or self.output_cost_per_million < 0:
            raise ValueError("model costs must be non-negative")
        if self.context_window is not None and self.context_window < 1:
            raise ValueError("context_window must be positive")

    def supports(self, task: Task, execution_class: ExecutionClass) -> bool:
        if execution_class not in self.execution_classes:
            return False
        if not task.requirements.issubset(self.capabilities):
            return False
        required_context = task.metadata.get("context_tokens")
        if (
            isinstance(required_context, int)
            and self.context_window is not None
            and required_context > self.context_window
        ):
            return False
        return True

    def estimate_cost(self, *, input_tokens: int = 0, output_tokens: int = 0) -> float:
        return (
            input_tokens * self.input_cost_per_million
            + output_tokens * self.output_cost_per_million
        ) / 1_000_000


class NoEligibleModel(RuntimeError):
    pass


class ModelRegistry:
    def __init__(self, profiles: Iterable[ModelProfile] = ()) -> None:
        self._profiles: dict[str, ModelProfile] = {profile.name: profile for profile in profiles}

    def register(self, profile: ModelProfile) -> None:
        self._profiles[profile.name] = profile

    def get(self, name: str) -> ModelProfile:
        return self._profiles[name]

    def eligible(self, task: Task, execution_class: ExecutionClass) -> list[ModelProfile]:
        return [
            profile
            for profile in self._profiles.values()
            if profile.supports(task, execution_class)
        ]

    def select(
        self,
        task: Task,
        execution_class: ExecutionClass,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        min_reliability: float = 0.0,
    ) -> ModelProfile:
        candidates = [
            profile
            for profile in self.eligible(task, execution_class)
            if profile.reliability >= min_reliability
        ]
        if not candidates:
            raise NoEligibleModel(
                f"no eligible model for execution class {execution_class.value}"
            )

        return min(
            candidates,
            key=lambda profile: (
                profile.estimate_cost(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ),
                -profile.reliability,
                profile.name,
            ),
        )
