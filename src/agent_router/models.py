from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from .pricing import PricingProfile
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
    pricing: PricingProfile | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError("reliability must be between 0 and 1")
        if self.input_cost_per_million < 0 or self.output_cost_per_million < 0:
            raise ValueError("model costs must be non-negative")
        if self.context_window is not None and self.context_window < 1:
            raise ValueError("context_window must be positive")

    @property
    def pricing_profile(self) -> PricingProfile:
        return self.pricing or PricingProfile(
            standard_input=self.input_cost_per_million,
            standard_output=self.output_cost_per_million,
        )

    def supports(self, task: Task, execution_class: ExecutionClass) -> bool:
        if execution_class not in self.execution_classes:
            return False
        if not task.requirements.issubset(self.capabilities):
            return False
        required_context = _required_context_tokens(task)
        return not (
            required_context is not None
            and self.context_window is not None
            and required_context > self.context_window
        )

    @property
    def list_price_per_million(self) -> float:
        """Flat list price used to break estimated-cost ties when token estimates are absent."""
        pricing = self.pricing_profile
        return pricing.standard_input + pricing.standard_output

    def estimate_cost(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
        cache_write_tokens: int = 0,
        batch: bool = False,
    ) -> float:
        return self.pricing_profile.estimate(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            cache_write_tokens=cache_write_tokens,
            batch=batch,
        )


def _required_context_tokens(task: Task) -> int | None:
    """Context size a task needs, from an explicit hint or the estimated token budget.

    Prefers an explicit ``context_tokens`` hint; otherwise falls back to the same
    ``estimated_input_tokens``/``estimated_output_tokens`` the routing pipeline populates,
    so the context-window guard is enforced on the real invocation path and not only when
    a caller happens to set ``context_tokens``.
    """
    explicit = task.metadata.get("context_tokens")
    if isinstance(explicit, int):
        return explicit
    estimated_input = task.metadata.get("estimated_input_tokens")
    estimated_output = task.metadata.get("estimated_output_tokens")
    total = 0
    seen = False
    for value in (estimated_input, estimated_output):
        if isinstance(value, int):
            total += value
            seen = True
    return total if seen else None


class NoEligibleModel(RuntimeError):
    pass


class ModelRegistry:
    def __init__(self, profiles: Iterable[ModelProfile] = ()) -> None:
        self._profiles: dict[str, ModelProfile] = {profile.name: profile for profile in profiles}
        self._aliases: dict[str, str] = {}

    def register(self, profile: ModelProfile) -> None:
        self._profiles[profile.name] = profile

    def register_alias(self, alias: str, target: str) -> None:
        if alias in self._profiles:
            raise ValueError(f"alias {alias!r} conflicts with a model name")
        if target not in self._profiles:
            raise KeyError(target)
        self._aliases[alias] = target

    def get(self, name: str) -> ModelProfile:
        return self._profiles[self._aliases.get(name, name)]

    def eligible(self, task: Task, execution_class: ExecutionClass) -> list[ModelProfile]:
        return [
            profile
            for profile in self._profiles.values()
            if profile.supports(task, execution_class)
        ]

    def ranked(
        self,
        task: Task,
        execution_class: ExecutionClass,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        min_reliability: float = 0.0,
        max_estimated_cost: float | None = None,
    ) -> list[ModelProfile]:
        candidates = [
            profile
            for profile in self.eligible(task, execution_class)
            if profile.reliability >= min_reliability
            and (
                max_estimated_cost is None
                or profile.estimate_cost(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
                <= max_estimated_cost
            )
        ]
        return sorted(
            candidates,
            key=lambda profile: (
                profile.estimate_cost(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ),
                profile.list_price_per_million,
                -profile.reliability,
                profile.name,
            ),
        )

    def select(
        self,
        task: Task,
        execution_class: ExecutionClass,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        min_reliability: float = 0.0,
        max_estimated_cost: float | None = None,
    ) -> ModelProfile:
        candidates = self.ranked(
            task,
            execution_class,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            min_reliability=min_reliability,
            max_estimated_cost=max_estimated_cost,
        )
        if not candidates:
            raise NoEligibleModel(
                f"no eligible model for execution class {execution_class.value}"
            )
        return candidates[0]
