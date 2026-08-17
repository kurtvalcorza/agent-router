from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .adaptive import AdaptivePolicy
from .models import ModelRegistry, NoEligibleModel
from .types import ExecutionContext, ExecutionResult, Task


@dataclass(frozen=True, slots=True)
class ModelResponse:
    output: object
    input_tokens: int = 0
    output_tokens: int = 0
    metadata: dict[str, object] | None = None


ModelInvoker = Callable[[str, str, Task], ModelResponse]
TokenEstimator = Callable[[Task], tuple[int, int]]


def default_token_estimator(task: Task) -> tuple[int, int]:
    input_tokens = task.metadata.get("estimated_input_tokens", 0)
    output_tokens = task.metadata.get("estimated_output_tokens", 0)
    return (
        input_tokens if isinstance(input_tokens, int) else 0,
        output_tokens if isinstance(output_tokens, int) else 0,
    )


class ModelInvocationFailed(RuntimeError):
    pass


class RoutedModelExecutor:
    def __init__(
        self,
        *,
        registry: ModelRegistry,
        invoke: ModelInvoker,
        min_reliability: float = 0.0,
        adaptive_policy: AdaptivePolicy | None = None,
        token_estimator: TokenEstimator = default_token_estimator,
    ) -> None:
        self.registry = registry
        self.invoke = invoke
        self.min_reliability = min_reliability
        self.adaptive_policy = adaptive_policy
        self.token_estimator = token_estimator

    def __call__(self, task: Task, context: ExecutionContext) -> ExecutionResult:
        estimated_input, estimated_output = self.token_estimator(task)
        reliability_floor = self.min_reliability
        if self.adaptive_policy is not None:
            reliability_floor = max(
                reliability_floor,
                self.adaptive_policy.reliability_floor(task),
            )

        remaining_cost = None
        if context.budget.max_cost_usd is not None:
            remaining_cost = max(
                context.budget.max_cost_usd - context.budget.cost_usd,
                0.0,
            )

        candidates = self.registry.ranked(
            task,
            context.decision.execution_class,
            input_tokens=estimated_input,
            output_tokens=estimated_output,
            min_reliability=reliability_floor,
            max_estimated_cost=remaining_cost,
        )
        if not candidates:
            raise NoEligibleModel(
                f"no eligible model for execution class "
                f"{context.decision.execution_class.value}"
            )

        failures: list[str] = []
        for profile in candidates:
            try:
                response = self.invoke(profile.provider, profile.name, task)
            except Exception as exc:  # provider adapters normalize provider failures
                failures.append(f"{profile.provider}/{profile.name}: {exc}")
                continue

            cost = profile.estimate_cost(
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )
            metadata = dict(response.metadata or {})
            metadata.update(
                {
                    "model": profile.name,
                    "provider": profile.provider,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "reliability_floor": reliability_floor,
                    "fallback_failures": tuple(failures),
                }
            )

            return ExecutionResult(
                output=response.output,
                cost_usd=cost,
                model_calls=len(failures) + 1,
                metadata=metadata,
            )

        raise ModelInvocationFailed("all eligible model invocations failed: " + "; ".join(failures))
