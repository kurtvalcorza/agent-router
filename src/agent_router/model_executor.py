from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .models import ModelRegistry
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


class RoutedModelExecutor:
    def __init__(
        self,
        *,
        registry: ModelRegistry,
        invoke: ModelInvoker,
        min_reliability: float = 0.0,
        token_estimator: TokenEstimator = default_token_estimator,
    ) -> None:
        self.registry = registry
        self.invoke = invoke
        self.min_reliability = min_reliability
        self.token_estimator = token_estimator

    def __call__(self, task: Task, context: ExecutionContext) -> ExecutionResult:
        estimated_input, estimated_output = self.token_estimator(task)
        profile = self.registry.select(
            task,
            context.decision.execution_class,
            input_tokens=estimated_input,
            output_tokens=estimated_output,
            min_reliability=self.min_reliability,
        )

        response = self.invoke(profile.provider, profile.name, task)
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
            }
        )

        return ExecutionResult(
            output=response.output,
            cost_usd=cost,
            model_calls=1,
            metadata=metadata,
        )
