from __future__ import annotations

from dataclasses import dataclass

from .adaptive import AdaptivePolicy
from .empirical import EmpiricalSelector
from .model_executor import (
    ModelInvocationFailed,
    ModelInvoker,
    TokenEstimator,
    UnknownProvider,
    _remaining_model_calls,
    default_token_estimator,
)
from .types import BudgetExceeded, ExecutionContext, ExecutionResult, Task


@dataclass(slots=True)
class EmpiricalRoutedModelExecutor:
    selector: EmpiricalSelector
    invoke: ModelInvoker
    adaptive_policy: AdaptivePolicy | None = None
    min_success_probability: float = 0.0
    token_estimator: TokenEstimator = default_token_estimator

    def __call__(self, task: Task, context: ExecutionContext) -> ExecutionResult:
        estimated_input, estimated_output = self.token_estimator(task)
        probability_floor = self.min_success_probability
        if self.adaptive_policy is not None:
            probability_floor = max(
                probability_floor,
                self.adaptive_policy.reliability_floor(task),
            )

        remaining_cost = None
        if context.budget.max_cost_usd is not None:
            remaining_cost = max(
                context.budget.max_cost_usd - context.budget.cost_usd,
                0.0,
            )

        candidates = self.selector.ranked(
            task,
            context.decision.execution_class,
            input_tokens=estimated_input,
            output_tokens=estimated_output,
            min_success_probability=probability_floor,
            max_estimated_cost=remaining_cost,
        )
        if not candidates:
            raise ModelInvocationFailed("no empirical candidate satisfies routing constraints")

        remaining_model_calls = _remaining_model_calls(context)

        failures: list[str] = []
        for candidate in candidates:
            if remaining_model_calls is not None and len(failures) >= remaining_model_calls:
                # Bound the fallback fan-out to the model-call budget (see RoutedModelExecutor).
                raise BudgetExceeded("model-call budget exhausted during provider fallback")
            profile = candidate.profile
            try:
                response = self.invoke(profile.provider, profile.name, task)
            except UnknownProvider:
                # Configuration error, not a transient failure: fail fast.
                raise
            except Exception as exc:
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
                    "empirical_success_probability": candidate.success_probability,
                    "empirical_expected_total_cost": candidate.expected_total_cost,
                    "empirical_feature_key": candidate.feature_key,
                    "empirical_probability_floor": probability_floor,
                    "fallback_failures": tuple(failures),
                }
            )
            return ExecutionResult(
                output=response.output,
                cost_usd=cost,
                model_calls=len(failures) + 1,
                metadata=metadata,
            )

        raise ModelInvocationFailed(
            "all empirical model invocations failed: " + "; ".join(failures)
        )
