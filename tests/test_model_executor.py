import pytest

from agent_router import (
    Budget,
    BudgetExceeded,
    ExecutionClass,
    ExecutionContext,
    ModelProfile,
    ModelRegistry,
    ModelResponse,
    Requirement,
    RouteDecision,
    RoutedModelExecutor,
    Task,
)


def _context(budget: Budget | None = None) -> ExecutionContext:
    return ExecutionContext(
        budget=budget or Budget(),
        attempt=1,
        decision=RouteDecision(ExecutionClass.LIGHT_REASONING, "semantic task"),
    )


def _task() -> Task:
    return Task(
        kind="semantic",
        payload={"question": "why?"},
        requirements={Requirement.SEMANTIC_REASONING},
        metadata={"estimated_input_tokens": 900, "estimated_output_tokens": 400},
    )


def test_executor_invokes_selected_provider_and_reports_usage() -> None:
    registry = ModelRegistry(
        [
            ModelProfile(
                name="model-small",
                provider="provider-a",
                execution_classes={ExecutionClass.LIGHT_REASONING},
                capabilities={Requirement.SEMANTIC_REASONING},
                input_cost_per_million=1.0,
                output_cost_per_million=2.0,
            )
        ]
    )
    calls: list[tuple[str, str]] = []

    def invoke(provider: str, model: str, task: Task) -> ModelResponse:
        calls.append((provider, model))
        return ModelResponse(output="answer", input_tokens=1_000, output_tokens=500)

    executor = RoutedModelExecutor(registry=registry, invoke=invoke)
    result = executor(_task(), _context())

    assert result.output == "answer"
    assert result.model_calls == 1
    assert result.cost_usd == 0.002
    assert result.metadata["model"] == "model-small"
    assert calls == [("provider-a", "model-small")]


def test_executor_falls_back_to_next_ranked_model() -> None:
    registry = ModelRegistry(
        [
            ModelProfile(
                name="cheap",
                provider="provider-a",
                execution_classes={ExecutionClass.LIGHT_REASONING},
                capabilities={Requirement.SEMANTIC_REASONING},
                input_cost_per_million=0.1,
                output_cost_per_million=0.2,
            ),
            ModelProfile(
                name="fallback",
                provider="provider-b",
                execution_classes={ExecutionClass.LIGHT_REASONING},
                capabilities={Requirement.SEMANTIC_REASONING},
                input_cost_per_million=1.0,
                output_cost_per_million=2.0,
            ),
        ]
    )
    calls: list[str] = []

    def invoke(provider: str, model: str, task: Task) -> ModelResponse:
        calls.append(model)
        if model == "cheap":
            raise RuntimeError("temporary provider failure")
        return ModelResponse(output="fallback answer", input_tokens=1_000, output_tokens=500)

    executor = RoutedModelExecutor(registry=registry, invoke=invoke)
    result = executor(_task(), _context())

    assert result.output == "fallback answer"
    assert result.model_calls == 2
    assert result.metadata["model"] == "fallback"
    assert len(result.metadata["fallback_failures"]) == 1
    assert calls == ["cheap", "fallback"]


def test_unknown_provider_fails_fast_without_exhausting_candidates() -> None:
    # Regression: an unregistered provider is a config error and must propagate immediately,
    # not be caught as a transient failure and retried across every candidate.
    import pytest

    from agent_router import ProviderInvoker, UnknownProvider

    registry = ModelRegistry(
        [
            ModelProfile(
                name="model-small",
                provider="provider-a",
                execution_classes={ExecutionClass.LIGHT_REASONING},
                capabilities={Requirement.SEMANTIC_REASONING},
                input_cost_per_million=1.0,
                output_cost_per_million=2.0,
            )
        ]
    )
    executor = RoutedModelExecutor(registry=registry, invoke=ProviderInvoker({}))

    with pytest.raises(UnknownProvider):
        executor(_task(), _context())


def _three_tier_registry() -> ModelRegistry:
    return ModelRegistry(
        [
            ModelProfile(
                name="c1",
                provider="p",
                execution_classes={ExecutionClass.LIGHT_REASONING},
                capabilities={Requirement.SEMANTIC_REASONING},
                input_cost_per_million=1.0,
                output_cost_per_million=1.0,
            ),
            ModelProfile(
                name="c2",
                provider="p",
                execution_classes={ExecutionClass.LIGHT_REASONING},
                capabilities={Requirement.SEMANTIC_REASONING},
                input_cost_per_million=2.0,
                output_cost_per_million=2.0,
            ),
            ModelProfile(
                name="c3",
                provider="p",
                execution_classes={ExecutionClass.LIGHT_REASONING},
                capabilities={Requirement.SEMANTIC_REASONING},
                input_cost_per_million=3.0,
                output_cost_per_million=3.0,
            ),
        ]
    )


def test_fallback_fan_out_cannot_exceed_model_call_budget() -> None:
    # Regression: a single executor invocation used to make one real provider call per
    # candidate, so fallback across N failing candidates could exceed max_model_calls before
    # runtime's post-call accounting ever ran.
    calls: list[str] = []

    def invoke(provider: str, model: str, task: Task) -> ModelResponse:
        calls.append(model)
        raise RuntimeError("temporary provider failure")

    executor = RoutedModelExecutor(registry=_three_tier_registry(), invoke=invoke)

    with pytest.raises(BudgetExceeded, match="model-call budget exhausted"):
        executor(_task(), _context(Budget(max_model_calls=2)))

    # Stopped at the budget: two real calls attempted, the third candidate never invoked.
    assert calls == ["c1", "c2"]


def test_fallback_within_model_call_budget_still_succeeds() -> None:
    # The guard must not break legitimate fallback that stays within budget.
    calls: list[str] = []

    def invoke(provider: str, model: str, task: Task) -> ModelResponse:
        calls.append(model)
        if model == "c1":
            raise RuntimeError("temporary provider failure")
        return ModelResponse(output="ok", input_tokens=1_000, output_tokens=500)

    executor = RoutedModelExecutor(registry=_three_tier_registry(), invoke=invoke)
    result = executor(_task(), _context(Budget(max_model_calls=2)))

    assert result.output == "ok"
    assert result.model_calls == 2
    assert calls == ["c1", "c2"]
