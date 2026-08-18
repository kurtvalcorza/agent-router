import pytest

from agent_router import (
    Budget,
    BudgetExceeded,
    EmpiricalRoutedModelExecutor,
    EmpiricalSelector,
    EmpiricalSuccessModel,
    ExecutionClass,
    ExecutionContext,
    ModelProfile,
    ModelRegistry,
    ModelResponse,
    Requirement,
    RouteDecision,
    Task,
)


def _registry() -> ModelRegistry:
    return ModelRegistry(
        [
            ModelProfile(
                name=name,
                provider="p",
                execution_classes={ExecutionClass.LIGHT_REASONING},
                capabilities={Requirement.SEMANTIC_REASONING},
                input_cost_per_million=cost,
                output_cost_per_million=cost,
            )
            for name, cost in (("c1", 1.0), ("c2", 2.0), ("c3", 3.0))
        ]
    )


def _executor(registry: ModelRegistry, invoke) -> EmpiricalRoutedModelExecutor:
    # An untrained success model returns a non-extreme 0.5 for every model, so the empirical
    # ranking reduces to ascending call cost: c1, c2, c3.
    selector = EmpiricalSelector(registry=registry, success_model=EmpiricalSuccessModel())
    return EmpiricalRoutedModelExecutor(selector=selector, invoke=invoke)


def _context(budget: Budget) -> ExecutionContext:
    return ExecutionContext(
        budget=budget,
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


def test_empirical_fallback_fan_out_cannot_exceed_model_call_budget() -> None:
    calls: list[str] = []

    def invoke(provider: str, model: str, task: Task) -> ModelResponse:
        calls.append(model)
        raise RuntimeError("temporary provider failure")

    executor = _executor(_registry(), invoke)

    with pytest.raises(BudgetExceeded, match="model-call budget exhausted"):
        executor(_task(), _context(Budget(max_model_calls=2)))

    assert calls == ["c1", "c2"]


def test_empirical_fallback_within_budget_succeeds() -> None:
    calls: list[str] = []

    def invoke(provider: str, model: str, task: Task) -> ModelResponse:
        calls.append(model)
        if model == "c1":
            raise RuntimeError("temporary provider failure")
        return ModelResponse(output="ok", input_tokens=1_000, output_tokens=500)

    executor = _executor(_registry(), invoke)
    result = executor(_task(), _context(Budget(max_model_calls=2)))

    assert result.output == "ok"
    assert result.model_calls == 2
    assert calls == ["c1", "c2"]
