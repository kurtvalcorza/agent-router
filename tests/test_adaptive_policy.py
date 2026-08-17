import pytest

from agent_router import (
    AdaptivePolicy,
    Budget,
    ExecutionClass,
    ExecutionContext,
    ModelProfile,
    ModelRegistry,
    ModelResponse,
    NoEligibleModel,
    PolicyMode,
    Requirement,
    Risk,
    RouteDecision,
    RoutedModelExecutor,
    Task,
)


def _registry() -> ModelRegistry:
    return ModelRegistry(
        [
            ModelProfile(
                name="cheap",
                provider="a",
                execution_classes={ExecutionClass.LIGHT_REASONING},
                capabilities={Requirement.SEMANTIC_REASONING},
                reliability=0.80,
                input_cost_per_million=0.10,
                output_cost_per_million=0.20,
            ),
            ModelProfile(
                name="strong",
                provider="b",
                execution_classes={ExecutionClass.LIGHT_REASONING},
                capabilities={Requirement.SEMANTIC_REASONING},
                reliability=0.98,
                input_cost_per_million=2.0,
                output_cost_per_million=4.0,
            ),
        ]
    )


def _task(risk: Risk = Risk.LOW) -> Task:
    return Task(
        kind="semantic",
        payload={"question": "why?"},
        requirements={Requirement.SEMANTIC_REASONING},
        risk=risk,
        metadata={"estimated_input_tokens": 1_000, "estimated_output_tokens": 500},
    )


def _context(budget: Budget | None = None) -> ExecutionContext:
    return ExecutionContext(
        budget=budget or Budget(),
        attempt=1,
        decision=RouteDecision(ExecutionClass.LIGHT_REASONING, "semantic task"),
    )


def test_economy_selects_cheapest_eligible_model() -> None:
    executor = RoutedModelExecutor(
        registry=_registry(),
        adaptive_policy=AdaptivePolicy(PolicyMode.ECONOMY),
        invoke=lambda provider, model, task: ModelResponse(output=model),
    )

    assert executor(_task(), _context()).output == "cheap"


def test_quality_rejects_low_reliability_model() -> None:
    executor = RoutedModelExecutor(
        registry=_registry(),
        adaptive_policy=AdaptivePolicy(PolicyMode.QUALITY),
        invoke=lambda provider, model, task: ModelResponse(output=model),
    )

    assert executor(_task(), _context()).output == "strong"


def test_high_risk_raises_floor_even_in_economy_mode() -> None:
    policy = AdaptivePolicy(PolicyMode.ECONOMY)
    assert policy.reliability_floor(_task(Risk.HIGH)) == 0.97


def test_budget_filters_models_by_estimated_cost() -> None:
    executor = RoutedModelExecutor(
        registry=_registry(),
        adaptive_policy=AdaptivePolicy(PolicyMode.QUALITY),
        invoke=lambda provider, model, task: ModelResponse(output=model),
    )

    with pytest.raises(NoEligibleModel):
        executor(_task(), _context(Budget(max_cost_usd=0.001)))
