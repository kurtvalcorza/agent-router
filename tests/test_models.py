import pytest

from agent_router import (
    ExecutionClass,
    ModelProfile,
    ModelRegistry,
    NoEligibleModel,
    Requirement,
    Task,
)


def test_selects_cheapest_eligible_model() -> None:
    registry = ModelRegistry(
        [
            ModelProfile(
                name="cheap",
                provider="a",
                execution_classes={ExecutionClass.LIGHT_REASONING},
                capabilities={Requirement.SEMANTIC_REASONING},
                input_cost_per_million=0.2,
                output_cost_per_million=0.8,
                reliability=0.9,
            ),
            ModelProfile(
                name="expensive",
                provider="b",
                execution_classes={ExecutionClass.LIGHT_REASONING},
                capabilities={Requirement.SEMANTIC_REASONING},
                input_cost_per_million=2.0,
                output_cost_per_million=8.0,
                reliability=0.99,
            ),
        ]
    )
    task = Task(
        kind="semantic",
        payload={},
        requirements={Requirement.SEMANTIC_REASONING},
    )

    selected = registry.select(
        task,
        ExecutionClass.LIGHT_REASONING,
        input_tokens=2_000,
        output_tokens=500,
    )

    assert selected.name == "cheap"


def test_reliability_floor_can_force_stronger_model() -> None:
    registry = ModelRegistry(
        [
            ModelProfile(
                name="cheap",
                provider="a",
                execution_classes={ExecutionClass.LIGHT_REASONING},
                capabilities={Requirement.SEMANTIC_REASONING},
                reliability=0.8,
            ),
            ModelProfile(
                name="reliable",
                provider="b",
                execution_classes={ExecutionClass.LIGHT_REASONING},
                capabilities={Requirement.SEMANTIC_REASONING},
                reliability=0.98,
            ),
        ]
    )
    task = Task(
        kind="semantic",
        payload={},
        requirements={Requirement.SEMANTIC_REASONING},
    )

    assert registry.select(
        task,
        ExecutionClass.LIGHT_REASONING,
        min_reliability=0.95,
    ).name == "reliable"


def test_capability_and_context_constraints_filter_models() -> None:
    registry = ModelRegistry(
        [
            ModelProfile(
                name="small-context",
                provider="a",
                execution_classes={ExecutionClass.DEEP_REASONING},
                capabilities={Requirement.SEMANTIC_REASONING, Requirement.LONG_CONTEXT},
                context_window=8_000,
            )
        ]
    )
    task = Task(
        kind="synthesis",
        payload={},
        requirements={Requirement.SEMANTIC_REASONING, Requirement.LONG_CONTEXT},
        metadata={"context_tokens": 20_000},
    )

    with pytest.raises(NoEligibleModel):
        registry.select(task, ExecutionClass.DEEP_REASONING)


def _two_tier_registry() -> ModelRegistry:
    return ModelRegistry(
        [
            ModelProfile(
                name="small",
                provider="a",
                execution_classes={ExecutionClass.LIGHT_REASONING},
                capabilities={Requirement.SEMANTIC_REASONING},
                input_cost_per_million=1.0,
                output_cost_per_million=2.0,
                reliability=0.90,
            ),
            ModelProfile(
                name="large",
                provider="b",
                execution_classes={ExecutionClass.LIGHT_REASONING},
                capabilities={Requirement.SEMANTIC_REASONING},
                input_cost_per_million=5.0,
                output_cost_per_million=10.0,
                reliability=0.99,
            ),
        ]
    )


def test_zero_estimate_ranking_prefers_cheaper_list_price_not_reliability() -> None:
    # Regression: with no token estimates every model estimates cost 0.0, and the old
    # ``-reliability`` tiebreak promoted the most expensive high-reliability model.
    registry = _two_tier_registry()
    task = Task(kind="qa", payload={}, requirements={Requirement.SEMANTIC_REASONING})

    ranked = registry.ranked(task, ExecutionClass.LIGHT_REASONING)

    assert [profile.name for profile in ranked] == ["small", "large"]


def test_context_window_enforced_from_estimated_tokens() -> None:
    # Regression: the guard used to key only on ``context_tokens`` and was a no-op on the
    # real routing path, which sets ``estimated_input_tokens``/``estimated_output_tokens``.
    registry = ModelRegistry(
        [
            ModelProfile(
                name="small-context",
                provider="a",
                execution_classes={ExecutionClass.LIGHT_REASONING},
                capabilities={Requirement.SEMANTIC_REASONING},
                context_window=8_000,
            )
        ]
    )
    task = Task(
        kind="qa",
        payload={},
        requirements={Requirement.SEMANTIC_REASONING},
        metadata={"estimated_input_tokens": 20_000, "estimated_output_tokens": 1_000},
    )

    with pytest.raises(NoEligibleModel):
        registry.select(task, ExecutionClass.LIGHT_REASONING)
