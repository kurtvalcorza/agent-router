import pytest

from agent_router.benchmark_runtime import case_to_task
from agent_router.empirical import EmpiricalSelector, EmpiricalSuccessModel
from agent_router.evaluation import EvaluationCase, EvaluationRun
from agent_router.models import ModelProfile, ModelRegistry
from agent_router.types import ExecutionClass, Requirement


def case(case_id: str, *, kind: str = "qa", deep: bool = False) -> EvaluationCase:
    requirements = [
        Requirement.DEEP_PLANNING.value if deep else Requirement.SEMANTIC_REASONING.value
    ]
    return EvaluationCase(
        id=case_id,
        task_kind=kind,
        minimum_quality=1.0,
        metadata={"requirements": requirements, "payload": {"prompt": case_id}},
    )


def run(case_id: str, model: str, success: bool) -> EvaluationRun:
    return EvaluationRun(
        case_id=case_id,
        strategy="baseline",
        model=model,
        quality=1.0 if success else 0.0,
        cost_usd=0.01,
        latency_seconds=0.1,
        success=success,
    )


def registry() -> ModelRegistry:
    capability = {Requirement.SEMANTIC_REASONING, Requirement.DEEP_PLANNING}
    classes = {ExecutionClass.LIGHT_REASONING, ExecutionClass.DEEP_REASONING}
    return ModelRegistry(
        [
            ModelProfile(
                name="cheap",
                provider="test",
                execution_classes=classes,
                capabilities=capability,
                input_cost_per_million=1.0,
                output_cost_per_million=1.0,
            ),
            ModelProfile(
                name="strong",
                provider="test",
                execution_classes=classes,
                capabilities=capability,
                input_cost_per_million=5.0,
                output_cost_per_million=5.0,
            ),
        ]
    )


def test_success_model_uses_task_specific_history():
    cases = [case("l1"), case("l2"), case("d1", deep=True), case("d2", deep=True)]
    runs = [
        run("l1", "cheap", True),
        run("l2", "cheap", True),
        run("d1", "cheap", False),
        run("d2", "cheap", False),
        run("l1", "strong", True),
        run("l2", "strong", True),
        run("d1", "strong", True),
        run("d2", "strong", True),
    ]
    model = EmpiricalSuccessModel.fit(cases, runs)

    light = model.estimate("cheap", case_to_task(cases[0]))
    deep = model.estimate("cheap", case_to_task(cases[2]))
    assert light.probability > deep.probability


def test_selector_prefers_cheap_when_both_clear_floor():
    cases = [case("a"), case("b")]
    runs = [
        run("a", "cheap", True),
        run("b", "cheap", True),
        run("a", "strong", True),
        run("b", "strong", True),
    ]
    model = EmpiricalSuccessModel.fit(cases, runs)
    selector = EmpiricalSelector(registry=registry(), success_model=model)
    task = case_to_task(cases[0])

    selected = selector.select(
        task,
        ExecutionClass.LIGHT_REASONING,
        input_tokens=1000,
        output_tokens=100,
        min_success_probability=0.5,
    )
    assert selected.profile.name == "cheap"


def test_selector_rejects_cheap_when_empirical_probability_is_too_low():
    cases = [case("d1", deep=True), case("d2", deep=True), case("d3", deep=True)]
    runs = [
        run("d1", "cheap", False),
        run("d2", "cheap", False),
        run("d3", "cheap", False),
        run("d1", "strong", True),
        run("d2", "strong", True),
        run("d3", "strong", True),
    ]
    model = EmpiricalSuccessModel.fit(cases, runs)
    selector = EmpiricalSelector(registry=registry(), success_model=model)
    task = case_to_task(cases[0])

    selected = selector.select(
        task,
        ExecutionClass.DEEP_REASONING,
        input_tokens=1000,
        output_tokens=100,
        min_success_probability=0.6,
    )
    assert selected.profile.name == "strong"


def test_unseen_model_uses_non_extreme_prior():
    model = EmpiricalSuccessModel.fit([case("a")], [run("a", "cheap", True)])
    estimate = model.estimate("unseen", case_to_task(case("x")))
    assert 0.0 < estimate.probability < 1.0


def test_sparse_feature_shrinks_toward_model_global_rate_not_half():
    # Regression: adding the raw prior on top of the hierarchical pseudo-counts biased a
    # sparse-feature estimate toward 0.5 instead of the model's global observed rate.
    light_cases = [case(f"l{i}") for i in range(20)]
    runs = [run(f"l{i}", "cheap", success=(i > 0)) for i in range(20)]  # 19/20 global
    model = EmpiricalSuccessModel.fit(light_cases, runs)

    global_rate = (model.prior_alpha + 19) / (model.prior_alpha + model.prior_beta + 20)
    unseen_deep = model.estimate("cheap", case_to_task(case("d", deep=True)))

    assert unseen_deep.trials == 0
    assert unseen_deep.probability == pytest.approx(global_rate)
    assert unseen_deep.probability > 0.9
