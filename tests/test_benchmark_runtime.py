from agent_router.benchmark_runtime import (
    BenchmarkSpecError,
    case_to_task,
    execute_task_strategy,
    grade_output,
)
from agent_router.evaluation import EvaluationCase
from agent_router.types import ExecutionResult, Requirement, Risk


def case(**metadata):
    return EvaluationCase(
        id="c1",
        task_kind="qa",
        minimum_quality=0.5,
        metadata=metadata,
    )


def test_case_compiles_to_task():
    task = case_to_task(
        case(
            payload={"prompt": "hi"},
            requirements=["semantic_reasoning", "long_context"],
            risk="medium",
            task_metadata={"estimated_input_tokens": 100},
        )
    )
    assert task.payload == {"prompt": "hi"}
    assert task.requirements == {Requirement.SEMANTIC_REASONING, Requirement.LONG_CONTEXT}
    assert task.risk is Risk.MEDIUM
    assert task.metadata["estimated_input_tokens"] == 100


def test_contains_all_grader_returns_fractional_quality():
    evaluation = case(grader="contains_all", expected=["alpha", "beta"])
    grade = grade_output(evaluation, "Alpha only")
    assert grade.quality == 0.5
    assert grade.success is True


def test_execute_task_strategy_records_model_cost_and_quality():
    evaluation = case(grader="text_exact", expected="answer", payload={"prompt": "q"})

    def execute(task):
        return ExecutionResult(
            output="answer",
            cost_usd=0.02,
            metadata={"model": "m1", "provider": "test"},
        )

    runs = execute_task_strategy([evaluation], strategy="router", execute_task=execute)
    assert len(runs) == 1
    assert runs[0].model == "m1"
    assert runs[0].quality == 1.0
    assert runs[0].cost_usd == 0.02
    assert runs[0].success is True


def test_invalid_requirement_fails_closed():
    evaluation = case(requirements=["not-real"])
    try:
        case_to_task(evaluation)
    except BenchmarkSpecError:
        pass
    else:
        raise AssertionError("expected BenchmarkSpecError")
