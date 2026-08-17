from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from .evaluation import EvaluationCase, EvaluationRun
from .evaluation_runner import StrategyResult, run_strategy
from .model_executor import ModelResponse
from .models import ModelProfile
from .types import Budget, Requirement, Risk, Task


class BenchmarkSpecError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Grade:
    quality: float
    success: bool
    metadata: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError("quality must be between 0 and 1")


TaskExecutor = Callable[[Task], Any]
DirectModelInvoker = Callable[[str, str, Task], ModelResponse]
Grader = Callable[[EvaluationCase, object], Grade]


def case_to_task(case: EvaluationCase) -> Task:
    metadata = case.metadata
    payload = metadata.get("payload", {})
    if not isinstance(payload, dict):
        raise BenchmarkSpecError(f"case {case.id!r} metadata.payload must be an object")

    raw_requirements = metadata.get("requirements", [Requirement.SEMANTIC_REASONING.value])
    if not isinstance(raw_requirements, list):
        raise BenchmarkSpecError(f"case {case.id!r} metadata.requirements must be a list")
    try:
        requirements = {Requirement(str(value)) for value in raw_requirements}
    except ValueError as exc:
        raise BenchmarkSpecError(f"case {case.id!r} has an invalid requirement") from exc

    try:
        risk = Risk(str(metadata.get("risk", Risk.LOW.value)))
    except ValueError as exc:
        raise BenchmarkSpecError(f"case {case.id!r} has an invalid risk") from exc

    task_metadata = metadata.get("task_metadata", {})
    if not isinstance(task_metadata, dict):
        raise BenchmarkSpecError(f"case {case.id!r} metadata.task_metadata must be an object")

    return Task(
        kind=case.task_kind,
        payload=dict(payload),
        requirements=requirements,
        risk=risk,
        metadata=dict(task_metadata),
    )


def grade_output(case: EvaluationCase, output: object) -> Grade:
    grader = case.metadata.get("grader", "exact_match")
    expected = case.metadata.get("expected")

    if grader == "exact_match":
        success = output == expected
        return Grade(quality=1.0 if success else 0.0, success=success)

    if grader == "text_exact":
        if not isinstance(expected, str):
            raise BenchmarkSpecError(f"case {case.id!r} text_exact grader requires string expected")
        actual = str(output).strip()
        success = actual == expected.strip()
        return Grade(quality=1.0 if success else 0.0, success=success)

    if grader == "contains_all":
        if not isinstance(expected, list) or not all(isinstance(item, str) for item in expected):
            raise BenchmarkSpecError(
                f"case {case.id!r} contains_all grader requires a list of strings"
            )
        text = str(output).lower()
        if not expected:
            return Grade(quality=1.0, success=True)
        matched = sum(item.lower() in text for item in expected)
        quality = matched / len(expected)
        return Grade(quality=quality, success=quality >= case.minimum_quality)

    raise BenchmarkSpecError(f"case {case.id!r} uses unsupported grader {grader!r}")


def execute_task_strategy(
    cases: Iterable[EvaluationCase],
    *,
    strategy: str,
    execute_task: TaskExecutor,
    grader: Grader = grade_output,
) -> tuple[EvaluationRun, ...]:
    def execute(case: EvaluationCase) -> StrategyResult:
        task = case_to_task(case)
        raw = execute_task(task)
        output = getattr(raw, "output", raw)
        grade = grader(case, output)
        metadata = dict(getattr(raw, "metadata", {}) or {})
        metadata.update(dict(grade.metadata or {}))
        return StrategyResult(
            model=metadata.get("model") if isinstance(metadata.get("model"), str) else None,
            quality=grade.quality,
            cost_usd=float(getattr(raw, "cost_usd", 0.0)),
            escalations=int(metadata.get("escalations", 0)),
            success=grade.success,
            metadata=metadata,
        )

    return run_strategy(cases, strategy=strategy, execute=execute)


def fixed_model_executor(
    profile: ModelProfile,
    invoke: DirectModelInvoker,
) -> TaskExecutor:
    def execute(task: Task):
        response = invoke(profile.provider, profile.name, task)
        cost = profile.estimate_cost(
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
        from .types import ExecutionResult

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

    return execute


def runtime_executor(runtime, *, budget_factory: Callable[[], Budget] | None = None) -> TaskExecutor:
    budget_factory = budget_factory or Budget

    def execute(task: Task):
        return runtime.execute(task, budget=budget_factory())

    return execute
