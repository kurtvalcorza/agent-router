import pytest

from agent_router import (
    Budget,
    BudgetExceeded,
    ExecutionClass,
    ExecutionResult,
    Requirement,
    RouterRuntime,
    Task,
    Verification,
    VerificationStatus,
)


def test_exact_computation_routes_deterministically() -> None:
    runtime = RouterRuntime()
    runtime.register_executor(
        ExecutionClass.DETERMINISTIC,
        lambda task, ctx: ExecutionResult(output=42, tool_calls=1),
    )

    result = runtime.execute(
        Task(
            kind="arithmetic",
            payload={"expression": "6 * 7"},
            requirements={Requirement.EXACT_COMPUTATION},
        )
    )

    assert result.output == 42


def test_failed_verification_escalates_to_stronger_executor() -> None:
    seen: list[ExecutionClass] = []

    def verifier(task: Task, result: ExecutionResult) -> Verification:
        if result.output == "weak":
            return Verification(VerificationStatus.ESCALATE, "insufficient evidence")
        return Verification(VerificationStatus.PASS)

    runtime = RouterRuntime(verifier=verifier)

    def deterministic(task, ctx):
        seen.append(ctx.decision.execution_class)
        return ExecutionResult(output="weak")

    def light(task, ctx):
        seen.append(ctx.decision.execution_class)
        return ExecutionResult(output="verified", model_calls=1, cost_usd=0.001)

    runtime.register_executor(ExecutionClass.DETERMINISTIC, deterministic)
    runtime.register_executor(ExecutionClass.LIGHT_REASONING, light)

    result = runtime.execute(Task(kind="classification", payload={"text": "example"}))

    assert result.output == "verified"
    assert seen == [ExecutionClass.DETERMINISTIC, ExecutionClass.LIGHT_REASONING]


def test_budget_is_enforced_after_executor_reports_usage() -> None:
    runtime = RouterRuntime()

    def costly_light_executor(task, ctx):
        return ExecutionResult(output="answer", model_calls=1, cost_usd=0.02)

    runtime.register_executor(
        ExecutionClass.LIGHT_REASONING,
        costly_light_executor,
    )

    task = Task(
        kind="semantic",
        payload={"question": "why?"},
        requirements={Requirement.SEMANTIC_REASONING},
    )

    with pytest.raises(BudgetExceeded, match="cost budget exceeded"):
        runtime.execute(task, budget=Budget(max_cost_usd=0.01))


def test_exhausted_model_call_budget_gates_before_invocation() -> None:
    # An already-spent model-call budget must not fund another real invocation only to raise
    # afterwards in ``consume``; the runtime gates before calling the executor.
    invocations: list[str] = []

    def light_executor(task, ctx):
        invocations.append(task.kind)
        return ExecutionResult(output="answer", model_calls=1, cost_usd=0.0)

    runtime = RouterRuntime()
    runtime.register_executor(ExecutionClass.LIGHT_REASONING, light_executor)

    task = Task(
        kind="semantic",
        payload={"question": "why?"},
        requirements={Requirement.SEMANTIC_REASONING},
    )

    with pytest.raises(BudgetExceeded, match="model-call budget exceeded"):
        runtime.execute(task, budget=Budget(max_model_calls=1, model_calls=1))
    assert invocations == []


def test_missing_executor_escalates() -> None:
    runtime = RouterRuntime(max_attempts=2)
    runtime.register_executor(
        ExecutionClass.LIGHT_REASONING,
        lambda task, ctx: ExecutionResult(output="fallback", model_calls=1),
    )

    result = runtime.execute(Task(kind="unknown", payload={}))

    assert result.output == "fallback"
