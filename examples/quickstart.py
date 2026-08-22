"""Run the router end to end without provider credentials.

Every routing decision the library makes -- hard policy, adaptive reliability floor,
capability and budget filtering, cheapest-eligible selection, verification-driven
escalation -- is exercised through ``invoke``, which is just a callable. Substituting a
stub for the provider adapters therefore shows the real control flow at zero cost.

Run it:

    python examples/quickstart.py
"""

from __future__ import annotations

from pathlib import Path

from agent_router import (
    AdaptivePolicy,
    Budget,
    ExecutionClass,
    ExecutionResult,
    ModelResponse,
    PolicyMode,
    Requirement,
    Risk,
    RoutedModelExecutor,
    RouterRuntime,
    Task,
    Verification,
    VerificationStatus,
    load_catalog,
)

CATALOG = Path(__file__).resolve().parent.parent / "config" / "models.example.yaml"


def stub_invoke(provider: str, model: str, task: Task) -> ModelResponse:
    """Stand in for a provider adapter.

    The cheap model answers the easy task and fails the hard one, which is what drives
    the escalation in scenario 2.
    """
    print(f"      provider call -> {provider}/{model}")
    answer = task.metadata["expected"] if model == "strong-model" else task.payload["cheap_answer"]
    return ModelResponse(
        output=answer,
        input_tokens=task.metadata["estimated_input_tokens"],
        output_tokens=task.metadata["estimated_output_tokens"],
    )


def verify_expected(task: Task, result: ExecutionResult) -> Verification:
    """Escalate to a stronger execution class whenever the answer is wrong."""
    if result.output == task.metadata["expected"]:
        return Verification(VerificationStatus.PASS)
    return Verification(VerificationStatus.ESCALATE, reason="answer did not match expectation")


def build_runtime(events: list) -> tuple[RouterRuntime, RoutedModelExecutor]:
    registry = load_catalog(CATALOG).registry()
    executor = RoutedModelExecutor(
        registry=registry,
        invoke=stub_invoke,
        adaptive_policy=AdaptivePolicy(PolicyMode.BALANCED),
    )
    runtime = RouterRuntime(verifier=verify_expected, telemetry=events.append)
    for execution_class in (ExecutionClass.LIGHT_REASONING, ExecutionClass.DEEP_REASONING):
        runtime.register_executor(execution_class, executor)
    return runtime, executor


def main() -> int:
    events: list = []
    runtime, _ = build_runtime(events)

    scenarios = (
        (
            "1. cheap task, cheap model",
            Task(
                kind="qa",
                payload={"prompt": "Capital of France?", "cheap_answer": "Paris"},
                requirements={Requirement.SEMANTIC_REASONING},
                risk=Risk.LOW,
                metadata={
                    "expected": "Paris",
                    "estimated_input_tokens": 20,
                    "estimated_output_tokens": 5,
                },
            ),
        ),
        (
            "2. high-reliability task skips the cheap model entirely",
            Task(
                kind="analysis",
                payload={"prompt": "Assess counterparty risk.", "cheap_answer": "looks fine"},
                requirements={Requirement.SEMANTIC_REASONING, Requirement.HIGH_RELIABILITY},
                risk=Risk.MEDIUM,
                metadata={
                    "expected": "material exposure identified",
                    "estimated_input_tokens": 40,
                    "estimated_output_tokens": 120,
                },
            ),
        ),
        (
            "3. cheap model answers wrong, verification escalates",
            Task(
                kind="qa",
                payload={"prompt": "Summarize the filing.", "cheap_answer": "no comment"},
                requirements={Requirement.SEMANTIC_REASONING},
                risk=Risk.LOW,
                metadata={
                    "expected": "revenue grew 12%",
                    "estimated_input_tokens": 60,
                    "estimated_output_tokens": 80,
                },
            ),
        ),
    )

    for label, task in scenarios:
        print(f"\n{label}")
        decision = runtime.policy.route(task)
        print(f"   routed to {decision.execution_class.value}: {decision.reason}")

        budget = Budget(max_cost_usd=0.10, max_model_calls=4)
        result = runtime.execute(task, budget=budget)
        print(
            f"   answered by {result.metadata['model']} "
            f"(floor {result.metadata['reliability_floor']:.2f}) "
            f"for ${budget.cost_usd:.6f} in {budget.model_calls} model call(s)"
        )

    total = sum(event.cost_usd for event in events)
    print(f"\n{len(events)} telemetry event(s), ${total:.6f} total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
