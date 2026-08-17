from agent_router import (
    Budget,
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
    task = Task(
        kind="semantic",
        payload={"question": "why?"},
        requirements={Requirement.SEMANTIC_REASONING},
        metadata={"estimated_input_tokens": 900, "estimated_output_tokens": 400},
    )
    context = ExecutionContext(
        budget=Budget(),
        attempt=1,
        decision=RouteDecision(ExecutionClass.LIGHT_REASONING, "semantic task"),
    )

    result = executor(task, context)

    assert result.output == "answer"
    assert result.model_calls == 1
    assert result.cost_usd == 0.002
    assert result.metadata["model"] == "model-small"
    assert calls == [("provider-a", "model-small")]
