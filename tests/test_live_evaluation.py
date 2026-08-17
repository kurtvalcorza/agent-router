from agent_router import (
    EvaluationCase,
    ModelResponse,
    PolicyMode,
    ProviderInvoker,
    parse_catalog,
    run_fixed_baseline,
    run_router_strategy,
)


def catalog():
    return parse_catalog(
        {
            "version": "test",
            "aliases": {"cheap": "small", "strong": "large"},
            "models": [
                {
                    "name": "small",
                    "provider": "test",
                    "execution_classes": ["light_reasoning"],
                    "capabilities": ["semantic_reasoning"],
                    "reliability": 0.90,
                    "pricing": {"input_per_million": 1.0, "output_per_million": 2.0},
                },
                {
                    "name": "large",
                    "provider": "test",
                    "execution_classes": ["deep_reasoning", "light_reasoning"],
                    "capabilities": ["semantic_reasoning", "deep_planning", "high_reliability"],
                    "reliability": 0.99,
                    "pricing": {"input_per_million": 5.0, "output_per_million": 10.0},
                },
            ],
        }
    )


def cases():
    return (
        EvaluationCase(
            id="light",
            task_kind="qa",
            metadata={
                "payload": {"prompt": "q"},
                "requirements": ["semantic_reasoning"],
                "grader": "text_exact",
                "expected": "ok",
            },
        ),
        EvaluationCase(
            id="deep",
            task_kind="analysis",
            metadata={
                "payload": {"prompt": "q"},
                "requirements": ["semantic_reasoning", "deep_planning"],
                "grader": "text_exact",
                "expected": "ok",
            },
        ),
    )


def invoker():
    return ProviderInvoker(
        {
            "test": lambda model, task: ModelResponse(
                output="ok",
                input_tokens=100,
                output_tokens=10,
            )
        }
    )


def test_router_uses_small_for_light_and_large_for_deep():
    runs = run_router_strategy(
        cases(),
        catalog=catalog(),
        mode=PolicyMode.BALANCED,
        invoke=invoker(),
    )
    assert [run.model for run in runs] == ["small", "large"]
    assert all(run.success for run in runs)


def test_fixed_baseline_uses_alias_target_for_every_case():
    runs = run_fixed_baseline(
        cases(),
        strategy="always-strong",
        catalog=catalog(),
        model="strong",
        invoke=invoker(),
    )
    assert [run.model for run in runs] == ["large", "large"]
