from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from agent_router import Budget, ExecutionClass, VerificationStatus

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "quickstart.py"


@pytest.fixture(scope="module")
def quickstart():
    """Import ``examples/quickstart.py``, which lives outside the installed package."""
    spec = importlib.util.spec_from_file_location("quickstart_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        del sys.modules[spec.name]


def test_example_runs_without_provider_credentials(quickstart, capsys) -> None:
    assert quickstart.main() == 0
    assert "telemetry event(s)" in capsys.readouterr().out


def test_cheap_task_uses_the_cheap_model(quickstart) -> None:
    events: list = []
    runtime, _ = quickstart.build_runtime(events)
    task = quickstart.Task(
        kind="qa",
        payload={"prompt": "Capital of France?", "cheap_answer": "Paris"},
        requirements={quickstart.Requirement.SEMANTIC_REASONING},
        risk=quickstart.Risk.LOW,
        metadata={
            "expected": "Paris",
            "estimated_input_tokens": 20,
            "estimated_output_tokens": 5,
        },
    )

    budget = Budget(max_cost_usd=0.10, max_model_calls=4)
    result = runtime.execute(task, budget=budget)

    assert result.metadata["model"] == "small-model"
    assert budget.model_calls == 1
    assert [event.verification for event in events] == [VerificationStatus.PASS]


def test_failed_verification_escalates_to_the_strong_model(quickstart) -> None:
    events: list = []
    runtime, _ = quickstart.build_runtime(events)
    task = quickstart.Task(
        kind="qa",
        payload={"prompt": "Summarize the filing.", "cheap_answer": "no comment"},
        requirements={quickstart.Requirement.SEMANTIC_REASONING},
        risk=quickstart.Risk.LOW,
        metadata={
            "expected": "revenue grew 12%",
            "estimated_input_tokens": 60,
            "estimated_output_tokens": 80,
        },
    )

    budget = Budget(max_cost_usd=0.10, max_model_calls=4)
    result = runtime.execute(task, budget=budget)

    assert result.metadata["model"] == "strong-model"
    assert budget.model_calls == 2
    assert [event.verification for event in events] == [
        VerificationStatus.ESCALATE,
        VerificationStatus.PASS,
    ]
    assert [event.execution_class for event in events] == [
        ExecutionClass.LIGHT_REASONING,
        ExecutionClass.DEEP_REASONING,
    ]
