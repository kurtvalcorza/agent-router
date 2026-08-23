import json

import pytest

from agent_router.adaptive import AdaptivePolicy, PolicyMode
from agent_router.catalog import parse_catalog
from agent_router.cli import main
from agent_router.delegation import (
    DEFAULT_DELEGATION_THRESHOLD_TOKENS,
    estimate_tokens,
    parse_requirements,
    parse_risk,
    plan_delegation,
)
from agent_router.types import Requirement, Risk, Task

CATALOG = {
    "version": "1",
    "pricing_as_of": "2026-08-01",
    "models": [
        {
            "name": "cheap",
            "provider": "openai",
            "execution_classes": ["light_reasoning"],
            "capabilities": ["semantic_reasoning"],
            "reliability": 0.90,
            "context_window": 100000,
            "pricing": {"input_per_million": 1.0, "output_per_million": 2.0},
        },
        {
            "name": "strong",
            "provider": "anthropic",
            "execution_classes": ["light_reasoning", "deep_reasoning"],
            "capabilities": [
                "semantic_reasoning",
                "deep_planning",
                "high_reliability",
            ],
            "reliability": 0.98,
            "context_window": 200000,
            "pricing": {"input_per_million": 10.0, "output_per_million": 30.0},
        },
    ],
}


def _registry():
    return parse_catalog(CATALOG).registry()


def _task(*, input_tokens: int, output_tokens: int, requirements=None, risk=Risk.LOW) -> Task:
    return Task(
        kind="subtask",
        payload={"prompt": "x"},
        requirements=requirements or {Requirement.SEMANTIC_REASONING},
        risk=risk,
        metadata={
            "estimated_input_tokens": input_tokens,
            "estimated_output_tokens": output_tokens,
        },
    )


def test_small_task_is_not_worth_delegating() -> None:
    decision = plan_delegation(_task(input_tokens=10, output_tokens=5), registry=_registry())

    assert decision.delegate is False
    assert "threshold" in decision.reason
    assert decision.model is None


def test_bulk_task_delegates_to_the_cheapest_eligible_model() -> None:
    decision = plan_delegation(_task(input_tokens=9000, output_tokens=2000), registry=_registry())

    assert decision.delegate is True
    assert (decision.provider, decision.model) == ("openai", "cheap")
    assert decision.estimated_cost_usd == pytest.approx(9000 / 1e6 * 1.0 + 2000 / 1e6 * 2.0)
    # The pricier eligible model is reported so the caller can see what was avoided.
    assert [(p, m) for p, m, _ in decision.alternatives] == [("anthropic", "strong")]


def test_high_reliability_requirement_excludes_the_cheap_model() -> None:
    decision = plan_delegation(
        _task(
            input_tokens=9000,
            output_tokens=2000,
            requirements={Requirement.SEMANTIC_REASONING, Requirement.HIGH_RELIABILITY},
        ),
        registry=_registry(),
        adaptive_policy=AdaptivePolicy(PolicyMode.BALANCED),
    )

    assert decision.delegate is True
    assert decision.model == "strong"
    assert decision.reliability_floor >= 0.95
    assert decision.alternatives == ()


def test_non_model_execution_class_is_reported_rather_than_priced() -> None:
    decision = plan_delegation(
        _task(input_tokens=9000, output_tokens=2000, requirements={Requirement.EXACT_COMPUTATION}),
        registry=_registry(),
    )

    assert decision.delegate is False
    assert "not a model execution class" in decision.reason
    assert decision.model is None


def test_cost_ceiling_below_every_model_blocks_delegation() -> None:
    decision = plan_delegation(
        _task(input_tokens=9000, output_tokens=2000),
        registry=_registry(),
        max_cost_usd=0.000_001,
    )

    assert decision.delegate is False
    assert "no model satisfies" in decision.reason


def test_threshold_is_configurable() -> None:
    task = _task(input_tokens=10, output_tokens=5)

    assert plan_delegation(task, registry=_registry(), threshold_tokens=0).delegate is True
    assert plan_delegation(task, registry=_registry()).delegate is False


def test_negative_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="threshold_tokens"):
        plan_delegation(
            _task(input_tokens=1, output_tokens=1),
            registry=_registry(),
            threshold_tokens=-1,
        )


def test_estimate_tokens_scales_with_prompt_length() -> None:
    small_in, small_out = estimate_tokens("hi")
    large_in, large_out = estimate_tokens("word " * 1000)

    assert large_in > small_in
    assert small_in >= 1 and small_out >= 1


def test_parse_requirements_round_trips_and_names_valid_values() -> None:
    assert parse_requirements("semantic_reasoning,long_context") == {
        Requirement.SEMANTIC_REASONING,
        Requirement.LONG_CONTEXT,
    }
    assert parse_requirements(None) == set()

    with pytest.raises(ValueError, match="semantic_reasoning"):
        parse_requirements("nonsense")


def test_parse_risk_names_valid_values() -> None:
    assert parse_risk("high") is Risk.HIGH
    assert parse_risk(None) is Risk.LOW

    with pytest.raises(ValueError, match="medium"):
        parse_risk("catastrophic")


# --- CLI surface -------------------------------------------------------------


def _catalog_file(tmp_path):
    path = tmp_path / "models.json"
    path.write_text(json.dumps(CATALOG), encoding="utf-8")
    return path


def test_route_defaults_to_planning_and_never_invokes_a_provider(tmp_path, capsys) -> None:
    code = main(
        [
            "route",
            "Classify these tickets.",
            "--catalog",
            str(_catalog_file(tmp_path)),
            "--input-tokens",
            "9000",
            "--output-tokens",
            "2000",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["delegate"] is True
    assert payload["executed"] is False
    assert payload["model"] == "cheap"


def test_route_reports_when_a_task_is_too_small_to_delegate(tmp_path, capsys) -> None:
    code = main(["route", "2+2?", "--catalog", str(_catalog_file(tmp_path)), "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["delegate"] is False
    assert str(DEFAULT_DELEGATION_THRESHOLD_TOKENS) in payload["reason"]


def test_route_rejects_an_empty_prompt(tmp_path, capsys) -> None:
    code = main(["route", "   ", "--catalog", str(_catalog_file(tmp_path))])

    assert code == 2
    assert "empty prompt" in capsys.readouterr().err


def test_route_below_threshold_does_not_execute_even_with_execute_flag(tmp_path, capsys) -> None:
    """--execute must stay inert when the threshold already said no: no provider adapter
    is constructed, so this passes without credentials."""
    code = main(
        ["route", "2+2?", "--catalog", str(_catalog_file(tmp_path)), "--execute", "--json"]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["delegate"] is False
    assert payload["executed"] is False
