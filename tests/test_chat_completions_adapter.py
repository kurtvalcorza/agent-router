import json
from types import SimpleNamespace

import pytest

from agent_router import (
    OpenAIChatCompletionsAdapter,
    Task,
)
from agent_router.catalog import parse_catalog
from agent_router.delegation import plan_delegation
from agent_router.types import Requirement, Risk


class FakeChatCompletions:
    """Stands in for any OpenAI-compatible server."""

    def __init__(self, response=None) -> None:
        self.calls: list[dict[str, object]] = []
        self._response = response

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._response is not None:
            return self._response
        return SimpleNamespace(
            id="chatcmpl-1",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="local answer"),
                )
            ],
            usage=SimpleNamespace(prompt_tokens=110, completion_tokens=20),
        )


def _adapter(response=None, **kwargs):
    completions = FakeChatCompletions(response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return OpenAIChatCompletionsAdapter(client=client, **kwargs), completions


def test_adapter_maps_chat_completion_usage_onto_the_shared_fields() -> None:
    adapter, completions = _adapter()

    result = adapter("qwen3-4b-instruct", Task(kind="qa", payload={"prompt": "hello"}))

    assert result.output == "local answer"
    # Chat completions says prompt_tokens/completion_tokens, not input/output.
    assert result.input_tokens == 110
    assert result.output_tokens == 20
    assert result.metadata == {"response_id": "chatcmpl-1", "finish_reason": "stop"}
    assert completions.calls == [
        {
            "model": "qwen3-4b-instruct",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1024,
        }
    ]


def test_system_prompt_is_prepended_when_configured() -> None:
    adapter, completions = _adapter(system_prompt="Answer in one word.", max_tokens=64)

    adapter("m", Task(kind="qa", payload={"prompt": "hi"}))

    assert completions.calls[0]["messages"] == [
        {"role": "system", "content": "Answer in one word."},
        {"role": "user", "content": "hi"},
    ]
    assert completions.calls[0]["max_tokens"] == 64


def test_missing_usage_block_degrades_to_zero_tokens() -> None:
    """Local runtimes frequently omit usage entirely; that must not crash."""
    adapter, _ = _adapter(
        SimpleNamespace(
            id=None,
            choices=[SimpleNamespace(finish_reason=None, message=SimpleNamespace(content="x"))],
            usage=None,
        )
    )

    result = adapter("m", Task(kind="qa", payload={"prompt": "hi"}))

    assert result.output == "x"
    assert (result.input_tokens, result.output_tokens) == (0, 0)


def test_empty_choices_yields_empty_output_rather_than_raising() -> None:
    adapter, _ = _adapter(SimpleNamespace(id="c", choices=[], usage=None))

    result = adapter("m", Task(kind="qa", payload={"prompt": "hi"}))

    assert result.output == ""


def test_refusal_with_null_content_yields_empty_output() -> None:
    adapter, _ = _adapter(
        SimpleNamespace(
            id="c",
            choices=[
                SimpleNamespace(
                    finish_reason="content_filter",
                    message=SimpleNamespace(content=None),
                )
            ],
            usage=None,
        )
    )

    result = adapter("m", Task(kind="qa", payload={"prompt": "hi"}))

    assert result.output == ""


def test_from_env_reports_missing_optional_dependency(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "openai" or name.startswith("openai."):
            raise ImportError("no openai sdk")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"agent-router\[openai\]"):
        OpenAIChatCompletionsAdapter.from_env(base_url="http://127.0.0.1:9379/v1")


# --- routing behaviour of a zero-cost local model -----------------------------
#
# A free model is ALWAYS the cheapest eligible candidate, so cost ranking can never
# gate it. Only context window, reliability floor, and capabilities can. These tests
# pin that, because getting it wrong sends every task to a 4B model.

LOCAL_AND_CLOUD = {
    "version": "1",
    "pricing_as_of": "2026-08-23",
    "models": [
        {
            "name": "qwen3-4b-instruct",
            "provider": "local",
            "execution_classes": ["light_reasoning"],
            "capabilities": ["semantic_reasoning"],
            "reliability": 0.75,
            "context_window": 4096,
            "pricing": {"input_per_million": 0.0, "output_per_million": 0.0},
        },
        {
            "name": "cloud-cheap",
            "provider": "google",
            "execution_classes": ["light_reasoning", "deep_reasoning"],
            "capabilities": [
                "semantic_reasoning",
                "long_context",
                "deep_planning",
                "high_reliability",
            ],
            "reliability": 0.90,
            "context_window": 1000000,
            "pricing": {"input_per_million": 0.30, "output_per_million": 2.50},
        },
    ],
}


def _task(input_tokens: int, output_tokens: int, requirements=None):
    return Task(
        kind="subtask",
        payload={"prompt": "x"},
        requirements=requirements or {Requirement.SEMANTIC_REASONING},
        risk=Risk.LOW,
        metadata={
            "estimated_input_tokens": input_tokens,
            "estimated_output_tokens": output_tokens,
        },
    )


def _registry():
    return parse_catalog(LOCAL_AND_CLOUD).registry()


def test_free_local_model_wins_when_it_is_eligible() -> None:
    decision = plan_delegation(_task(1000, 200), registry=_registry(), threshold_tokens=100)

    assert decision.delegate is True
    assert (decision.provider, decision.model) == ("local", "qwen3-4b-instruct")
    assert decision.estimated_cost_usd == 0.0


def test_oversized_task_falls_through_to_the_cloud_model() -> None:
    """The context window is what stops a big prompt reaching a runtime that would
    break its HTTP response rather than refuse cleanly."""
    decision = plan_delegation(_task(9000, 2000), registry=_registry(), threshold_tokens=100)

    assert decision.delegate is True
    assert (decision.provider, decision.model) == ("google", "cloud-cheap")


def test_reliability_floor_excludes_the_free_model_despite_zero_cost() -> None:
    from agent_router.adaptive import AdaptivePolicy, PolicyMode

    decision = plan_delegation(
        _task(1000, 200),
        registry=_registry(),
        adaptive_policy=AdaptivePolicy(PolicyMode.BALANCED),
        threshold_tokens=100,
    )

    assert decision.reliability_floor >= 0.82
    assert decision.model == "cloud-cheap"


def test_long_context_escalates_past_the_free_model() -> None:
    """LONG_CONTEXT routes to deep_reasoning, which the local entry does not serve, so
    the free model is excluded by execution class before cost is ever considered."""
    decision = plan_delegation(
        _task(1000, 200, requirements={Requirement.SEMANTIC_REASONING, Requirement.LONG_CONTEXT}),
        registry=_registry(),
        threshold_tokens=100,
    )

    assert decision.execution_class.value == "deep_reasoning"
    assert (decision.provider, decision.model) == ("google", "cloud-cheap")


def test_local_catalog_round_trips_through_json() -> None:
    """The catalog shape documented for local runtimes must actually parse."""
    catalog = parse_catalog(json.loads(json.dumps(LOCAL_AND_CLOUD)))

    profile = catalog.registry().get("qwen3-4b-instruct")
    assert profile.provider == "local"
    assert profile.estimate_cost(input_tokens=100_000, output_tokens=100_000) == 0.0
