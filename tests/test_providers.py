from enum import Enum
from types import SimpleNamespace

import pytest

from agent_router import (
    AnthropicMessagesAdapter,
    GeminiAdapter,
    OpenAIResponsesAdapter,
    ProviderInvoker,
    Task,
    UnknownProvider,
)


class FakeOpenAIResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="resp_1",
            output_text="openai answer",
            usage=SimpleNamespace(input_tokens=120, output_tokens=30),
        )


class FakeAnthropicMessages:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="msg_1",
            stop_reason="end_turn",
            content=[
                SimpleNamespace(type="text", text="anthropic "),
                SimpleNamespace(type="text", text="answer"),
            ],
            usage=SimpleNamespace(input_tokens=100, output_tokens=25),
        )


class FakeGeminiModels:
    def __init__(self, response=None) -> None:
        self.calls: list[dict[str, object]] = []
        self._response = response

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self._response is not None:
            return self._response
        return SimpleNamespace(
            response_id="gen_1",
            text="gemini answer",
            candidates=[SimpleNamespace(finish_reason="STOP")],
            usage_metadata=SimpleNamespace(
                prompt_token_count=140,
                candidates_token_count=35,
            ),
        )


def test_openai_adapter_uses_responses_api_and_disables_storage() -> None:
    responses = FakeOpenAIResponses()
    adapter = OpenAIResponsesAdapter(client=SimpleNamespace(responses=responses))

    result = adapter("model-a", Task(kind="qa", payload={"prompt": "hello"}))

    assert result.output == "openai answer"
    assert result.input_tokens == 120
    assert result.output_tokens == 30
    assert result.metadata == {"response_id": "resp_1"}
    assert responses.calls == [
        {"model": "model-a", "input": "hello", "store": False}
    ]


def test_anthropic_adapter_uses_messages_api_and_collects_text_blocks() -> None:
    messages = FakeAnthropicMessages()
    adapter = AnthropicMessagesAdapter(
        client=SimpleNamespace(messages=messages),
        max_tokens=512,
    )

    result = adapter("model-b", Task(kind="qa", payload={"question": "why?"}))

    assert result.output == "anthropic answer"
    assert result.input_tokens == 100
    assert result.output_tokens == 25
    assert result.metadata == {"response_id": "msg_1", "stop_reason": "end_turn"}
    assert messages.calls == [
        {
            "model": "model-b",
            "max_tokens": 512,
            "messages": [{"role": "user", "content": "why?"}],
        }
    ]


def test_gemini_adapter_uses_generate_content_and_normalizes_usage() -> None:
    models = FakeGeminiModels()
    adapter = GeminiAdapter(client=SimpleNamespace(models=models))

    result = adapter("model-c", Task(kind="qa", payload={"prompt": "how?"}))

    assert result.output == "gemini answer"
    assert result.input_tokens == 140
    assert result.output_tokens == 35
    assert result.metadata == {"response_id": "gen_1", "finish_reason": "STOP"}
    assert models.calls == [{"model": "model-c", "contents": "how?"}]


def test_gemini_adapter_unwraps_enum_finish_reason() -> None:
    """The real SDK enum subclasses str, so it must be unwrapped, not passed through."""

    class FinishReason(str, Enum):  # noqa: UP042 - mirrors the SDK's str-mixin enum
        STOP = "STOP"

    models = FakeGeminiModels(
        SimpleNamespace(
            response_id="gen_2",
            text="answer",
            candidates=[SimpleNamespace(finish_reason=FinishReason.STOP)],
            usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=2),
        )
    )
    adapter = GeminiAdapter(client=SimpleNamespace(models=models))

    result = adapter("model-c", Task(kind="qa", payload={"prompt": "x"}))

    assert result.metadata is not None
    finish_reason = result.metadata["finish_reason"]
    assert finish_reason == "STOP"
    assert type(finish_reason) is str, "enum must be unwrapped to a plain str"
    assert str(finish_reason) == "STOP"


def test_gemini_adapter_tolerates_blocked_response_without_text_or_usage() -> None:
    models = FakeGeminiModels(
        SimpleNamespace(
            response_id="gen_3",
            text=None,
            candidates=[],
            usage_metadata=None,
        )
    )
    adapter = GeminiAdapter(client=SimpleNamespace(models=models))

    result = adapter("model-c", Task(kind="qa", payload={"prompt": "x"}))

    assert result.output == ""
    assert result.input_tokens == 0
    assert result.output_tokens == 0
    assert result.metadata == {"response_id": "gen_3", "finish_reason": None}


def test_gemini_adapter_from_env_reports_missing_optional_dependency(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "google" or name.startswith("google."):
            raise ImportError("no google sdk")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"agent-router\[google\]"):
        GeminiAdapter.from_env()


def test_provider_invoker_dispatches_by_provider() -> None:
    calls: list[tuple[str, str]] = []

    def adapter(model: str, task: Task):
        calls.append((model, task.kind))
        return OpenAIResponsesAdapter(
            client=SimpleNamespace(responses=FakeOpenAIResponses())
        )(model, task)

    invoker = ProviderInvoker({"openai": adapter})
    result = invoker("openai", "model-a", Task(kind="qa", payload={"prompt": "x"}))

    assert result.output == "openai answer"
    assert calls == [("model-a", "qa")]


def test_provider_invoker_rejects_unknown_provider() -> None:
    invoker = ProviderInvoker()

    with pytest.raises(UnknownProvider):
        invoker("missing", "model", Task(kind="qa", payload={}))
