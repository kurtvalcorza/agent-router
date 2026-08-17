from types import SimpleNamespace

import pytest

from agent_router import (
    AnthropicMessagesAdapter,
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
