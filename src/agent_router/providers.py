from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .model_executor import ModelResponse
from .types import Task

PromptBuilder = Callable[[Task], str]


class ProviderAdapter(Protocol):
    def __call__(self, model: str, task: Task) -> ModelResponse: ...


def default_prompt_builder(task: Task) -> str:
    prompt = task.payload.get("prompt")
    if isinstance(prompt, str):
        return prompt
    question = task.payload.get("question")
    if isinstance(question, str):
        return question
    return str(task.payload)


class UnknownProvider(KeyError):
    pass


class ProviderInvoker:
    """Dispatch model invocations to provider-specific adapters."""

    def __init__(self, adapters: Mapping[str, ProviderAdapter] | None = None) -> None:
        self._adapters = dict(adapters or {})

    def register(self, provider: str, adapter: ProviderAdapter) -> None:
        self._adapters[provider] = adapter

    def __call__(self, provider: str, model: str, task: Task) -> ModelResponse:
        try:
            adapter = self._adapters[provider]
        except KeyError as exc:
            raise UnknownProvider(provider) from exc
        return adapter(model, task)


@dataclass(slots=True)
class OpenAIResponsesAdapter:
    """Thin adapter for the OpenAI Responses API.

    The client is injected to keep the OpenAI SDK optional. ``from_env`` imports and
    constructs the official SDK client when the ``openai`` extra is installed.
    """

    client: Any
    prompt_builder: PromptBuilder = default_prompt_builder
    store: bool = False

    @classmethod
    def from_env(
        cls,
        *,
        prompt_builder: PromptBuilder = default_prompt_builder,
        store: bool = False,
        **client_kwargs: Any,
    ) -> OpenAIResponsesAdapter:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI adapter requires the optional 'openai' dependency; "
                "install agent-router[openai]"
            ) from exc
        return cls(
            client=OpenAI(**client_kwargs),
            prompt_builder=prompt_builder,
            store=store,
        )

    def __call__(self, model: str, task: Task) -> ModelResponse:
        response = self.client.responses.create(
            model=model,
            input=self.prompt_builder(task),
            store=self.store,
        )
        usage = getattr(response, "usage", None)
        return ModelResponse(
            output=getattr(response, "output_text", ""),
            input_tokens=_usage_value(usage, "input_tokens"),
            output_tokens=_usage_value(usage, "output_tokens"),
            metadata={"response_id": getattr(response, "id", None)},
        )


@dataclass(slots=True)
class AnthropicMessagesAdapter:
    """Thin adapter for Anthropic's Messages API with an injected SDK client."""

    client: Any
    prompt_builder: PromptBuilder = default_prompt_builder
    max_tokens: int = 1024

    @classmethod
    def from_env(
        cls,
        *,
        prompt_builder: PromptBuilder = default_prompt_builder,
        max_tokens: int = 1024,
        **client_kwargs: Any,
    ) -> AnthropicMessagesAdapter:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError(
                "Anthropic adapter requires the optional 'anthropic' dependency; "
                "install agent-router[anthropic]"
            ) from exc
        return cls(
            client=Anthropic(**client_kwargs),
            prompt_builder=prompt_builder,
            max_tokens=max_tokens,
        )

    def __call__(self, model: str, task: Task) -> ModelResponse:
        response = self.client.messages.create(
            model=model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": self.prompt_builder(task)}],
        )
        usage = getattr(response, "usage", None)
        return ModelResponse(
            output=_anthropic_text(getattr(response, "content", ())),
            input_tokens=_usage_value(usage, "input_tokens"),
            output_tokens=_usage_value(usage, "output_tokens"),
            metadata={
                "response_id": getattr(response, "id", None),
                "stop_reason": getattr(response, "stop_reason", None),
            },
        )


def _usage_value(usage: Any, name: str) -> int:
    value = getattr(usage, name, 0) if usage is not None else 0
    return value if isinstance(value, int) else 0


def _anthropic_text(content: Any) -> str:
    parts: list[str] = []
    for block in content or ():
        if getattr(block, "type", None) == "text":
            text = getattr(block, "text", "")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)
