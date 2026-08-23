from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .model_executor import ModelResponse, UnknownProvider
from .types import Task

PromptBuilder = Callable[[Task], str]

__all__ = [
    "AnthropicMessagesAdapter",
    "GeminiAdapter",
    "OpenAIChatCompletionsAdapter",
    "OpenAIResponsesAdapter",
    "ProviderAdapter",
    "ProviderInvoker",
    "UnknownProvider",
    "default_prompt_builder",
]


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


@dataclass(slots=True)
class GeminiAdapter:
    """Thin adapter for the Google Gemini API with an injected SDK client."""

    client: Any
    prompt_builder: PromptBuilder = default_prompt_builder

    @classmethod
    def from_env(
        cls,
        *,
        prompt_builder: PromptBuilder = default_prompt_builder,
        **client_kwargs: Any,
    ) -> GeminiAdapter:
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "Gemini adapter requires the optional 'google' dependency; "
                "install agent-router[google]"
            ) from exc
        return cls(
            client=genai.Client(**client_kwargs),
            prompt_builder=prompt_builder,
        )

    def __call__(self, model: str, task: Task) -> ModelResponse:
        response = self.client.models.generate_content(
            model=model,
            contents=self.prompt_builder(task),
        )
        usage = getattr(response, "usage_metadata", None)
        text = getattr(response, "text", "")
        return ModelResponse(
            output=text if isinstance(text, str) else "",
            input_tokens=_usage_value(usage, "prompt_token_count"),
            output_tokens=_usage_value(usage, "candidates_token_count"),
            metadata={
                "response_id": getattr(response, "response_id", None),
                "finish_reason": _gemini_finish_reason(getattr(response, "candidates", ())),
            },
        )


@dataclass(slots=True)
class OpenAIChatCompletionsAdapter:
    """Adapter for any server speaking the OpenAI chat-completions API.

    Distinct from :class:`OpenAIResponsesAdapter`, which uses the newer Responses API
    that self-hosted servers generally do not implement. Pointing ``base_url`` at a
    local runtime -- LiteRT-LM, Ollama, llama.cpp, vLLM, LM Studio -- routes work to it
    through the same contract as a hosted provider.

    Local servers commonly omit ``usage``; token counts then fall back to 0, which is
    correct for a zero-priced catalog entry but means telemetry carries no token counts.
    """

    client: Any
    prompt_builder: PromptBuilder = default_prompt_builder
    max_tokens: int = 1024
    system_prompt: str | None = None

    @classmethod
    def from_env(
        cls,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        prompt_builder: PromptBuilder = default_prompt_builder,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
        **client_kwargs: Any,
    ) -> OpenAIChatCompletionsAdapter:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI-compatible chat adapter requires the optional 'openai' dependency; "
                "install agent-router[openai]"
            ) from exc

        if base_url is not None:
            client_kwargs["base_url"] = base_url
            # A local runtime authenticates nobody, but the SDK still requires a key to
            # construct. Only substitute a placeholder when the environment has none, so
            # a real key is never silently overridden.
            if api_key is None and not os.environ.get("OPENAI_API_KEY"):
                api_key = "not-required-by-local-server"
        if api_key is not None:
            client_kwargs["api_key"] = api_key

        return cls(
            client=OpenAI(**client_kwargs),
            prompt_builder=prompt_builder,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )

    def __call__(self, model: str, task: Task) -> ModelResponse:
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.prompt_builder(task)})

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=self.max_tokens,
        )
        choice = next(iter(getattr(response, "choices", ()) or ()), None)
        usage = getattr(response, "usage", None)
        return ModelResponse(
            output=_chat_text(choice),
            # Chat completions names these differently from the Responses API.
            input_tokens=_usage_value(usage, "prompt_tokens"),
            output_tokens=_usage_value(usage, "completion_tokens"),
            metadata={
                "response_id": getattr(response, "id", None),
                "finish_reason": getattr(choice, "finish_reason", None),
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


def _gemini_finish_reason(candidates: Any) -> str | None:
    """Normalize Gemini's ``FinishReason`` into a plain string.

    The SDK enum subclasses ``str``, so it compares equal to its value but still
    renders as ``FinishReason.STOP``. Unwrapping ``.value`` first keeps persisted
    result metadata predictable.
    """
    candidate = next(iter(candidates or ()), None)
    reason = getattr(candidate, "finish_reason", None)
    if reason is None:
        return None
    value = getattr(reason, "value", reason)
    return value if type(value) is str else str(value)


def _chat_text(choice: Any) -> str:
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None)
    return content if isinstance(content, str) else ""
