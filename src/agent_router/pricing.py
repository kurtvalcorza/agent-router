from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PricingProfile:
    """Provider pricing dimensions expressed as USD per million tokens."""

    standard_input: float = 0.0
    standard_output: float = 0.0
    cached_input: float | None = None
    cache_write: float | None = None
    batch_input: float | None = None
    batch_output: float | None = None
    long_context_input: float | None = None
    long_context_output: float | None = None
    long_context_threshold: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "standard_input",
            "standard_output",
            "cached_input",
            "cache_write",
            "batch_input",
            "batch_output",
            "long_context_input",
            "long_context_output",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.long_context_threshold is not None and self.long_context_threshold < 1:
            raise ValueError("long_context_threshold must be positive")

    def estimate(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
        cache_write_tokens: int = 0,
        batch: bool = False,
    ) -> float:
        if min(input_tokens, output_tokens, cached_input_tokens, cache_write_tokens) < 0:
            raise ValueError("token counts must be non-negative")
        if cached_input_tokens + cache_write_tokens > input_tokens:
            raise ValueError("cached/cache-write tokens cannot exceed input tokens")

        ordinary_input = input_tokens - cached_input_tokens - cache_write_tokens
        input_rate = self.standard_input
        output_rate = self.standard_output

        if (
            self.long_context_threshold is not None
            and input_tokens > self.long_context_threshold
        ):
            input_rate = self.long_context_input or input_rate
            output_rate = self.long_context_output or output_rate

        if batch:
            input_rate = self.batch_input if self.batch_input is not None else input_rate
            output_rate = self.batch_output if self.batch_output is not None else output_rate

        cached_rate = self.cached_input if self.cached_input is not None else input_rate
        cache_write_rate = self.cache_write if self.cache_write is not None else input_rate

        return (
            ordinary_input * input_rate
            + cached_input_tokens * cached_rate
            + cache_write_tokens * cache_write_rate
            + output_tokens * output_rate
        ) / 1_000_000
