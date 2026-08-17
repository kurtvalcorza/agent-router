from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from time import perf_counter

from .evaluation import EvaluationCase, EvaluationRun


@dataclass(frozen=True, slots=True)
class StrategyResult:
    model: str | None
    quality: float
    cost_usd: float
    escalations: int = 0
    success: bool | None = None
    metadata: dict[str, object] | None = None


StrategyExecutor = Callable[[EvaluationCase], StrategyResult]


def run_strategy(
    cases: Iterable[EvaluationCase],
    *,
    strategy: str,
    execute: StrategyExecutor,
) -> tuple[EvaluationRun, ...]:
    if not strategy:
        raise ValueError("strategy must be non-empty")

    runs: list[EvaluationRun] = []
    for case in cases:
        started = perf_counter()
        result = execute(case)
        elapsed = perf_counter() - started
        runs.append(
            EvaluationRun(
                case_id=case.id,
                strategy=strategy,
                model=result.model,
                quality=result.quality,
                cost_usd=result.cost_usd,
                latency_seconds=elapsed,
                escalations=result.escalations,
                success=result.success,
                metadata=dict(result.metadata or {}),
            )
        )
    return tuple(runs)
