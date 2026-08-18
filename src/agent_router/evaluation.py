from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from statistics import mean


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    id: str
    task_kind: str
    minimum_quality: float = 1.0
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("evaluation case id must be non-empty")
        if not 0.0 <= self.minimum_quality <= 1.0:
            raise ValueError("minimum_quality must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    case_id: str
    strategy: str
    model: str | None
    quality: float
    cost_usd: float
    latency_seconds: float
    escalations: int = 0
    success: bool | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError("quality must be between 0 and 1")
        if self.cost_usd < 0 or self.latency_seconds < 0 or self.escalations < 0:
            raise ValueError("cost, latency, and escalations must be non-negative")


@dataclass(frozen=True, slots=True)
class StrategySummary:
    strategy: str
    runs: int
    success_rate: float
    mean_quality: float
    total_cost_usd: float
    mean_cost_usd: float
    mean_latency_seconds: float
    escalation_rate: float


@dataclass(frozen=True, slots=True)
class StrategyComparison:
    strategy: StrategySummary
    baseline: StrategySummary
    cost_savings_fraction: float
    quality_delta: float
    success_rate_delta: float
    latency_delta_seconds: float


def summarize_strategy(
    cases: Iterable[EvaluationCase],
    runs: Iterable[EvaluationRun],
    *,
    strategy: str,
) -> StrategySummary:
    case_map = {case.id: case for case in cases}
    selected = [run for run in runs if run.strategy == strategy]
    if not selected:
        raise ValueError(f"no evaluation runs for strategy {strategy!r}")

    successes: list[bool] = []
    for run in selected:
        case = case_map.get(run.case_id)
        if case is None:
            raise ValueError(f"run references unknown case {run.case_id!r}")
        successes.append(
            run.success if run.success is not None else run.quality >= case.minimum_quality
        )

    total_cost = sum(run.cost_usd for run in selected)
    return StrategySummary(
        strategy=strategy,
        runs=len(selected),
        success_rate=sum(successes) / len(successes),
        mean_quality=mean(run.quality for run in selected),
        total_cost_usd=total_cost,
        mean_cost_usd=total_cost / len(selected),
        mean_latency_seconds=mean(run.latency_seconds for run in selected),
        escalation_rate=sum(run.escalations > 0 for run in selected) / len(selected),
    )


def compare_strategies(
    strategy: StrategySummary,
    baseline: StrategySummary,
) -> StrategyComparison:
    if baseline.total_cost_usd == 0:
        cost_savings = 0.0 if strategy.total_cost_usd == 0 else float("-inf")
    else:
        cost_savings = 1.0 - strategy.total_cost_usd / baseline.total_cost_usd

    return StrategyComparison(
        strategy=strategy,
        baseline=baseline,
        cost_savings_fraction=cost_savings,
        quality_delta=strategy.mean_quality - baseline.mean_quality,
        success_rate_delta=strategy.success_rate - baseline.success_rate,
        latency_delta_seconds=strategy.mean_latency_seconds - baseline.mean_latency_seconds,
    )


def evaluate_gate(
    comparison: StrategyComparison,
    *,
    minimum_cost_savings: float = 0.0,
    maximum_quality_loss: float = 0.0,
    maximum_success_rate_loss: float = 0.0,
) -> tuple[bool, tuple[str, ...]]:
    failures: list[str] = []
    if comparison.cost_savings_fraction < minimum_cost_savings:
        failures.append(
            f"cost savings {comparison.cost_savings_fraction:.3f} below required "
            f"{minimum_cost_savings:.3f}"
        )
    if comparison.quality_delta < -maximum_quality_loss:
        failures.append(
            f"quality delta {comparison.quality_delta:.3f} exceeds allowed loss "
            f"{maximum_quality_loss:.3f}"
        )
    if comparison.success_rate_delta < -maximum_success_rate_loss:
        failures.append(
            f"success-rate delta {comparison.success_rate_delta:.3f} exceeds allowed loss "
            f"{maximum_success_rate_loss:.3f}"
        )
    return not failures, tuple(failures)
