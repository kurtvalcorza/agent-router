from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from .benchmark_runtime import case_to_task
from .evaluation import EvaluationCase, EvaluationRun
from .models import ModelProfile, ModelRegistry, NoEligibleModel
from .types import ExecutionClass, Task


@dataclass(frozen=True, slots=True)
class SuccessEstimate:
    model: str
    feature_key: str
    successes: int
    trials: int
    probability: float


@dataclass(frozen=True, slots=True)
class EmpiricalSelection:
    profile: ModelProfile
    success_probability: float
    estimated_call_cost: float
    expected_total_cost: float
    feature_key: str


class EmpiricalSuccessModel:
    """Estimate P(success | task features, model) with hierarchical Beta smoothing."""

    def __init__(
        self,
        *,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        feature_weight: float = 3.0,
    ) -> None:
        if prior_alpha <= 0 or prior_beta <= 0 or feature_weight < 0:
            raise ValueError("invalid empirical prior parameters")
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.feature_weight = feature_weight
        self._global: dict[str, tuple[int, int]] = {}
        self._feature: dict[tuple[str, str], tuple[int, int]] = {}

    @classmethod
    def fit(
        cls,
        cases: Iterable[EvaluationCase],
        runs: Iterable[EvaluationRun],
        *,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        feature_weight: float = 3.0,
    ) -> EmpiricalSuccessModel:
        model = cls(
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            feature_weight=feature_weight,
        )
        case_map = {case.id: case for case in cases}
        global_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        feature_counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])

        for run in runs:
            if not run.model:
                continue
            case = case_map.get(run.case_id)
            if case is None:
                raise ValueError(f"run references unknown case {run.case_id!r}")
            success = (
                run.success
                if run.success is not None
                else run.quality >= case.minimum_quality
            )
            key = task_feature_key(case_to_task(case))
            global_counts[run.model][1] += 1
            feature_counts[(run.model, key)][1] += 1
            if success:
                global_counts[run.model][0] += 1
                feature_counts[(run.model, key)][0] += 1

        model._global = {
            name: tuple(values) for name, values in global_counts.items()
        }
        model._feature = {
            key: tuple(values) for key, values in feature_counts.items()
        }
        return model

    def estimate(self, model: str, task: Task) -> SuccessEstimate:
        feature_key = task_feature_key(task)
        global_successes, global_trials = self._global.get(model, (0, 0))
        global_probability = (self.prior_alpha + global_successes) / (
            self.prior_alpha + self.prior_beta + global_trials
        )

        successes, trials = self._feature.get((model, feature_key), (0, 0))
        # Hierarchical Beta: the feature prior is the model's global rate, expressed as
        # ``feature_weight`` pseudo-observations. The raw prior is NOT re-added here — it is
        # already folded into ``global_probability`` above. With no feature data the estimate
        # equals ``global_probability`` (shrink toward the global rate), not 0.5.
        alpha = self.feature_weight * global_probability + successes
        beta = self.feature_weight * (1.0 - global_probability) + (trials - successes)
        denominator = alpha + beta
        probability = alpha / denominator if denominator > 0 else global_probability
        return SuccessEstimate(
            model=model,
            feature_key=feature_key,
            successes=successes,
            trials=trials,
            probability=probability,
        )


def task_feature_key(task: Task) -> str:
    requirements = ",".join(sorted(value.value for value in task.requirements)) or "none"
    return f"kind={task.kind}|risk={task.risk.value}|requirements={requirements}"


class EmpiricalSelector:
    """Choose the lowest expected-cost eligible model that clears a reliability floor."""

    def __init__(
        self,
        *,
        registry: ModelRegistry,
        success_model: EmpiricalSuccessModel,
        recovery_cost_multiplier: float = 1.0,
    ) -> None:
        if recovery_cost_multiplier < 0:
            raise ValueError("recovery_cost_multiplier must be non-negative")
        self.registry = registry
        self.success_model = success_model
        self.recovery_cost_multiplier = recovery_cost_multiplier

    def ranked(
        self,
        task: Task,
        execution_class: ExecutionClass,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        min_success_probability: float = 0.0,
        max_estimated_cost: float | None = None,
    ) -> list[EmpiricalSelection]:
        if not 0.0 <= min_success_probability <= 1.0:
            raise ValueError("min_success_probability must be between 0 and 1")

        candidates: list[EmpiricalSelection] = []
        eligible = self.registry.eligible(task, execution_class)
        if not eligible:
            return []

        fallback_cost = max(
            (
                profile.estimate_cost(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
                for profile in eligible
            ),
            default=0.0,
        )

        for profile in eligible:
            estimate = self.success_model.estimate(profile.name, task)
            if estimate.probability < min_success_probability:
                continue
            call_cost = profile.estimate_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            expected_total = call_cost + (
                (1.0 - estimate.probability)
                * fallback_cost
                * self.recovery_cost_multiplier
            )
            if max_estimated_cost is not None and expected_total > max_estimated_cost:
                continue
            candidates.append(
                EmpiricalSelection(
                    profile=profile,
                    success_probability=estimate.probability,
                    estimated_call_cost=call_cost,
                    expected_total_cost=expected_total,
                    feature_key=estimate.feature_key,
                )
            )

        return sorted(
            candidates,
            key=lambda item: (
                item.expected_total_cost,
                -item.success_probability,
                item.profile.name,
            ),
        )

    def select(self, *args, **kwargs) -> EmpiricalSelection:
        candidates = self.ranked(*args, **kwargs)
        if not candidates:
            raise NoEligibleModel("no empirically eligible model")
        return candidates[0]
