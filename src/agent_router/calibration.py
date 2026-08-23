"""Propose catalog ``reliability`` values from benchmark evidence.

``reliability`` does real policy work: it decides whether a model is eligible for a
policy tier at all, so a guessed value silently encodes routing behaviour. This module
turns benchmark runs into an *evidence-backed proposal* for that number.

It deliberately proposes and never applies. The catalog stays a reviewed policy artifact:

    benchmark runs -> empirical fit -> calibration proposal -> review -> catalog update

Two different questions are kept apart, and this module answers only the first:

* catalog ``reliability`` -- is this model trustworthy enough to enter this tier at all?
  An eligibility prior, model-level, changed rarely and under review.
* :class:`~agent_router.empirical.EmpiricalSuccessModel` -- given this task shape, how
  likely is this model to succeed relative to other eligible models? A task-conditional
  ranking signal, refit freely from run history.

Nothing here mutates a catalog, and nothing here writes a file.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from .adaptive import PolicyMode
from .benchmark_runtime import case_to_task
from .empirical import task_feature_key
from .evaluation import EvaluationCase, EvaluationRun

__all__ = [
    "CALIBRATION_METHOD",
    "CALIBRATION_METHOD_VERSION",
    "DEFAULT_CREDIBLE_LEVEL",
    "DEFAULT_MIN_TRIALS",
    "CalibrationParameters",
    "CalibrationProposal",
    "TaskClassEvidence",
    "calibrate_reliability",
]

CALIBRATION_METHOD = "beta-posterior-conservative"
# Bump when the aggregation rule changes, so an applied value can be traced to the rule
# that produced it. Stored alongside the value in catalog metadata.
CALIBRATION_METHOD_VERSION = "1"

DEFAULT_CREDIBLE_LEVEL = 0.90
DEFAULT_MIN_TRIALS = 20

# Policy floors that a proposal can move a model across. Crossing one changes which
# tiers may route to the model at all, which is the change a reviewer most needs to see.
_POLICY_FLOORS: tuple[tuple[str, float], ...] = (
    (PolicyMode.ECONOMY.value, 0.70),
    (PolicyMode.BALANCED.value, 0.82),
    (PolicyMode.QUALITY.value, 0.90),
    (PolicyMode.CRITICAL.value, 0.97),
)


@dataclass(frozen=True, slots=True)
class CalibrationParameters:
    """The knob settings a proposal was produced under.

    Recorded so a later comparison can tell a change in *evidence* from a change in
    *policy parameters*. Several of these are experimental values chosen by judgement
    rather than measurement -- ``min_trials`` and ``dominant_class_warning_share``
    especially -- so sensitivity analysis needs them attached to the run, not
    reconstructed afterwards from whatever the defaults happen to be by then.
    """

    credible_level: float
    min_trials: int
    prior_alpha: float
    prior_beta: float
    dominant_class_warning_share: float

    def as_dict(self) -> dict[str, object]:
        return {
            "credible_level": self.credible_level,
            "min_trials": self.min_trials,
            "prior_alpha": self.prior_alpha,
            "prior_beta": self.prior_beta,
            "dominant_class_warning_share": self.dominant_class_warning_share,
        }


@dataclass(frozen=True, slots=True)
class TaskClassEvidence:
    """Per-task-class counts, exposed so corpus skew is visible rather than averaged away."""

    task_class: str
    successes: int
    trials: int
    posterior_mean: float

    def as_dict(self) -> dict[str, object]:
        return {
            "task_class": self.task_class,
            "successes": self.successes,
            "trials": self.trials,
            "posterior_mean": round(self.posterior_mean, 6),
        }


@dataclass(frozen=True, slots=True)
class ThresholdCrossing:
    """A policy floor the proposal would move the model across."""

    mode: str
    floor: float
    eligible_before: bool
    eligible_after: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "floor": self.floor,
            "eligible_before": self.eligible_before,
            "eligible_after": self.eligible_after,
        }


@dataclass(frozen=True, slots=True)
class CalibrationProposal:
    """An evidence-backed proposal. Never applied automatically."""

    model: str
    current_reliability: float | None
    proposed_reliability: float
    successes: int
    trials: int
    posterior_mean: float
    credible_interval: tuple[float, float]
    pooled_lower_bound: float
    task_class_balanced_mean: float
    dominant_class_share: float
    coverage: tuple[TaskClassEvidence, ...]
    threshold_crossings: tuple[ThresholdCrossing, ...]
    evidence_ref: str
    status: str
    parameters: CalibrationParameters
    warnings: tuple[str, ...] = field(default_factory=tuple)
    method: str = CALIBRATION_METHOD
    method_version: str = CALIBRATION_METHOD_VERSION

    def as_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "current_reliability": self.current_reliability,
            "proposed_reliability": round(self.proposed_reliability, 6),
            "successes": self.successes,
            "trials": self.trials,
            "posterior_mean": round(self.posterior_mean, 6),
            "credible_interval": [
                round(self.credible_interval[0], 6),
                round(self.credible_interval[1], 6),
            ],
            "pooled_lower_bound": round(self.pooled_lower_bound, 6),
            "task_class_balanced_mean": round(self.task_class_balanced_mean, 6),
            "dominant_class_share": round(self.dominant_class_share, 6),
            "coverage": [item.as_dict() for item in self.coverage],
            "threshold_crossings": [item.as_dict() for item in self.threshold_crossings],
            "evidence_ref": self.evidence_ref,
            "status": self.status,
            "parameters": self.parameters.as_dict(),
            "warnings": list(self.warnings),
            "method": self.method,
            "method_version": self.method_version,
        }


def calibrate_reliability(
    cases: Iterable[EvaluationCase],
    runs: Iterable[EvaluationRun],
    *,
    current_reliability: Mapping[str, float] | None = None,
    evidence_ref: str,
    credible_level: float = DEFAULT_CREDIBLE_LEVEL,
    min_trials: int = DEFAULT_MIN_TRIALS,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    dominant_class_warning_share: float = 0.60,
) -> tuple[CalibrationProposal, ...]:
    """Derive one proposal per model observed in ``runs``.

    The aggregation is deliberately conservative. A benchmark skewed toward easy cases
    would otherwise promote a model across a policy floor on thin evidence, so the
    proposal is the lower of:

    * the lower bound of the pooled Beta posterior credible interval, and
    * the mean of the per-task-class posterior means, weighting each observed class
      equally so a class with many easy cases cannot dominate.

    Both inputs are reported, so the conservatism is inspectable rather than implicit.
    """
    if not 0.0 < credible_level < 1.0:
        raise ValueError("credible_level must be between 0 and 1")
    if min_trials < 0:
        raise ValueError("min_trials must be non-negative")
    if prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError("prior parameters must be positive")
    if not evidence_ref:
        raise ValueError("evidence_ref must be non-empty; a proposal without provenance "
                         "cannot be reviewed")

    parameters = CalibrationParameters(
        credible_level=credible_level,
        min_trials=min_trials,
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
        dominant_class_warning_share=dominant_class_warning_share,
    )

    case_map = {case.id: case for case in cases}
    pooled: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    per_class: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])

    for run in runs:
        if not run.model:
            continue
        case = case_map.get(run.case_id)
        if case is None:
            raise ValueError(f"run references unknown case {run.case_id!r}")
        # Same success rule as EmpiricalSuccessModel.fit, so calibration and the
        # task-conditional model never disagree about what counted as a success.
        success = run.success if run.success is not None else run.quality >= case.minimum_quality
        task_class = task_feature_key(case_to_task(case))

        pooled[run.model][1] += 1
        per_class[(run.model, task_class)][1] += 1
        if success:
            pooled[run.model][0] += 1
            per_class[(run.model, task_class)][0] += 1

    current = dict(current_reliability or {})
    tail = (1.0 - credible_level) / 2.0

    proposals: list[CalibrationProposal] = []
    for model in sorted(pooled):
        successes, trials = pooled[model]
        alpha = prior_alpha + successes
        beta = prior_beta + (trials - successes)

        posterior_mean = alpha / (alpha + beta)
        lower = _beta_quantile(alpha, beta, tail)
        upper = _beta_quantile(alpha, beta, 1.0 - tail)

        coverage = tuple(
            TaskClassEvidence(
                task_class=task_class,
                successes=class_successes,
                trials=class_trials,
                posterior_mean=(prior_alpha + class_successes)
                / (prior_alpha + prior_beta + class_trials),
            )
            for (owner, task_class), (
                class_successes,
                class_trials,
            ) in sorted(per_class.items())
            if owner == model
        )
        balanced = (
            sum(item.posterior_mean for item in coverage) / len(coverage)
            if coverage
            else posterior_mean
        )
        dominant_share = (
            max(item.trials for item in coverage) / trials if coverage and trials else 1.0
        )

        proposed = min(lower, balanced)
        before = current.get(model)

        warnings: list[str] = []
        if trials < min_trials:
            warnings.append(
                f"only {trials} trial(s); below the {min_trials}-trial minimum for a "
                "reviewable proposal"
            )
        if len(coverage) < 2:
            warnings.append(
                "evidence covers a single task class; the proposal says nothing about "
                "behaviour on task shapes not benchmarked"
            )
        elif dominant_share >= dominant_class_warning_share:
            warnings.append(
                f"{dominant_share:.0%} of trials come from one task class; the corpus is "
                "skewed and the balanced estimate carries that skew"
            )
        if before is not None and proposed > before:
            warnings.append(
                "proposal would RAISE reliability, widening the tiers this model may "
                "enter; confirm the corpus is representative before accepting"
            )

        proposals.append(
            CalibrationProposal(
                model=model,
                current_reliability=before,
                proposed_reliability=proposed,
                successes=successes,
                trials=trials,
                posterior_mean=posterior_mean,
                credible_interval=(lower, upper),
                pooled_lower_bound=lower,
                task_class_balanced_mean=balanced,
                dominant_class_share=dominant_share,
                coverage=coverage,
                threshold_crossings=_crossings(before, proposed),
                evidence_ref=evidence_ref,
                status="INSUFFICIENT_EVIDENCE" if trials < min_trials else "REVIEW_REQUIRED",
                parameters=parameters,
                warnings=tuple(warnings),
            )
        )
    return tuple(proposals)


def _crossings(before: float | None, after: float) -> tuple[ThresholdCrossing, ...]:
    if before is None:
        return ()
    return tuple(
        ThresholdCrossing(
            mode=mode,
            floor=floor,
            eligible_before=before >= floor,
            eligible_after=after >= floor,
        )
        for mode, floor in _POLICY_FLOORS
        if (before >= floor) != (after >= floor)
    )


# --- Beta posterior maths (no third-party dependency) -------------------------


def _beta_quantile(alpha: float, beta: float, probability: float) -> float:
    """Inverse Beta CDF by bisection on the regularized incomplete beta function.

    Bisection rather than Newton because the derivative vanishes at the boundaries for
    small shape parameters; 200 halvings reach far below the precision anyone reads off
    a reliability value.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    if probability in (0.0, 1.0):
        return probability

    low, high = 0.0, 1.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if _regularized_incomplete_beta(alpha, beta, middle) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """I_x(a, b) -- the Beta CDF."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    # The continued fraction converges quickly only on this side of the symmetry point.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Lentz's algorithm for the continued fraction of the incomplete beta function."""
    tiny = 1e-30
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d

    for m in range(1, 300):
        two_m = 2 * m

        numerator = m * (b - m) * x / ((a + two_m - 1.0) * (a + two_m))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c

        numerator = -(a + m) * (a + b + m) * x / ((a + two_m) * (a + two_m + 1.0))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta

        if abs(delta - 1.0) < 1e-12:
            break
    return result
