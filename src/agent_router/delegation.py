"""Host-agnostic delegation decisions for agents that shell out to the router.

An agent asking "should I hand this subtask to a cheaper model?" pays its own tokens
to ask. Below a certain size the asking costs more than the routing saves, so the
threshold check happens here -- in the callee, once -- instead of being re-derived by
every host that calls it.

Nothing in this module invokes a provider. It answers *which* model and *whether it is
worth it*; execution stays with the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .adaptive import AdaptivePolicy
from .models import ModelProfile, ModelRegistry
from .policy import RoutingPolicy
from .types import ExecutionClass, Requirement, Risk, Task

__all__ = [
    "DEFAULT_DELEGATION_THRESHOLD_TOKENS",
    "DelegationDecision",
    "estimate_tokens",
    "parse_requirements",
    "parse_risk",
    "plan_delegation",
]

# Below this much estimated work, a frontier host answering directly generally beats
# the round trip: the host still pays to compose the call and read the result, and a
# provider call has its own fixed latency. Tuned as a starting point, not a measurement
# -- override it per workload with the threshold argument.
DEFAULT_DELEGATION_THRESHOLD_TOKENS = 400

# Rough character-per-token ratio for English prose. Deliberately crude: this only has
# to be good enough to rank a task against the threshold and pick a cost bracket.
_CHARS_PER_TOKEN = 4


@dataclass(frozen=True, slots=True)
class DelegationDecision:
    """Why a task should or should not go to a cheaper model, and which one."""

    delegate: bool
    reason: str
    execution_class: ExecutionClass
    routing_reason: str
    reliability_floor: float
    estimated_input_tokens: int
    estimated_output_tokens: int
    provider: str | None = None
    model: str | None = None
    estimated_cost_usd: float | None = None
    alternatives: tuple[tuple[str, str, float], ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        return {
            "delegate": self.delegate,
            "reason": self.reason,
            "execution_class": self.execution_class.value,
            "routing_reason": self.routing_reason,
            "reliability_floor": self.reliability_floor,
            "estimated_input_tokens": self.estimated_input_tokens,
            "estimated_output_tokens": self.estimated_output_tokens,
            "provider": self.provider,
            "model": self.model,
            "estimated_cost_usd": self.estimated_cost_usd,
            "alternatives": [
                {"provider": p, "model": m, "estimated_cost_usd": c}
                for p, m, c in self.alternatives
            ],
        }


def estimate_tokens(prompt: str, *, output_ratio: float = 0.35) -> tuple[int, int]:
    """Estimate input/output tokens from prompt text.

    Only used when the caller supplies no estimate of its own. A caller that knows the
    real shape of the work should pass explicit counts instead.
    """
    if output_ratio < 0:
        raise ValueError("output_ratio must be non-negative")
    input_tokens = max(1, len(prompt) // _CHARS_PER_TOKEN)
    return input_tokens, max(1, int(input_tokens * output_ratio))


def plan_delegation(
    task: Task,
    *,
    registry: ModelRegistry,
    adaptive_policy: AdaptivePolicy | None = None,
    policy: RoutingPolicy | None = None,
    threshold_tokens: int = DEFAULT_DELEGATION_THRESHOLD_TOKENS,
    max_cost_usd: float | None = None,
) -> DelegationDecision:
    """Decide whether to delegate ``task``, and to which model. Invokes nothing."""
    if threshold_tokens < 0:
        raise ValueError("threshold_tokens must be non-negative")

    routing = (policy or RoutingPolicy()).route(task)
    input_tokens = _int_metadata(task, "estimated_input_tokens")
    output_tokens = _int_metadata(task, "estimated_output_tokens")

    reliability_floor = 0.0
    if adaptive_policy is not None:
        reliability_floor = adaptive_policy.reliability_floor(task)

    base = DelegationDecision(
        delegate=False,
        reason="",
        execution_class=routing.execution_class,
        routing_reason=routing.reason,
        reliability_floor=reliability_floor,
        estimated_input_tokens=input_tokens,
        estimated_output_tokens=output_tokens,
    )

    # A task the hard policy already keeps away from a model is not a delegation
    # question at all -- report it rather than pricing models that will not run.
    if routing.execution_class in _NON_MODEL_CLASSES:
        return _replace(
            base,
            reason=(
                f"routed to {routing.execution_class.value}, which is not a model "
                "execution class"
            ),
        )

    total = input_tokens + output_tokens
    if total < threshold_tokens:
        return _replace(
            base,
            reason=(
                f"estimated {total} tokens is below the {threshold_tokens}-token "
                "delegation threshold; answering directly is cheaper than the round trip"
            ),
        )

    candidates = registry.ranked(
        task,
        routing.execution_class,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        min_reliability=reliability_floor,
        max_estimated_cost=max_cost_usd,
    )
    if not candidates:
        return _replace(
            base,
            reason=(
                f"no model satisfies execution class {routing.execution_class.value} at "
                f"reliability floor {reliability_floor:.2f}"
                + (f" within ${max_cost_usd:.6f}" if max_cost_usd is not None else "")
            ),
        )

    chosen = candidates[0]
    return _replace(
        base,
        delegate=True,
        reason=(
            f"{total} estimated tokens clears the {threshold_tokens}-token threshold; "
            f"{chosen.name} is the cheapest model meeting every constraint"
        ),
        provider=chosen.provider,
        model=chosen.name,
        estimated_cost_usd=_cost(chosen, input_tokens, output_tokens),
        alternatives=tuple(
            (p.provider, p.name, _cost(p, input_tokens, output_tokens)) for p in candidates[1:]
        ),
    )


_NON_MODEL_CLASSES = frozenset(
    {
        ExecutionClass.DETERMINISTIC,
        ExecutionClass.RETRIEVAL,
        ExecutionClass.HUMAN_REVIEW,
    }
)


def _cost(profile: ModelProfile, input_tokens: int, output_tokens: int) -> float:
    return profile.estimate_cost(input_tokens=input_tokens, output_tokens=output_tokens)


def _int_metadata(task: Task, key: str) -> int:
    value = task.metadata.get(key, 0)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _replace(decision: DelegationDecision, **changes: object) -> DelegationDecision:
    return replace(decision, **changes)  # type: ignore[arg-type]


def parse_requirements(raw: str | None) -> set[Requirement]:
    """Parse a comma-separated requirement list, naming the valid values on error."""
    if not raw:
        return set()
    parsed: set[Requirement] = set()
    for item in raw.split(","):
        name = item.strip()
        if not name:
            continue
        try:
            parsed.add(Requirement(name))
        except ValueError as exc:
            valid = ", ".join(sorted(r.value for r in Requirement))
            raise ValueError(f"unknown requirement {name!r}; valid values: {valid}") from exc
    return parsed


def parse_risk(raw: str | None) -> Risk:
    if not raw:
        return Risk.LOW
    try:
        return Risk(raw.strip())
    except ValueError as exc:
        valid = ", ".join(sorted(r.value for r in Risk))
        raise ValueError(f"unknown risk {raw!r}; valid values: {valid}") from exc
