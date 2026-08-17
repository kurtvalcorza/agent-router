# agent-router

A policy-driven execution router for cost-efficient AI agents.

`agent-router` routes work to the cheapest execution path that can satisfy a task's requirements, verifies the result, and escalates when necessary. It is intentionally provider- and framework-agnostic.

## Design

```text
Request
  ↓
Task normalization
  ↓
Hard routing policy ──→ deterministic/tool execution
  ↓
Adaptive policy mode
  ↓
Capability + reliability + budget filtering
  ↓
cheapest eligible executor/model
  ↓
verification
  ├─ PASS      → result
  ├─ RETRY     → bounded retry
  └─ ESCALATE  → stronger executor / human review
```

The core principles are:

- **Route capabilities, not model names.** Policies select execution classes; model profiles declare capabilities, context limits, reliability, and cost.
- **Deterministic first.** Exact computation, validation, retrieval, and other toolable work should not consume LLM inference.
- **Verify before escalating.** Prefer the cheapest plausible executor, then escalate based on observed failure or uncertainty.
- **Route subtasks independently.** One agent run can mix tools, retrieval, small models, and frontier models.
- **Budgets are first-class.** Cost, latency, model-call, and tool-call ceilings travel with a run.
- **Observable by default.** Every route can emit a structured decision and execution event.

## Policy modes

`AdaptivePolicy` provides four reusable operating modes:

- `economy` — lowest reliability floor; aggressively prefers low-cost eligible models.
- `balanced` — moderate reliability floor for general-purpose workloads.
- `quality` — excludes lower-reliability models before cost ranking.
- `critical` — highest reliability floor and intended for high-consequence workloads.

Risk and explicit `HIGH_RELIABILITY` requirements can raise the floor above the mode default. Remaining run budget is also applied before invocation, so models whose estimated call cost exceeds the available budget are excluded.

```python
from agent_router import AdaptivePolicy, PolicyMode, RoutedModelExecutor

executor = RoutedModelExecutor(
    registry=registry,
    invoke=invoke,
    adaptive_policy=AdaptivePolicy(PolicyMode.QUALITY),
)
```

## Current capabilities

- task and capability models
- execution classes and hard-policy routing
- budget accounting
- pluggable executors and verifiers
- verification-driven escalation
- structured telemetry hooks
- provider-neutral model profiles and registry
- capability/context/reliability eligibility filtering
- adaptive `economy`, `balanced`, `quality`, and `critical` policies
- remaining-budget filtering before model invocation
- cheapest-eligible-model selection using estimated token cost
- same-class provider/model fallback when an invocation fails

Concrete provider SDK adapters and learned routing remain outside the core package.

## Development

Requires Python 3.11+.

```bash
python -m pip install -e '.[dev]'
pytest
ruff check .
```

## Non-goals

`agent-router` is not an agent framework, prompt library, model gateway, or provider SDK. It is the policy and execution-control layer that can sit underneath those systems.
