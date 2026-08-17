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
Capability matching
  ↓
cheap plausible executor
  ↓
verification
  ├─ PASS      → result
  ├─ RETRY     → bounded retry
  └─ ESCALATE  → stronger executor / human review
```

The core principles are:

- **Route capabilities, not model names.** Policies select execution classes; adapters map those classes to models or tools.
- **Deterministic first.** Exact computation, validation, retrieval, and other toolable work should not consume LLM inference.
- **Verify before escalating.** Prefer the cheapest plausible executor, then escalate based on observed failure or uncertainty.
- **Route subtasks independently.** One agent run can mix tools, retrieval, small models, and frontier models.
- **Budgets are first-class.** Cost, latency, model-call, and tool-call ceilings travel with a run.
- **Observable by default.** Every route can emit a structured decision and execution event.

## Status

Initial architecture scaffold. The package currently provides:

- task and capability models
- execution classes and routing decisions
- hard-policy routing
- budget accounting
- pluggable executors and verifiers
- verification-driven escalation
- structured telemetry hooks

Provider adapters and learned routing are intentionally outside the core package.

## Quick start

```python
from agent_router import (
    ExecutionClass,
    ExecutionResult,
    Requirement,
    RouterRuntime,
    Task,
)

runtime = RouterRuntime()

runtime.register_executor(
    ExecutionClass.DETERMINISTIC,
    lambda task, ctx: ExecutionResult(output={"answer": 42}),
)

result = runtime.execute(
    Task(
        kind="arithmetic",
        payload={"expression": "6 * 7"},
        requirements={Requirement.EXACT_COMPUTATION},
    )
)

assert result.output == {"answer": 42}
```

## Development

Requires Python 3.11+.

```bash
python -m pip install -e '.[dev]'
pytest
ruff check .
```

## Non-goals

`agent-router` is not an agent framework, prompt library, model gateway, or provider SDK. It is the policy and execution-control layer that can sit underneath those systems.
