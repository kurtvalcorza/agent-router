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

## Status

The package currently provides:

- task and capability models
- execution classes and routing decisions
- hard-policy routing
- budget accounting
- pluggable executors and verifiers
- verification-driven escalation
- structured telemetry hooks
- provider-neutral model profiles and registry
- capability/context/reliability eligibility filtering
- cheapest-eligible-model selection using estimated token cost
- same-class provider/model fallback when an invocation fails

Concrete provider SDK adapters and learned routing remain outside the core package.

## Model routing

```python
from agent_router import (
    ExecutionClass,
    ModelProfile,
    ModelRegistry,
    ModelResponse,
    Requirement,
    RoutedModelExecutor,
    RouterRuntime,
    Task,
)

registry = ModelRegistry(
    [
        ModelProfile(
            name="small-model",
            provider="provider-a",
            execution_classes={ExecutionClass.LIGHT_REASONING},
            capabilities={Requirement.SEMANTIC_REASONING},
            context_window=128_000,
            input_cost_per_million=0.25,
            output_cost_per_million=1.00,
            reliability=0.90,
        ),
        ModelProfile(
            name="strong-model",
            provider="provider-b",
            execution_classes={
                ExecutionClass.LIGHT_REASONING,
                ExecutionClass.DEEP_REASONING,
            },
            capabilities={
                Requirement.SEMANTIC_REASONING,
                Requirement.DEEP_PLANNING,
            },
            context_window=200_000,
            input_cost_per_million=3.00,
            output_cost_per_million=15.00,
            reliability=0.98,
        ),
    ]
)


def invoke(provider: str, model: str, task: Task) -> ModelResponse:
    # Dispatch to the relevant provider SDK here.
    return ModelResponse(output={"provider": provider, "model": model})


runtime = RouterRuntime()
model_executor = RoutedModelExecutor(registry=registry, invoke=invoke)
runtime.register_executor(ExecutionClass.LIGHT_REASONING, model_executor)
runtime.register_executor(ExecutionClass.DEEP_REASONING, model_executor)

result = runtime.execute(
    Task(
        kind="semantic-analysis",
        payload={"question": "What is driving this change?"},
        requirements={Requirement.SEMANTIC_REASONING},
        metadata={
            "estimated_input_tokens": 4_000,
            "estimated_output_tokens": 800,
        },
    )
)
```

## Deterministic quick start

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
