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
Declarative model catalog
  ↓
Capability + reliability + budget filtering
  ↓
cheapest eligible executor/model
  ↓
provider adapter
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
- **Pricing is data, not code.** Catalogs carry dated pricing metadata so cost updates do not require runtime changes.

## Policy modes

`AdaptivePolicy` provides four reusable operating modes:

- `economy` — lowest reliability floor; aggressively prefers low-cost eligible models.
- `balanced` — moderate reliability floor for general-purpose workloads.
- `quality` — excludes lower-reliability models before cost ranking.
- `critical` — highest reliability floor and intended for high-consequence workloads.

Risk and explicit `HIGH_RELIABILITY` requirements can raise the floor above the mode default. Remaining run budget is also applied before invocation, so models whose estimated call cost exceeds the available budget are excluded.

## Model catalogs

Model profiles can be loaded from JSON with no extra dependency, or YAML with the optional catalog extra:

```bash
python -m pip install 'agent-router[catalog]'
```

```python
from agent_router import load_catalog

catalog = load_catalog("config/models.yaml")
registry = catalog.registry()

print(catalog.metadata.pricing_as_of)
print(registry.get("fast").name)
```

Catalog entries define provider, execution classes, capabilities, context window, locally governed reliability, and token pricing. Aliases such as `fast` or `strong` can point to concrete model names without changing application code.

```yaml
version: "2026-08-17"
pricing_as_of: "2026-08-17"
pricing_source: "authoritative-provider-pricing-page"

aliases:
  fast: small-model

models:
  - name: small-model
    provider: openai
    execution_classes: [light_reasoning]
    capabilities: [semantic_reasoning]
    context_window: 128000
    reliability: 0.90
    pricing:
      input_per_million: 0.25
      output_per_million: 1.00
```

`config/models.example.yaml` intentionally contains illustrative values only. Production catalogs should record an authoritative pricing source and date. Capability and reliability judgments remain local policy inputs rather than being inferred from provider marketing metadata.

## Provider adapters

The core package has no provider SDK dependency. Install only the adapters you need:

```bash
python -m pip install 'agent-router[openai]'
python -m pip install 'agent-router[anthropic]'
# or both
python -m pip install 'agent-router[providers]'
```

Adapters expose one common invocation contract through `ProviderInvoker`:

```python
from agent_router import (
    AnthropicMessagesAdapter,
    OpenAIResponsesAdapter,
    ProviderInvoker,
    RoutedModelExecutor,
)

providers = ProviderInvoker(
    {
        "openai": OpenAIResponsesAdapter.from_env(),
        "anthropic": AnthropicMessagesAdapter.from_env(max_tokens=2048),
    }
)

executor = RoutedModelExecutor(
    registry=registry,
    invoke=providers,
    adaptive_policy=adaptive_policy,
)
```

`OpenAIResponsesAdapter` uses the Responses API and defaults to `store=False`. `AnthropicMessagesAdapter` uses the Messages API. Both normalize output text and input/output token usage into `ModelResponse`; provider-specific response IDs and stop metadata remain available in result metadata.

For custom task shapes, inject a `prompt_builder: Callable[[Task], str]` rather than teaching the routing core about provider prompt formats.

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
- optional OpenAI Responses and Anthropic Messages adapters
- provider dispatch through a common invocation contract
- JSON/YAML declarative model catalogs
- model aliases and catalog validation
- dated pricing metadata and source provenance fields

Learned routing and automated catalog refresh remain outside the core package.

## Development

Requires Python 3.11+.

```bash
python -m pip install -e '.[dev]'
pytest
ruff check .
```

## Non-goals

`agent-router` is not an agent framework, prompt library, model gateway, or provider SDK. It is the policy and execution-control layer that can sit underneath those systems.
