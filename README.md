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
- **Provider facts do not define policy.** Upstream metadata may refresh prices and context limits, but capability, reliability, aliases, and execution-class assignments require local review.

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

Catalog entries define provider, execution classes, capabilities, context window, locally governed reliability, and workload-aware token pricing. Aliases such as `fast` or `strong` can point to concrete model names without changing application code.

`config/models.example.yaml` intentionally contains illustrative values only. Production catalogs should record an authoritative pricing source and date. Capability and reliability judgments remain local policy inputs rather than being inferred from provider marketing metadata.

## Catalog synchronization

Provider metadata can be staged into a candidate catalog without automatically changing routing policy:

```python
from agent_router import ProviderModelSnapshot, promote_candidate, synchronize_catalog

sync = synchronize_catalog(
    catalog,
    [ProviderModelSnapshot(provider="openai", name="small-model", context_window=200_000)],
    pricing_as_of="2026-08-17",
    pricing_source="provider-pricing-page",
)

for change in sync.diff.changed:
    print(change.model, change.field, change.before, "→", change.after)

catalog = promote_candidate(catalog, sync.candidate)
```

Synchronization intentionally updates only operational provider facts: context windows, pricing, and provider snapshot metadata. Newly discovered models are reported but not auto-added because they have no reviewed capability or reliability policy. Promotion rejects changes to model membership, aliases, provider ownership, execution classes, capabilities, or reliability.

## CLI

Installing the package exposes an `agent-router` command:

```bash
agent-router catalog check config/models.yaml
agent-router catalog diff config/models.yaml config/models.candidate.yaml
agent-router catalog sync config/models.yaml snapshots.json \
  --output config/models.candidate.yaml \
  --pricing-as-of 2026-08-17 \
  --pricing-source provider-pricing-page
```

`catalog sync` never overwrites the pinned source catalog. It writes a candidate file, prints warnings for unmanaged or missing provider models, and prints the structured diff for review.

### Anthropic pricing fetch

Anthropic pricing is parsed from the official pricing page into normalized `PricingRecord` JSON. Display-name-to-model-ID mapping is explicit and reviewable; the parser does not infer API model IDs from marketing names.

```bash
agent-router pricing fetch anthropic \
  --model-map config/anthropic-model-map.example.json \
  --output .agent-router/anthropic-pricing.json
```

The model map may also declare reviewed long-context thresholds:

```json
{
  "models": {
    "Claude Sonnet 4": "claude-sonnet-4-20250514"
  },
  "long_context_thresholds": {
    "Claude Sonnet 4": 200000
  }
}
```

When a reviewed threshold is declared, the pricing parser requires the official long-context pricing table and records its premium input/output rates in the `PricingProfile`. If the expected source structure is missing or changes, the fetch fails closed rather than guessing.

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
- workload-aware standard/cache/batch/long-context pricing
- remaining-budget filtering before model invocation
- cheapest-eligible-model selection using estimated token cost
- same-class provider/model fallback when an invocation fails
- optional OpenAI Responses and Anthropic Messages adapters
- JSON/YAML declarative model catalogs
- model aliases and catalog validation
- dated pricing metadata and source provenance fields
- candidate catalog synchronization and structured diffs
- guarded catalog promotion that preserves locally reviewed policy
- OpenAI model inventory fetcher with provenance
- availability reconciliation with repeated-miss confirmation
- Anthropic authoritative pricing parser with long-context rules
- operational catalog and Anthropic pricing CLI commands

Learned routing and additional live provider pricing/inventory adapters remain outside the core package.

## Development

Requires Python 3.11+.

```bash
python -m pip install -e '.[dev]'
pytest
ruff check .
```

## Non-goals

`agent-router` is not an agent framework, prompt library, model gateway, or provider SDK. It is the policy and execution-control layer that can sit underneath those systems.
