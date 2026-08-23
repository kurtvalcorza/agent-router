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

## Quickstart

The router calls providers through a plain callable, so the whole decision path runs offline with a
stub in place of the provider adapters. `examples/quickstart.py` does exactly that -- no API keys,
no spend:

```bash
python -m pip install -e '.[catalog]'
python examples/quickstart.py
```

```text
1. cheap task, cheap model
   routed to light_reasoning: task requires bounded semantic reasoning
      provider call -> openai/small-model
   answered by small-model (floor 0.82) for $0.000010 in 1 model call(s)

2. high-reliability task skips the cheap model entirely
   routed to deep_reasoning: task requires high-capability reasoning
      provider call -> anthropic/strong-model
   answered by strong-model (floor 0.95) for $0.001920 in 1 model call(s)

3. cheap model answers wrong, verification escalates
   routed to light_reasoning: task requires bounded semantic reasoning
      provider call -> openai/small-model
      provider call -> anthropic/strong-model
   answered by strong-model (floor 0.82) for $0.001475 in 2 model call(s)
```

Scenario 2 shows the adaptive reliability floor rising from 0.82 to 0.95 and disqualifying the
0.90-reliability cheap model before cost ranking. Scenario 3 shows verification-driven escalation
across execution classes. Swap `stub_invoke` for a real `ProviderInvoker` to run the same paths
against live providers.

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

catalog = load_catalog("config/models.example.yaml")
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
agent-router catalog check config/models.example.yaml
agent-router catalog diff config/models.example.yaml config/models.candidate.yaml
agent-router catalog sync config/models.example.yaml snapshots.json \
  --output config/models.candidate.yaml \
  --pricing-as-of 2026-08-17 \
  --pricing-source provider-pricing-page
```

`catalog sync` never overwrites the pinned source catalog. It writes a candidate file, prints warnings for unmanaged or missing provider models, and prints the structured diff for review.

The commands above use the shipped `config/models.example.yaml` so they run against a clean checkout. In production, point them at your own reviewed catalog instead.

### Provider inventory and pricing

```bash
agent-router provider fetch openai --output .agent-router/openai-inventory.json
agent-router provider fetch anthropic --output .agent-router/anthropic-inventory.json

agent-router pricing fetch anthropic \
  --model-map config/anthropic-model-map.example.json \
  --output .agent-router/anthropic-pricing.json

agent-router pricing fetch openai \
  --model-map config/openai-model-map.example.json \
  --output .agent-router/openai-pricing.json
```

Pricing adapters preserve source provenance and fail closed when expected authoritative source structures change. Model identity mappings and special long-context thresholds remain explicit reviewed configuration.

## Benchmark execution and evaluation

`benchmarks/cases.example.json` shows the benchmark case convention. Each case declares its task payload, routing requirements, risk, expected output, token estimates, and grader inside `metadata`. The harness compiles that deterministically into a real `Task`.

`agent-router-benchmark` issues **real, billed provider calls**. It builds adapters with
`OpenAIResponsesAdapter.from_env()` / `AnthropicMessagesAdapter.from_env()` for every provider in
the catalog, so it needs `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` and spends money on every run.

It also needs a catalog naming **real** models. The `small-model` and `strong-model` entries in
`config/models.example.yaml` are placeholders that exist at neither provider, so the benchmark
cannot run against the shipped example even with valid keys. Create your own catalog first:

```bash
cp config/models.example.yaml config/models.yaml
# then replace the model names, pricing, reliability, and pricing source with reviewed values
```

`config/models.yaml` is git-ignored: it is your reviewed catalog, not a shipped artifact.

Verify model names with a real call rather than a listing. A provider's model list can
advertise models your key cannot actually invoke -- Gemini's `models.list()` reports
`generateContent` support for models that return `404 "no longer available to new users"`, and
free-tier keys get `429` with `limit: 0` for pro-tier models. `reliability` is a local policy
judgement and should come from your own evaluation evidence, never from provider marketing
metadata.

The commands in this section and in `docs/catalog-refresh.md` assume that `config/models.yaml`.
Install provider extras and run the static router plus fixed baselines against the same corpus:

```bash
agent-router-benchmark \
  --cases benchmarks/cases.example.json \
  --catalog config/models.yaml \
  --cheap fast \
  --strong strong \
  --mode balanced \
  --output benchmarks/runs.json
```

The runner supports:

- `router` — the policy + capability + static reliability + cost router
- `empirical-router` — the benchmark-trained success-probability + expected-cost router
- `always-cheap` — a fixed low-cost catalog model or alias
- `always-strong` — a fixed high-capability catalog model or alias

Built-in deterministic graders currently include `exact_match`, `text_exact`, and `contains_all`. Grading is provider-neutral and can also be replaced programmatically.

Then gate a strategy against the strong baseline:

```bash
agent-router evaluation report \
  --cases benchmarks/cases.example.json \
  --runs benchmarks/runs.json \
  --strategy router \
  --baseline always-strong \
  --minimum-cost-savings 0.30 \
  --maximum-quality-loss 0.05 \
  --maximum-success-rate-loss 0.00
```

This reports success rate, mean quality, total cost, latency, escalation rate, and deltas against the selected baseline. A failed acceptance gate returns a non-zero exit status so the benchmark can be used in CI.

## Empirical routing

Fit the empirical model from a **historical/training corpus**, not from the evaluation corpus you intend to use for comparison:

```bash
agent-router evaluation train-empirical \
  --cases benchmarks/train-cases.json \
  --runs benchmarks/train-runs.json \
  --output .agent-router/empirical-router.json
```

The model estimates `P(success | task features, model)` using hierarchical Beta smoothing. Task features currently include task kind, risk, and declared requirements. Sparse task/model combinations shrink toward the model's global observed success rate instead of producing brittle 0% or 100% estimates.

Run the held-out corpus with the trained empirical router:

```bash
agent-router-benchmark \
  --cases benchmarks/eval-cases.json \
  --catalog config/models.yaml \
  --cheap fast \
  --strong strong \
  --empirical-model .agent-router/empirical-router.json \
  --strategies empirical-router always-cheap always-strong \
  --output benchmarks/empirical-runs.json
```

The empirical selector first enforces capability, execution-class, reliability-floor, and budget constraints. It then ranks eligible models by expected total cost:

```text
expected_total_cost = call_cost + P(failure) × recovery_cost
```

`--recovery-cost-multiplier` can tune how aggressively the router penalizes likely recovery/escalation work. The selected model, empirical success probability, feature key, and expected total cost are retained in result metadata for auditability.

## Provider adapters

The core package has no provider SDK dependency. Install only the adapters you need:

```bash
python -m pip install 'agent-router[openai]'
python -m pip install 'agent-router[anthropic]'
python -m pip install 'agent-router[google]'
# or all three
python -m pip install 'agent-router[providers]'
```

Each adapter's `from_env()` constructs the provider SDK client, and the SDK reads its own
credential from the process environment: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and — for Gemini —
`GOOGLE_API_KEY` (or `GEMINI_API_KEY`; if both are set the Google SDK uses `GOOGLE_API_KEY` and
warns). `agent-router` itself never reads a `.env` file, so export the variable in your shell or
set it through your process manager.

Adapters expose one common invocation contract through `ProviderInvoker`:

```python
from agent_router import (
    AnthropicMessagesAdapter,
    GeminiAdapter,
    OpenAIResponsesAdapter,
    ProviderInvoker,
    RoutedModelExecutor,
)

providers = ProviderInvoker(
    {
        "openai": OpenAIResponsesAdapter.from_env(),
        "anthropic": AnthropicMessagesAdapter.from_env(max_tokens=2048),
        "google": GeminiAdapter.from_env(),
    }
)

executor = RoutedModelExecutor(
    registry=registry,
    invoke=providers,
    adaptive_policy=adaptive_policy,
)
```

`OpenAIResponsesAdapter` uses the Responses API and defaults to `store=False`. `AnthropicMessagesAdapter` uses the Messages API. `GeminiAdapter` uses `google-genai`'s `generate_content` API and maps `usage_metadata.prompt_token_count` / `candidates_token_count` onto the shared token fields. All three normalize output text and input/output token usage into `ModelResponse`; provider-specific response IDs and stop/finish metadata remain available in result metadata.

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
- optional OpenAI Responses, Anthropic Messages, and Google Gemini adapters
- JSON/YAML declarative model catalogs
- model aliases and catalog validation
- dated pricing metadata and source provenance fields
- candidate catalog synchronization and structured diffs
- guarded catalog promotion that preserves locally reviewed policy
- OpenAI and Anthropic model inventory fetchers
- availability reconciliation with repeated-miss confirmation
- OpenAI and Anthropic authoritative pricing ingestion
- runnable benchmark corpus and task compilation
- live router / always-cheap / always-strong benchmark execution
- persisted evaluation runs and CI-compatible acceptance gates
- persisted empirical success model trained from benchmark history
- hierarchical task/model success estimation
- expected-total-cost empirical model selection
- live empirical-router benchmark strategy
- Python 3.11–3.13 CI plus wheel-build/install smoke validation

## v0.1 status

The v0.1 architecture is feature-complete. Remaining items are intentionally deferred to later releases: telemetry exporters, persistent provider health/circuit breakers, shared caching, concurrency/rate-limit orchestration, richer benchmark graders, and service/dashboard integrations.

See `CHANGELOG.md` for the v0.1.0 release scope.

## Development

Requires Python 3.11+.

```bash
python -m pip install -e '.[dev]'
pytest
ruff check .
python -m build
```

## Non-goals

`agent-router` is not an agent framework, prompt library, model gateway, or provider SDK. It is the policy and execution-control layer that can sit underneath those systems.
