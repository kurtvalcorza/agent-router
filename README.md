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

### Delegating a subtask (`route`)

`agent-router route` is a host-agnostic delegation entry point: any agent that can run a
subprocess can ask it whether a subtask is worth handing to a cheaper model, and which one.
No server, no per-host plugin.

`--plan` is the default and **never calls a provider**:

```bash
agent-router route "Classify each ticket by urgency and product area."   --catalog config/models.yaml   --input-tokens 12000 --output-tokens 3000
```

```text
DELEGATE: 15000 estimated tokens clears the 400-token threshold; gemini-3.5-flash-lite is the cheapest model meeting every constraint
  execution class  : light_reasoning (task requires bounded semantic reasoning)
  reliability floor: 0.82
  estimated tokens : 12000 in / 3000 out
  selected         : google/gemini-3.5-flash-lite (est. $0.000570)
  alternative      : google/gemini-3.5-flash (est. $0.002430)
```

The command enforces a delegation threshold itself, because a caller pays its own tokens to
ask. Below `--threshold-tokens` (default 400) it reports `DO NOT DELEGATE` rather than routing
work that costs more to hand off than to do. `--json` emits the same decision machine-readably.

`--execute` runs the task through the selected model and issues **real, billed provider calls**.
It stays inert when the plan already said no, so a below-threshold task cannot spend by accident:

```bash
agent-router route - --catalog config/models.yaml --execute --max-cost-usd 0.02 < prompt.txt
```

`--requirements` defaults to `semantic_reasoning` and **replaces** rather than extends that
default; add `high_reliability` to raise the reliability floor before cost ranking.

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

## Reliability calibration

`reliability` is not configuration. It decides whether a model may enter a policy tier at
all, so a guessed value silently encodes routing behaviour -- and a zero-cost local model is
gated by nothing else. `evaluation calibrate` turns benchmark evidence into a **proposal** for
that number.

Measurement and policy approval stay separate:

```text
benchmark runs -> empirical fit -> calibration proposal -> review -> catalog update
```

Neither `evaluation calibrate` nor `evaluation train-empirical` may modify a catalog.

```bash
agent-router evaluation calibrate \
  --cases benchmarks/cases.json --runs benchmarks/runs.json \
  --catalog config/models.yaml \
  --evidence-ref benchmark-run-2026-08-23 \
  --output .agent-router/proposals.json
```

```text
qwen3:8b: 0.800 -> 0.741  [REVIEW_REQUIRED]
  evidence   : 47/56, posterior mean 0.828, credible 0.741-0.902
  aggregation: min(lower bound 0.741, class-balanced 0.774)
  coverage   : kind=classify|risk=low|requirements=semantic_reasoning -> 36/40 (0.881)
  coverage   : kind=summarize|risk=low|requirements=semantic_reasoning -> 11/16 (0.667)
  warning    : 71% of trials come from one task class; the corpus is skewed
```

The proposal is deliberately **below** the posterior mean. A small corpus skewed toward easy
cases would otherwise promote a model across a policy floor on thin evidence, so the proposed
value is the lower of the pooled posterior's credible lower bound and the mean of the
per-task-class posteriors, weighting each observed class equally. Both inputs are printed, so
the conservatism is inspectable rather than implicit. Coverage is printed for the same reason:
a number derived from one task class says nothing about task shapes that were never
benchmarked.

When a proposal moves a model across a policy floor, that is called out explicitly -- it is the
change that alters which tiers may route to the model at all:

```text
  THRESHOLD  : LOSES eligibility in quality (floor 0.90)
```

Applying is a separate act, and never touches the catalog you point it at:

```bash
agent-router catalog apply-calibration config/models.yaml .agent-router/proposals.json \
  --output config/models.candidate.yaml \
  --accept qwen3:8b
```

Nothing is applied without `--accept` naming models (or `--accept all`), and proposals marked
`INSUFFICIENT_EVIDENCE` are skipped unless `--allow-insufficient-evidence` is passed.

**Applying is guarded against stale proposals.** A proposal records the reliability it was
reviewed against, and application refuses if the catalog has moved since -- otherwise a decision
made about `0.80` could silently overwrite a newer `0.92`. The check covers the whole accepted
set: if any proposal is stale, nothing is applied, because the set was reviewed together against
one catalog state. This is why `--catalog` is required at calibration time; a proposal with no
recorded baseline cannot be checked and is refused.

```text
STALE: the catalog has moved since these proposals were calibrated.
  qwen3:8b: calibrated against 0.8, catalog now holds 0.92
Nothing applied. Re-run 'evaluation calibrate' against the current catalog.
```

Each applied value carries its provenance into the candidate catalog:

```yaml
metadata:
  reliability_evidence:
    evidence_ref: benchmark-run-2026-08-23
    method: beta-posterior-conservative
    method_version: "1"
    successes: 47
    trials: 56
    credible_interval: [0.74064, 0.90161]
    previous_reliability: 0.8
    review_state: applied-by-explicit-action
```

Calibrating a **local** model costs nothing, which is convenient, because it is the model whose
reliability is most load-bearing: it is the one a zero price cannot gate.

### Two different numbers

| | catalog `reliability` | `EmpiricalSuccessModel` |
| :--- | :--- | :--- |
| answers | may this model enter this tier at all? | given this task shape, how likely is success? |
| scope | model-level eligibility prior | task-conditional ranking signal |
| changes | rarely, under review | refit freely from run history |

The empirical executor applies the catalog floor **before** empirical ranking, so adopting
`empirical-router` does not remove the need for a calibrated `reliability`.

## Provider adapters

The core package has no provider SDK dependency. Install only the adapters you need:

```bash
python -m pip install 'agent-router[openai]'
python -m pip install 'agent-router[anthropic]'
python -m pip install 'agent-router[google]'
# or all three
python -m pip install 'agent-router[providers]'
```

The `openai` extra also covers self-hosted runtimes: `OpenAIChatCompletionsAdapter` speaks the
OpenAI chat-completions API, so it drives LiteRT-LM, Ollama, llama.cpp, vLLM, or LM Studio
through the same contract as a hosted provider.

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

`OpenAIResponsesAdapter` uses the Responses API and defaults to `store=False`. `OpenAIChatCompletionsAdapter` uses the older chat-completions API, which is what self-hosted servers implement, and takes a `base_url`; it maps `prompt_tokens`/`completion_tokens` onto the shared token fields and tolerates a missing `usage` block, which local runtimes routinely omit. `AnthropicMessagesAdapter` uses the Messages API. `GeminiAdapter` uses `google-genai`'s `generate_content` API and maps `usage_metadata.prompt_token_count` / `candidates_token_count` onto the shared token fields. All three normalize output text and input/output token usage into `ModelResponse`; provider-specific response IDs and stop/finish metadata remain available in result metadata.

## Local and self-hosted models

A local runtime is just another provider. `config/models.local.example.yaml` shows a tiered
catalog -- free local, then cheap cloud, then strong cloud -- and documents the one property
that makes such a catalog behave:

> A zero-priced model is **always** the cheapest eligible candidate, so cost ranking can never
> gate it. Only `context_window`, `reliability`, and `capabilities` can.

Set those three honestly and routing becomes a ladder:

```bash
# small, low-stakes -> the free local model
agent-router route "Summarize this paragraph." \
  --catalog config/models.local.example.yaml --mode economy \
  --input-tokens 800 --output-tokens 200
```

```text
DELEGATE: 1000 estimated tokens clears the 400-token threshold; qwen3-4b-instruct is the cheapest model meeting every constraint
  selected         : local/qwen3-4b-instruct (est. $0.000000)
  alternative      : google/cloud-flash-lite (est. $0.000740)
```

Raise the prompt past the local model's `context_window` and it drops out of the ranking
entirely; require `high_reliability` and the floor excludes it before cost is considered.

`context_window` deserves particular care for a local runtime: it is the **serving** limit, not
the model's native context. LiteRT-LM 0.14.0 runs at a fixed `max_num_tokens=4096` whatever the
model card says, and a prompt past its practical ceiling breaks the HTTP response rather than
returning an error status. Declaring the real limit turns that failure mode into an ordinary
routing rule.

The `local` provider defaults to `http://127.0.0.1:9379/v1` (LiteRT-LM's port). Override with
`AGENT_ROUTER_LOCAL_BASE_URL`. **Setting `base_url` deliberately severs the credential path to `OPENAI_API_KEY`.** That
variable is never read for a custom endpoint, because doing so would put your real OpenAI
credential in an `Authorization` header addressed to whatever host `base_url` names. Supply a
credential for a non-OpenAI endpoint explicitly, or through `AGENT_ROUTER_LOCAL_API_KEY`;
otherwise a harmless placeholder is sent, which local servers ignore. Calls with no `base_url`
still resolve `OPENAI_API_KEY` through the SDK as usual.

Verified against two runtimes, which disagree on optional fields in ways worth knowing:

| | LiteRT-LM 0.14.0 (`:9379`) | Ollama (`:11434`) |
| :--- | :--- | :--- |
| `usage` block | absent -- tokens degrade to `0` | present |
| reasoning tokens | n/a | counted in `completion_tokens`, absent from the text |

Both are handled, but they have consequences. Missing `usage` means telemetry carries no token
counts for that runtime -- harmless at zero price, misleading if you ever price it. And a
reasoning model bills for tokens you never see: a one-word answer from `qwen3:8b` reported 151
completion tokens. Cost estimates for such a model must be based on its *total* output, not the
visible reply.

**One `local` provider means one base URL.** `provider_invoker_from_catalog` builds a single
`local` adapter, so a catalog cannot currently point two entries at two different servers. Running
LiteRT-LM and Ollama side by side means choosing one per run via `AGENT_ROUTER_LOCAL_BASE_URL`, or
constructing the `ProviderInvoker` yourself with one adapter per provider name.

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
- OpenAI-compatible chat-completions adapter for self-hosted runtimes (LiteRT-LM, Ollama, llama.cpp, vLLM, LM Studio)
- host-agnostic `route` delegation command with a plan/execute split and a delegation threshold
- evidence-backed reliability calibration proposals, gated behind explicit review
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
