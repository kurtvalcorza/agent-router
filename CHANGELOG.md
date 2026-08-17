# Changelog

## 0.1.0 — 2026-08-18

Initial private release of `agent-router`.

### Added

- Policy-driven execution classes for deterministic, retrieval, light-reasoning, deep-reasoning, and human-review work.
- Verification-driven retry and escalation through `RouterRuntime`.
- First-class cost, latency, model-call, and tool-call budgets.
- Provider-neutral model profiles, capability matching, reliability floors, and cheapest-eligible-model selection.
- Adaptive `economy`, `balanced`, `quality`, and `critical` policy modes.
- OpenAI Responses and Anthropic Messages execution adapters.
- Workload-aware standard, cached-input, cache-write, batch, and long-context pricing.
- Declarative JSON/YAML model catalogs with guarded synchronization and promotion.
- OpenAI and Anthropic inventory ingestion plus authoritative pricing-source adapters with provenance.
- Availability reconciliation that requires repeated misses before confirming a model unavailable.
- Runnable benchmark corpus, deterministic graders, router/fixed-model baselines, evaluation reports, and CI-compatible acceptance gates.
- Empirical routing trained from historical benchmark runs using smoothed task/model success estimates and expected-total-cost selection.
- CLI entry points for catalog operations, provider/pricing refresh, evaluation, and live benchmark execution.

### Release hardening

- CI covers Python 3.11–3.13, Ruff, pytest, package build, wheel installation, and CLI smoke tests.
- OpenAI optional dependency requires the 2.x SDK generation used by the Responses API integration.
- Anthropic optional dependency requires the current 0.120 SDK generation used by the Models and Messages APIs.

### Known limitations

- Provider health/circuit-breaker state is not yet persisted across processes.
- Telemetry is callback-based; OpenTelemetry/JSONL exporters are future work.
- Built-in benchmark graders are intentionally simple; domain-specific or model-based graders should be supplied externally.
- Empirical routing quality depends on representative held-out benchmark data and must not be trained on the same cases used for evaluation.
