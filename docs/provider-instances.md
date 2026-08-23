# Provider instances — design spec (not implemented)

Status: **proposed**. Nothing in this document is built. It exists to be argued with before
code, because it changes what `provider` means across the repository.

## Problem

`provider` currently means *adapter type*: `openai`, `anthropic`, `google`, `local`.
`provider_invoker_from_catalog` maps that string to one adapter, and the `local` adapter takes
its base URL from a single environment variable. Consequences:

- One `local` provider means **one base URL**. A catalog cannot address LiteRT-LM on `:9379`
  and Ollama on `:11434` at once, which is a configuration that already exists in practice.
- Adding a second endpoint of the same kind means another global environment variable, so
  topology accumulates in the environment rather than in the reviewed artifact.
- `google/qwen3:8b` served from two different hosts is indistinguishable in telemetry, because
  provenance carries `provider` and `model` and nothing else.

## Proposal

Split the two concepts that `provider` currently conflates:

- **`kind`** selects the adapter implementation.
- **`provider`** becomes a **provider instance id** — a name the catalog author chooses.

```yaml
providers:
  ollama-local:
    kind: openai_chat
    base_url: http://127.0.0.1:11434/v1

  litert-gpu:
    kind: openai_chat
    base_url: http://127.0.0.1:9379/v1

  remote-vllm:
    kind: openai_chat
    base_url: https://inference.example.org/v1
    api_key_env: VLLM_API_KEY

  gemini:
    kind: google

models:
  - name: qwen3:8b
    provider: ollama-local

  - name: qwen3-4b-instruct
    provider: litert-gpu
```

`kind` values map to the existing adapters: `openai_responses`, `openai_chat`, `anthropic`,
`google`.

## Non-negotiable constraints

1. **Credential fields name environment variables; they never carry values.** The schema
   permits `api_key_env` and rejects any `api_key` key outright, with an error naming the
   field. This is the same boundary established when `base_url` was severed from
   `OPENAI_API_KEY`: a catalog is a reviewed, shared, committed artifact, and a secret must
   not be able to enter it even by accident.
2. **Serialization refuses to emit a suspected secret.** `catalog_to_dict` gains a guard that
   raises rather than writing a provider block containing a value that looks like a
   credential. Failing the write is correct: silently redacting produces a catalog that no
   longer round-trips, and a reviewer would not see what was dropped.
3. **Instance identity flows into provenance and telemetry.** `ExecutionResult.metadata`
   already carries `provider` and `model`; with instances those become the instance id and the
   model name, so `ollama-local/qwen3:8b` and `remote-vllm/qwen3:8b` stay distinguishable even
   when the model name is identical. Benchmark runs, empirical feature keys, and calibration
   evidence all inherit that distinction — which matters, because the same weights behind two
   endpoints can have genuinely different reliability.

## Migration

The change is backward compatible if bare adapter-type names keep working:

- A `provider` naming no declared instance, but matching a known adapter type, resolves to an
  implicit instance of that kind. Every catalog that exists today keeps parsing.
- `providers:` is optional. Declaring an instance whose id collides with an adapter type name
  is an error, so `provider: google` can never be ambiguous.
- `AGENT_ROUTER_LOCAL_BASE_URL` continues to configure the implicit `local` instance, and is
  documented as superseded by a declared instance.

Deprecation is a later decision. Nothing forces existing catalogs to migrate.

## What this touches

Enumerated because the breadth is the argument for speccing first:

| Area | Change |
| :--- | :--- |
| `catalog.py` | parse and validate a `providers:` block; instance/adapter-type collision rules |
| `models.py` | `ModelProfile.provider` becomes an instance id |
| `serialize.py` | round-trip the block; secret guard |
| `live_evaluation.py` | build one adapter per declared instance rather than per adapter type |
| `catalog_sync.py` | instance identity is reviewed policy; promotion must reject silent changes to it |
| telemetry / `records_io` | provenance carries instance id |
| `empirical.py` | decide whether the feature key includes instance id |
| `calibration.py` | proposals become per instance+model, not per model name |
| examples, README, `docs/catalog-refresh.md` | new shape |

## Settled: identity levels

Decided 2026-08-23, before implementation. This was previously an open question; it is not.

| level | key | role |
| :--- | :--- | :--- |
| model / weights | `name` | shrinkage identity — borrows strength across instances |
| **execution identity** | `(provider_instance, name)` | **where evidence is collected** |
| task-conditional | `(provider_instance, name, task_class)` | ranking signal |

**Evidence is collected at the execution-identity level.** Two instances serving the same
weights can differ in quantization, context limit, sampling defaults, hardware failure modes,
and load. Pooling them into one empirical model is the same hidden boundary crossing that
produced the credential leak and the lost update: a value silently crossing a line nobody
checked. Sparse evidence is a reason for hierarchical shrinkage, not for collapsing two
execution environments into one identity.

**Shrinkage borrows strength from the model-name level.** A new instance of a known model
starts from what that model generally does and moves as its own evidence accumulates. That
preserves the statistical sharing without asserting the environments are interchangeable.

Note this needs **no new catalog field**: the shrinkage group is the existing `name`, and the
leaf is `(provider_instance, name)`. It does, however, make the empirical estimator three-level
where it is two-level today, which is a change to `EmpiricalSuccessModel` and not merely a
schema change. `feature_weight` becomes two weights needing separate justification.

**Compound identity must be a first-class value, not a string.** No `"instance/model"` formatting
at call sites. A dedicated frozen type with the two fields, used as the dict key and the
serialized shape, so an aliasing bug is a type error rather than a silent match. String
concatenation is exactly how `ollama-local/qwen3:8b` ends up accepting evidence gathered from
`remote-vllm/qwen3:8b`.

## Migration warning: every structure keyed by bare model name

Each of these keys on `name` alone today. Each is a place where two instances of the same model
silently alias, and each must be audited during implementation:

| structure | where | failure if missed |
| :--- | :--- | :--- |
| calibration proposal `model` field | `calibration.py`, `calibration_io.py` | a proposal calibrated on one instance applies to another |
| `by_name` lookup | `calibration_io.apply_proposals` | wrong profile updated |
| stale-proposal check | `calibration_io.apply_proposals` | baseline compared against the wrong instance's value |
| `reliability_evidence` | catalog metadata | provenance cannot say which endpoint produced the evidence |
| `EmpiricalSuccessModel._global` | `empirical.py` | two instances pooled into one success rate |
| `EmpiricalSuccessModel._feature` | `empirical.py` | same, per task class |
| `ModelRegistry` lookups and aliases | `models.py`, `catalog.py` | an alias resolves to an ambiguous target |
| `ExecutionResult.metadata["model"]` | `model_executor.py` | telemetry joins collapse instances |
| benchmark `EvaluationRun.model` | `evaluation.py`, `records_io.py` | historical runs cannot be attributed |
| `--accept` argument values | `cli.py` | accepting a name accepts every instance of it |

The `--accept` row deserves particular attention: today `--accept qwen3:8b` names one model. Under
instances it would silently accept every instance serving those weights unless the argument takes
compound identities.

**Migration semantics.** A bare name in a proposal or a run predates instances and cannot be
attributed to one. It must be rejected on load rather than defaulted to an implicit instance --
defaulting is precisely the silent aliasing this design exists to prevent. Historical evidence
either gets re-attributed deliberately or is excluded from calibration.

## Open questions

- **Do per-instance timeout/retry policies belong here?** They fit naturally, and they are also
  scope creep for a schema change. Recommend deferring until the identity change has landed.
- **Should `kind` be inferred from the instance id when unambiguous?** Convenient, and exactly
  the kind of implicitness that made `provider` overloaded in the first place. Recommend not.

## Sequencing

Not to be implemented before the first real calibration benchmark. That run may show whether
serving-instance effects are large enough to justify the extra estimator level immediately, or
whether the two-level estimator is adequate for now. Building the identity change first would
mean guessing at exactly the question the benchmark answers.
