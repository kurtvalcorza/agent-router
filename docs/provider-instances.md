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

## Open questions

- **Does the empirical feature key include the instance?** Splitting evidence per instance is
  more correct and halves the data behind each estimate. Probably yes, with shrinkage toward a
  model-level rate, but that is a modelling decision, not a schema one.
- **Do per-instance timeout/retry policies belong here?** They fit naturally, and they are also
  scope creep for a schema change. Recommend deferring until the identity change has landed.
- **Should `kind` be inferred from the instance id when unambiguous?** Convenient, and exactly
  the kind of implicitness that made `provider` overloaded in the first place. Recommend not.
