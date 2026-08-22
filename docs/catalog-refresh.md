# Catalog refresh workflow

The operational refresh path keeps provider observations, pricing, availability state, and the pinned catalog separate.

This workflow reconciles against live provider state, so it needs credentials and a catalog
naming real models. `config/models.yaml` below is the catalog you create by copying
`config/models.example.yaml` and replacing its placeholder values; see the README's benchmark
section. `agent-router catalog check config/models.example.yaml` runs without either.

```bash
# 1. Fetch live OpenAI model inventory.
agent-router provider fetch openai \
  --output .agent-router/openai-inventory.json

# 2. Fetch authoritative Anthropic pricing.
agent-router pricing fetch anthropic \
  --model-map config/anthropic-model-map.example.json \
  --output .agent-router/anthropic-pricing.json

# 3. Reconcile managed catalog models with observed provider state.
agent-router catalog reconcile config/models.yaml \
  --inventory .agent-router/openai-inventory.json \
  --pricing .agent-router/anthropic-pricing.json \
  --previous-state .agent-router/availability-state.json \
  --state-output .agent-router/availability-state.next.json \
  --snapshots-output .agent-router/provider-snapshots.json

# 4. Build a candidate catalog; the pinned catalog is never overwritten.
agent-router catalog sync config/models.yaml \
  .agent-router/provider-snapshots.json \
  --output config/models.candidate.yaml

# 5. Review the candidate.
agent-router catalog diff config/models.yaml config/models.candidate.yaml
```

On the first reconciliation run, omit `--previous-state`. Managed models from the pinned catalog are still seeded into reconciliation, so a model absent from the first provider inventory is recorded as `suspect_missing` rather than disappearing from observation.

Availability transitions require repeated absence by default:

```text
available -> suspect_missing -> confirmed_unavailable
```

Use `--missing-threshold N` to change the confirmation threshold. Reappearance resets the missing count and returns the model to `available`.

Provider inventory and pricing files carry provenance metadata. Catalog synchronization may stage operational provider facts, but locally reviewed capabilities, reliability, execution classes, aliases, and model membership remain protected by promotion validation.
