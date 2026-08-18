from dataclasses import replace

import pytest

from agent_router import (
    CatalogMetadata,
    CatalogPromotionError,
    ExecutionClass,
    ModelCatalog,
    ModelProfile,
    PricingProfile,
    ProviderModelSnapshot,
    Requirement,
    diff_catalogs,
    promote_candidate,
    synchronize_catalog,
)


def _catalog() -> ModelCatalog:
    return ModelCatalog(
        metadata=CatalogMetadata(
            version="1",
            pricing_as_of="2026-08-01",
            pricing_source="local",
        ),
        profiles=(
            ModelProfile(
                name="model-a",
                provider="provider-a",
                execution_classes={ExecutionClass.LIGHT_REASONING},
                capabilities={Requirement.SEMANTIC_REASONING},
                context_window=100_000,
                input_cost_per_million=1.0,
                output_cost_per_million=2.0,
                reliability=0.91,
                metadata={"owner": "local"},
            ),
        ),
        aliases={"fast": "model-a"},
    )


def test_sync_updates_operational_metadata_only() -> None:
    current = _catalog()
    result = synchronize_catalog(
        current,
        [
            ProviderModelSnapshot(
                provider="provider-a",
                name="model-a",
                context_window=200_000,
                input_cost_per_million=0.5,
                output_cost_per_million=1.5,
                metadata={"upstream_id": "abc"},
            )
        ],
        pricing_as_of="2026-08-17",
        pricing_source="provider-docs",
    )

    profile = result.candidate.profiles[0]
    assert profile.context_window == 200_000
    assert profile.input_cost_per_million == 0.5
    assert profile.output_cost_per_million == 1.5
    assert profile.reliability == 0.91
    assert profile.capabilities == {Requirement.SEMANTIC_REASONING}
    assert profile.execution_classes == {ExecutionClass.LIGHT_REASONING}
    assert profile.metadata["owner"] == "local"
    assert profile.metadata["provider_snapshot"] == {"upstream_id": "abc"}
    assert result.candidate.aliases == {"fast": "model-a"}
    assert result.candidate.metadata.pricing_as_of == "2026-08-17"


def test_unknown_provider_model_is_warning_not_auto_add() -> None:
    result = synchronize_catalog(
        _catalog(),
        [ProviderModelSnapshot(provider="provider-a", name="new-model")],
    )

    assert [profile.name for profile in result.candidate.profiles] == ["model-a"]
    assert "unmanaged provider model discovered: provider-a/new-model" in result.warnings
    assert "catalog model missing from provider snapshot: provider-a/model-a" in result.warnings


def test_duplicate_provider_snapshot_is_rejected() -> None:
    snapshot = ProviderModelSnapshot(provider="provider-a", name="model-a")

    with pytest.raises(ValueError, match="duplicate provider snapshot"):
        synchronize_catalog(_catalog(), [snapshot, snapshot])


def test_diff_reports_pricing_and_context_changes() -> None:
    before = _catalog()
    after = synchronize_catalog(
        before,
        [
            ProviderModelSnapshot(
                provider="provider-a",
                name="model-a",
                context_window=120_000,
                input_cost_per_million=0.8,
            )
        ],
    ).candidate

    diff = diff_catalogs(before, after)
    fields = {(change.model, change.field) for change in diff.changed}

    assert ("model-a", "context_window") in fields
    assert ("model-a", "input_cost_per_million") in fields


def test_promotion_rejects_capability_change() -> None:
    current = _catalog()
    changed_profile = replace(
        current.profiles[0],
        capabilities={Requirement.SEMANTIC_REASONING, Requirement.DEEP_PLANNING},
    )
    candidate = replace(current, profiles=(changed_profile,))

    with pytest.raises(CatalogPromotionError, match="capabilities"):
        promote_candidate(current, candidate)


def test_flat_cost_snapshot_updates_structured_pricing_profile() -> None:
    # Regression: a flat-cost-only snapshot updated the flat fields (and the diff) but left
    # the structured ``pricing`` profile stale, so cost estimation and serialization kept the
    # old rate that the diff claimed had changed.
    current = ModelCatalog(
        metadata=CatalogMetadata(version="1", pricing_as_of="2026-08-01", pricing_source="local"),
        profiles=(
            ModelProfile(
                name="model-a",
                provider="provider-a",
                execution_classes={ExecutionClass.LIGHT_REASONING},
                capabilities={Requirement.SEMANTIC_REASONING},
                pricing=PricingProfile(standard_input=2.0, standard_output=8.0, cached_input=0.2),
            ),
        ),
    )

    candidate = synchronize_catalog(
        current,
        [
            ProviderModelSnapshot(
                provider="provider-a",
                name="model-a",
                input_cost_per_million=99.0,
                output_cost_per_million=88.0,
            )
        ],
    ).candidate

    profile = candidate.profiles[0]
    # The structured profile the router actually uses now reflects the update...
    assert profile.pricing_profile.standard_input == 99.0
    assert profile.pricing_profile.standard_output == 88.0
    # ...while preserving other pricing dimensions...
    assert profile.pricing_profile.cached_input == 0.2
    # ...and the estimated cost matches the advertised change.
    assert profile.estimate_cost(input_tokens=1_000_000) == 99.0


def test_promotion_accepts_pricing_change() -> None:
    current = _catalog()
    candidate = synchronize_catalog(
        current,
        [
            ProviderModelSnapshot(
                provider="provider-a",
                name="model-a",
                input_cost_per_million=0.75,
            )
        ],
    ).candidate

    assert promote_candidate(current, candidate) is candidate
