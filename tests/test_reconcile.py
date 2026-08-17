from agent_router import (
    AvailabilityObservation,
    AvailabilityStatus,
    InventoryRecord,
    PricingProfile,
    PricingRecord,
    SourceProvenance,
    reconcile_records,
)


def provenance(source: str = "test") -> SourceProvenance:
    return SourceProvenance.from_payload(
        source=source,
        payload={"ok": True},
        retrieved_at="2026-08-17T00:00:00Z",
        parser_version="1",
    )


def test_present_model_is_available_and_resets_missing_count():
    result = reconcile_records(
        [InventoryRecord(provider="openai", model_id="m1", available=True, provenance=provenance())],
        previous=[
            AvailabilityObservation(
                provider="openai",
                model="m1",
                status=AvailabilityStatus.SUSPECT_MISSING,
                consecutive_missing=1,
            )
        ],
    )
    observation = result.observations[0]
    assert observation.status is AvailabilityStatus.AVAILABLE
    assert observation.consecutive_missing == 0


def test_single_absence_is_only_suspect():
    result = reconcile_records(
        [],
        previous=[AvailabilityObservation(provider="openai", model="m1", status=AvailabilityStatus.AVAILABLE)],
        missing_threshold=2,
    )
    assert result.observations[0].status is AvailabilityStatus.SUSPECT_MISSING
    assert result.observations[0].consecutive_missing == 1


def test_first_run_missing_managed_model_is_observed():
    result = reconcile_records(
        [],
        expected_models=[("openai", "m1")],
        missing_threshold=2,
    )
    assert result.observations[0].model == "m1"
    assert result.observations[0].status is AvailabilityStatus.SUSPECT_MISSING


def test_repeated_absence_confirms_unavailable():
    result = reconcile_records(
        [],
        previous=[
            AvailabilityObservation(
                provider="openai",
                model="m1",
                status=AvailabilityStatus.SUSPECT_MISSING,
                consecutive_missing=1,
            )
        ],
        missing_threshold=2,
    )
    assert result.observations[0].status is AvailabilityStatus.CONFIRMED_UNAVAILABLE
    assert result.observations[0].consecutive_missing == 2


def test_pricing_is_carried_into_catalog_snapshot():
    result = reconcile_records(
        [
            InventoryRecord(
                provider="openai",
                model_id="m1",
                available=True,
                provenance=provenance("inventory"),
            )
        ],
        [
            PricingRecord(
                provider="openai",
                model_id="m1",
                pricing=PricingProfile(
                    standard_input=1.5,
                    standard_output=5.0,
                    cached_input=0.15,
                ),
                provenance=provenance("pricing"),
            )
        ],
    )
    snapshot = result.snapshots[0]
    assert snapshot.input_cost_per_million == 1.5
    assert snapshot.output_cost_per_million == 5.0
    assert snapshot.pricing.cached_input == 0.15
    assert snapshot.metadata["availability_status"] == "available"
