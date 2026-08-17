from agent_router import AvailabilityStatus, PricingProfile
from agent_router.provenance import InventoryRecord, PricingRecord, SourceProvenance
from agent_router.records_io import (
    load_availability_state,
    load_inventory,
    load_pricing,
    write_availability_state,
    write_inventory,
    write_pricing,
)
from agent_router.reconcile import AvailabilityObservation


def provenance():
    return SourceProvenance(
        source="test-source",
        retrieved_at="2026-08-18T00:00:00+00:00",
        content_hash="abc123",
        parser_version="1",
    )


def test_inventory_and_pricing_round_trip(tmp_path):
    inventory_path = tmp_path / "inventory.json"
    pricing_path = tmp_path / "pricing.json"
    write_inventory(
        inventory_path,
        (InventoryRecord(provider="openai", model_id="model-a", provenance=provenance()),),
    )
    write_pricing(
        pricing_path,
        (
            PricingRecord(
                provider="openai",
                model_id="model-a",
                pricing=PricingProfile(
                    standard_input=1.0,
                    standard_output=4.0,
                    cached_input=0.1,
                ),
                provenance=provenance(),
            ),
        ),
    )

    inventory = load_inventory(inventory_path)
    pricing = load_pricing(pricing_path)

    assert inventory[0].model_id == "model-a"
    assert inventory[0].provenance.content_hash == "abc123"
    assert pricing[0].pricing.cached_input == 0.1


def test_availability_state_round_trip(tmp_path):
    path = tmp_path / "state.json"
    write_availability_state(
        path,
        (
            AvailabilityObservation(
                provider="openai",
                model="model-a",
                status=AvailabilityStatus.SUSPECT_MISSING,
                consecutive_missing=1,
            ),
        ),
    )

    state = load_availability_state(path)
    assert state[0].status is AvailabilityStatus.SUSPECT_MISSING
    assert state[0].consecutive_missing == 1
