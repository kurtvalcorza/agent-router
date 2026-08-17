from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from .catalog_sync import ProviderModelSnapshot
from .provenance import InventoryRecord, PricingRecord


class AvailabilityStatus(str, Enum):
    AVAILABLE = "available"
    SUSPECT_MISSING = "suspect_missing"
    CONFIRMED_UNAVAILABLE = "confirmed_unavailable"


@dataclass(frozen=True, slots=True)
class AvailabilityObservation:
    provider: str
    model: str
    status: AvailabilityStatus
    consecutive_missing: int = 0
    last_seen_at: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    observations: tuple[AvailabilityObservation, ...]
    snapshots: tuple[ProviderModelSnapshot, ...]
    warnings: tuple[str, ...] = ()


def reconcile_records(
    inventory: Iterable[InventoryRecord],
    pricing: Iterable[PricingRecord] = (),
    *,
    previous: Iterable[AvailabilityObservation] = (),
    missing_threshold: int = 2,
) -> ReconciliationResult:
    if missing_threshold < 1:
        raise ValueError("missing_threshold must be at least 1")

    inventory_by_key = {(item.provider, item.model_id): item for item in inventory}
    pricing_by_key = {(item.provider, item.model_id): item for item in pricing}
    previous_by_key = {(item.provider, item.model): item for item in previous}
    keys = set(previous_by_key) | set(inventory_by_key) | set(pricing_by_key)

    observations: list[AvailabilityObservation] = []
    snapshots: list[ProviderModelSnapshot] = []
    warnings: list[str] = []

    for provider, model in sorted(keys):
        key = (provider, model)
        inv = inventory_by_key.get(key)
        prior = previous_by_key.get(key)

        if inv is not None and inv.available:
            status = AvailabilityStatus.AVAILABLE
            missing_count = 0
            last_seen = inv.provenance.retrieved_at
        else:
            missing_count = (prior.consecutive_missing if prior else 0) + 1
            status = (
                AvailabilityStatus.CONFIRMED_UNAVAILABLE
                if missing_count >= missing_threshold
                else AvailabilityStatus.SUSPECT_MISSING
            )
            last_seen = prior.last_seen_at if prior else None
            message = (
                f"model confirmed unavailable after repeated absence: {provider}/{model}"
                if status is AvailabilityStatus.CONFIRMED_UNAVAILABLE
                else f"model missing once: {provider}/{model}"
            )
            warnings.append(message)

        metadata: dict[str, object] = {}
        if inv is not None:
            metadata["inventory"] = dict(inv.metadata)
            metadata["inventory_provenance"] = _provenance_dict(inv.provenance)

        price = pricing_by_key.get(key)
        if price is not None:
            metadata["pricing_provenance"] = _provenance_dict(price.provenance)

        observations.append(
            AvailabilityObservation(
                provider=provider,
                model=model,
                status=status,
                consecutive_missing=missing_count,
                last_seen_at=last_seen,
                metadata=metadata,
            )
        )

        pricing_profile = price.pricing if price is not None else None
        snapshots.append(
            ProviderModelSnapshot(
                provider=provider,
                name=model,
                input_cost_per_million=(pricing_profile.standard_input if pricing_profile else None),
                output_cost_per_million=(pricing_profile.standard_output if pricing_profile else None),
                pricing=pricing_profile,
                metadata={
                    "availability_status": status.value,
                    "consecutive_missing": missing_count,
                    **metadata,
                },
            )
        )

    return ReconciliationResult(
        observations=tuple(observations),
        snapshots=tuple(snapshots),
        warnings=tuple(warnings),
    )


def _provenance_dict(provenance) -> dict[str, object]:
    return {
        "source": provenance.source,
        "retrieved_at": provenance.retrieved_at,
        "content_hash": provenance.content_hash,
        "parser_version": provenance.parser_version,
    }
