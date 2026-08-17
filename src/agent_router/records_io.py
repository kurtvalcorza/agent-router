from __future__ import annotations

import json
from pathlib import Path

from .pricing import PricingProfile
from .provenance import InventoryRecord, PricingRecord, SourceProvenance
from .reconcile import AvailabilityObservation, AvailabilityStatus


class RecordIOError(ValueError):
    pass


def write_inventory(path: str | Path, records: tuple[InventoryRecord, ...]) -> None:
    payload = [_inventory_to_dict(record) for record in records]
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_inventory(path: str | Path) -> tuple[InventoryRecord, ...]:
    data = _load_list(path)
    return tuple(_inventory_from_dict(item) for item in data)


def write_pricing(path: str | Path, records: tuple[PricingRecord, ...]) -> None:
    payload = [_pricing_to_dict(record) for record in records]
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_pricing(path: str | Path) -> tuple[PricingRecord, ...]:
    data = _load_list(path)
    return tuple(_pricing_from_dict(item) for item in data)


def write_availability_state(
    path: str | Path,
    observations: tuple[AvailabilityObservation, ...],
) -> None:
    payload = [
        {
            "provider": item.provider,
            "model": item.model,
            "status": item.status.value,
            "consecutive_missing": item.consecutive_missing,
            "last_seen_at": item.last_seen_at,
            "metadata": item.metadata,
        }
        for item in observations
    ]
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_availability_state(path: str | Path) -> tuple[AvailabilityObservation, ...]:
    data = _load_list(path)
    observations: list[AvailabilityObservation] = []
    for item in data:
        if not isinstance(item, dict):
            raise RecordIOError("availability state entries must be objects")
        try:
            observations.append(
                AvailabilityObservation(
                    provider=str(item["provider"]),
                    model=str(item["model"]),
                    status=AvailabilityStatus(str(item["status"])),
                    consecutive_missing=int(item.get("consecutive_missing", 0)),
                    last_seen_at=item.get("last_seen_at"),
                    metadata=dict(item.get("metadata", {})),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RecordIOError(f"invalid availability state entry: {item!r}") from exc
    return tuple(observations)


def _load_list(path: str | Path) -> list[object]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RecordIOError("record file root must be a JSON list")
    return data


def _provenance_to_dict(value: SourceProvenance | None) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "source": value.source,
        "retrieved_at": value.retrieved_at,
        "content_hash": value.content_hash,
        "parser_version": value.parser_version,
    }


def _provenance_from_dict(value: object) -> SourceProvenance | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise RecordIOError("provenance must be an object")
    return SourceProvenance(
        source=str(value["source"]),
        retrieved_at=str(value["retrieved_at"]),
        content_hash=str(value["content_hash"]),
        parser_version=str(value.get("parser_version", "1")),
    )


def _inventory_to_dict(record: InventoryRecord) -> dict[str, object]:
    return {
        "provider": record.provider,
        "model_id": record.model_id,
        "available": record.available,
        "created_at": record.created_at,
        "owned_by": record.owned_by,
        "metadata": record.metadata,
        "provenance": _provenance_to_dict(record.provenance),
    }


def _inventory_from_dict(item: object) -> InventoryRecord:
    if not isinstance(item, dict):
        raise RecordIOError("inventory entries must be objects")
    return InventoryRecord(
        provider=str(item["provider"]),
        model_id=str(item["model_id"]),
        available=bool(item.get("available", True)),
        created_at=item.get("created_at") if isinstance(item.get("created_at"), int) else None,
        owned_by=item.get("owned_by") if isinstance(item.get("owned_by"), str) else None,
        metadata=dict(item.get("metadata", {})),
        provenance=_provenance_from_dict(item.get("provenance")),
    )


def _pricing_to_dict(record: PricingRecord) -> dict[str, object]:
    return {
        "provider": record.provider,
        "model_id": record.model_id,
        "pricing": record.pricing.__dict__ if hasattr(record.pricing, "__dict__") else {
            "standard_input": record.pricing.standard_input,
            "standard_output": record.pricing.standard_output,
            "cached_input": record.pricing.cached_input,
            "cache_write": record.pricing.cache_write,
            "batch_input": record.pricing.batch_input,
            "batch_output": record.pricing.batch_output,
            "long_context_input": record.pricing.long_context_input,
            "long_context_output": record.pricing.long_context_output,
            "long_context_threshold": record.pricing.long_context_threshold,
        },
        "effective_at": record.effective_at,
        "metadata": record.metadata,
        "provenance": _provenance_to_dict(record.provenance),
    }


def _pricing_from_dict(item: object) -> PricingRecord:
    if not isinstance(item, dict):
        raise RecordIOError("pricing entries must be objects")
    raw = item.get("pricing")
    if not isinstance(raw, dict):
        raise RecordIOError("pricing field must be an object")
    return PricingRecord(
        provider=str(item["provider"]),
        model_id=str(item["model_id"]),
        pricing=PricingProfile(**raw),
        effective_at=item.get("effective_at") if isinstance(item.get("effective_at"), str) else None,
        metadata=dict(item.get("metadata", {})),
        provenance=_provenance_from_dict(item.get("provenance")),
    )
