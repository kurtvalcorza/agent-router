from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Protocol

from .catalog_sync import ProviderModelSnapshot
from .pricing import PricingProfile


class SnapshotFetcher(Protocol):
    def fetch(self) -> Iterable[ProviderModelSnapshot]: ...


class SnapshotError(RuntimeError):
    pass


def load_snapshots(path: str | Path) -> tuple[ProviderModelSnapshot, ...]:
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"failed to load snapshot file {source}: {exc}") from exc

    if not isinstance(data, list):
        raise SnapshotError("snapshot file must contain a JSON array")

    snapshots: list[ProviderModelSnapshot] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise SnapshotError(f"snapshots[{index}] must be an object")
        try:
            snapshots.append(
                ProviderModelSnapshot(
                    provider=_required_string(item, "provider", index),
                    name=_required_string(item, "name", index),
                    context_window=_optional_positive_int(item.get("context_window"), index),
                    input_cost_per_million=_optional_non_negative_number(
                        item.get("input_cost_per_million"), index, "input_cost_per_million"
                    ),
                    output_cost_per_million=_optional_non_negative_number(
                        item.get("output_cost_per_million"), index, "output_cost_per_million"
                    ),
                    pricing=_pricing(item.get("pricing"), index),
                    metadata=_metadata(item.get("metadata", {}), index),
                )
            )
        except (TypeError, ValueError) as exc:
            raise SnapshotError(f"invalid snapshots[{index}]: {exc}") from exc
    return tuple(snapshots)


def write_snapshots(path: str | Path, snapshots: Iterable[ProviderModelSnapshot]) -> None:
    target = Path(path)
    payload = [asdict(snapshot) for snapshot in snapshots]
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _required_string(item: dict[str, object], key: str, index: int) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value:
        raise SnapshotError(f"snapshots[{index}].{key} must be a non-empty string")
    return value


def _optional_positive_int(value: object, index: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SnapshotError(f"snapshots[{index}].context_window must be a positive integer")
    return value


def _optional_non_negative_number(value: object, index: int, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise SnapshotError(f"snapshots[{index}].{field_name} must be a non-negative number")
    return float(value)


def _pricing(value: object, index: int) -> PricingProfile | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise SnapshotError(f"snapshots[{index}].pricing must be an object")
    try:
        return PricingProfile(**value)
    except TypeError as exc:
        raise SnapshotError(f"snapshots[{index}].pricing contains unsupported fields") from exc


def _metadata(value: object, index: int) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise SnapshotError(f"snapshots[{index}].metadata must be an object with string keys")
    return dict(value)
