from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from .pricing import PricingProfile


@dataclass(frozen=True, slots=True)
class SourceProvenance:
    source: str
    retrieved_at: str
    content_hash: str
    parser_version: str = "1"

    @classmethod
    def from_payload(
        cls,
        *,
        source: str,
        payload: object,
        retrieved_at: str | None = None,
        parser_version: str = "1",
    ) -> "SourceProvenance":
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return cls(
            source=source,
            retrieved_at=retrieved_at or datetime.now(timezone.utc).isoformat(),
            content_hash=digest,
            parser_version=parser_version,
        )


@dataclass(frozen=True, slots=True)
class InventoryRecord:
    provider: str
    model_id: str
    available: bool = True
    created_at: int | None = None
    owned_by: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    provenance: SourceProvenance | None = None


@dataclass(frozen=True, slots=True)
class PricingRecord:
    provider: str
    model_id: str
    pricing: PricingProfile
    effective_at: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    provenance: SourceProvenance | None = None


def inventory_record_to_dict(record: InventoryRecord) -> dict[str, object]:
    result = asdict(record)
    return result


def pricing_record_to_dict(record: PricingRecord) -> dict[str, object]:
    result = asdict(record)
    return result
