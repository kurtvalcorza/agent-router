from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .provenance import PricingRecord, pricing_record_to_dict


def write_pricing_records(path: str | Path, records: Iterable[PricingRecord]) -> None:
    target = Path(path)
    payload = [pricing_record_to_dict(record) for record in records]
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
