from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .provenance import InventoryRecord, SourceProvenance


@dataclass(slots=True)
class OpenAIInventoryFetcher:
    client: object
    source: str = "https://api.openai.com/v1/models"
    parser_version: str = "1"

    @classmethod
    def from_env(cls) -> "OpenAIInventoryFetcher":
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI inventory fetching requires the optional 'openai' dependency: "
                "pip install 'agent-router[openai]'"
            ) from exc
        return cls(client=OpenAI())

    def fetch(self) -> tuple[InventoryRecord, ...]:
        response = self.client.models.list()
        models = list(_iter_models(response))
        payload = [_model_payload(model) for model in models]
        provenance = SourceProvenance.from_payload(
            source=self.source,
            payload=payload,
            parser_version=self.parser_version,
        )
        records = [
            InventoryRecord(
                provider="openai",
                model_id=str(_get(model, "id")),
                available=True,
                created_at=_optional_int(_get(model, "created", None)),
                owned_by=_optional_str(_get(model, "owned_by", None)),
                metadata={"object": _get(model, "object", "model")},
                provenance=provenance,
            )
            for model in models
            if _get(model, "id", None)
        ]
        return tuple(sorted(records, key=lambda record: record.model_id))


def _iter_models(response: object) -> Iterable[object]:
    data = _get(response, "data", None)
    if data is None:
        return response if isinstance(response, Iterable) else ()
    return data


def _model_payload(model: object) -> dict[str, object]:
    return {
        "id": _get(model, "id", None),
        "object": _get(model, "object", None),
        "created": _get(model, "created", None),
        "owned_by": _get(model, "owned_by", None),
    }


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
