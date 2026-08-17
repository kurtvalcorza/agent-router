from __future__ import annotations

import json
from pathlib import Path

from .catalog import ModelCatalog


def catalog_to_dict(catalog: ModelCatalog) -> dict[str, object]:
    result: dict[str, object] = {
        "version": catalog.metadata.version,
        "models": [],
    }
    if catalog.metadata.pricing_as_of is not None:
        result["pricing_as_of"] = catalog.metadata.pricing_as_of
    if catalog.metadata.pricing_source is not None:
        result["pricing_source"] = catalog.metadata.pricing_source
    if catalog.metadata.metadata:
        result["metadata"] = dict(catalog.metadata.metadata)
    if catalog.aliases:
        result["aliases"] = dict(catalog.aliases)

    models: list[dict[str, object]] = []
    for profile in catalog.profiles:
        item: dict[str, object] = {
            "name": profile.name,
            "provider": profile.provider,
            "execution_classes": sorted(value.value for value in profile.execution_classes),
            "capabilities": sorted(value.value for value in profile.capabilities),
            "reliability": profile.reliability,
            "pricing": {
                "input_per_million": profile.input_cost_per_million,
                "output_per_million": profile.output_cost_per_million,
            },
        }
        if profile.context_window is not None:
            item["context_window"] = profile.context_window
        if profile.metadata:
            item["metadata"] = dict(profile.metadata)
        models.append(item)
    result["models"] = models
    return result


def write_catalog(path: str | Path, catalog: ModelCatalog) -> None:
    target = Path(path)
    suffix = target.suffix.lower()
    payload = catalog_to_dict(catalog)
    if suffix == ".json":
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "YAML output requires the optional 'catalog' dependency: "
                "pip install 'agent-router[catalog]'"
            ) from exc
        target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return
    raise ValueError(f"unsupported catalog format: {suffix or '<none>'}")
