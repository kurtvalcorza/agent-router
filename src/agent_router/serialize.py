from __future__ import annotations

import json
from pathlib import Path

from .catalog import ModelCatalog


def catalog_to_dict(catalog: ModelCatalog) -> dict[str, object]:
    result: dict[str, object] = {"version": catalog.metadata.version, "models": []}
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
        pricing = profile.pricing_profile
        pricing_data: dict[str, object] = {
            "input_per_million": pricing.standard_input,
            "output_per_million": pricing.standard_output,
        }
        optional = {
            "cached_input_per_million": pricing.cached_input,
            "cache_write_per_million": pricing.cache_write,
            "batch_input_per_million": pricing.batch_input,
            "batch_output_per_million": pricing.batch_output,
            "long_context_input_per_million": pricing.long_context_input,
            "long_context_output_per_million": pricing.long_context_output,
            "long_context_threshold": pricing.long_context_threshold,
        }
        pricing_data.update({key: value for key, value in optional.items() if value is not None})
        item: dict[str, object] = {
            "name": profile.name,
            "provider": profile.provider,
            "execution_classes": sorted(value.value for value in profile.execution_classes),
            "capabilities": sorted(value.value for value in profile.capabilities),
            "reliability": profile.reliability,
            "pricing": pricing_data,
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
