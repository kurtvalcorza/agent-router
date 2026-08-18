from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import ModelProfile, ModelRegistry
from .pricing import PricingProfile
from .types import ExecutionClass, Requirement


class CatalogError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CatalogMetadata:
    version: str
    pricing_as_of: str | None = None
    pricing_source: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelCatalog:
    metadata: CatalogMetadata
    profiles: tuple[ModelProfile, ...]
    aliases: dict[str, str] = field(default_factory=dict)

    def registry(self) -> ModelRegistry:
        registry = ModelRegistry(self.profiles)
        for alias, target in self.aliases.items():
            registry.register_alias(alias, target)
        return registry


def load_catalog(path: str | Path) -> ModelCatalog:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".json":
        return parse_catalog(json.loads(source.read_text(encoding="utf-8")))
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise CatalogError(
                "YAML catalogs require the optional 'catalog' dependency: "
                "pip install 'agent-router[catalog]'"
            ) from exc
        return parse_catalog(yaml.safe_load(source.read_text(encoding="utf-8")))
    raise CatalogError(f"unsupported catalog format: {suffix or '<none>'}")


def parse_catalog(data: object) -> ModelCatalog:
    if not isinstance(data, dict):
        raise CatalogError("catalog root must be an object")
    metadata = CatalogMetadata(
        version=_required_string(data, "version"),
        pricing_as_of=_optional_string(data, "pricing_as_of"),
        pricing_source=_optional_string(data, "pricing_source"),
        metadata=_object_dict(data.get("metadata", {}), "metadata"),
    )
    raw_models = data.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise CatalogError("catalog 'models' must be a non-empty list")
    profiles = tuple(_parse_profile(item, index) for index, item in enumerate(raw_models))
    names = {profile.name for profile in profiles}
    if len(names) != len(profiles):
        raise CatalogError("model names must be unique")
    return ModelCatalog(
        metadata=metadata,
        profiles=profiles,
        aliases=_parse_aliases(data.get("aliases", {}), names),
    )


def _parse_profile(item: object, index: int) -> ModelProfile:
    if not isinstance(item, dict):
        raise CatalogError(f"models[{index}] must be an object")
    prefix = f"models[{index}]."
    pricing_data = item.get("pricing", {})
    if not isinstance(pricing_data, dict):
        raise CatalogError(prefix + "pricing must be an object")
    pp = prefix + "pricing."
    standard_input = _non_negative_number(
        pricing_data.get("input_per_million", 0.0), pp + "input_per_million"
    )
    standard_output = _non_negative_number(
        pricing_data.get("output_per_million", 0.0), pp + "output_per_million"
    )
    pricing = PricingProfile(
        standard_input=standard_input,
        standard_output=standard_output,
        cached_input=_optional_non_negative_number(
            pricing_data.get("cached_input_per_million"), pp + "cached_input_per_million"
        ),
        cache_write=_optional_non_negative_number(
            pricing_data.get("cache_write_per_million"), pp + "cache_write_per_million"
        ),
        batch_input=_optional_non_negative_number(
            pricing_data.get("batch_input_per_million"), pp + "batch_input_per_million"
        ),
        batch_output=_optional_non_negative_number(
            pricing_data.get("batch_output_per_million"), pp + "batch_output_per_million"
        ),
        long_context_input=_optional_non_negative_number(
            pricing_data.get("long_context_input_per_million"),
            pp + "long_context_input_per_million",
        ),
        long_context_output=_optional_non_negative_number(
            pricing_data.get("long_context_output_per_million"),
            pp + "long_context_output_per_million",
        ),
        long_context_threshold=_optional_positive_int(pricing_data, "long_context_threshold", pp),
    )
    return ModelProfile(
        name=_required_string(item, "name", prefix),
        provider=_required_string(item, "provider", prefix),
        execution_classes=_enum_set(
            item.get("execution_classes"), ExecutionClass, prefix + "execution_classes"
        ),
        capabilities=_enum_set(item.get("capabilities", []), Requirement, prefix + "capabilities"),
        context_window=_optional_positive_int(item, "context_window", prefix),
        input_cost_per_million=standard_input,
        output_cost_per_million=standard_output,
        reliability=_bounded_number(item.get("reliability", 1.0), prefix + "reliability"),
        metadata=_object_dict(item.get("metadata", {}), prefix + "metadata"),
        pricing=pricing,
    )


def _parse_aliases(value: object, names: set[str]) -> dict[str, str]:
    if not isinstance(value, dict):
        raise CatalogError("catalog 'aliases' must be an object")
    aliases: dict[str, str] = {}
    for alias, target in value.items():
        if not isinstance(alias, str) or not alias:
            raise CatalogError("catalog aliases must use non-empty string keys")
        if not isinstance(target, str) or target not in names:
            raise CatalogError(f"alias {alias!r} targets unknown model {target!r}")
        if alias in names:
            raise CatalogError(f"alias {alias!r} conflicts with a model name")
        aliases[alias] = target
    return aliases


def _enum_set(value: object, enum_type: type[Any], path: str) -> set[Any]:
    if not isinstance(value, list) or not value:
        raise CatalogError(f"{path} must be a non-empty list")
    result = set()
    for raw in value:
        if not isinstance(raw, str):
            raise CatalogError(f"{path} entries must be strings")
        try:
            result.add(enum_type(raw))
        except ValueError as exc:
            raise CatalogError(f"invalid {path} value: {raw!r}") from exc
    return result


def _required_string(data: dict[str, Any], key: str, prefix: str = "") -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise CatalogError(f"{prefix}{key} must be a non-empty string")
    return value


def _optional_string(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise CatalogError(f"{key} must be a non-empty string when provided")
    return value


def _optional_positive_int(data: dict[str, Any], key: str, prefix: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CatalogError(f"{prefix}{key} must be a positive integer")
    return value


def _non_negative_number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise CatalogError(f"{path} must be a non-negative number")
    return float(value)


def _optional_non_negative_number(value: object, path: str) -> float | None:
    return None if value is None else _non_negative_number(value, path)


def _bounded_number(value: object, path: str) -> float:
    number = _non_negative_number(value, path)
    if number > 1:
        raise CatalogError(f"{path} must be between 0 and 1")
    return number


def _object_dict(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise CatalogError(f"{path} must be an object with string keys")
    return dict(value)
