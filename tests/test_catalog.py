import json

import pytest

from agent_router import (
    CatalogError,
    ExecutionClass,
    Requirement,
    load_catalog,
    parse_catalog,
)


def sample_catalog() -> dict[str, object]:
    return {
        "version": "2026-08-17",
        "pricing_as_of": "2026-08-17",
        "pricing_source": "manual-test-fixture",
        "aliases": {"fast": "small-model"},
        "models": [
            {
                "name": "small-model",
                "provider": "provider-a",
                "execution_classes": ["light_reasoning"],
                "capabilities": ["semantic_reasoning"],
                "context_window": 128000,
                "reliability": 0.9,
                "pricing": {
                    "input_per_million": 0.25,
                    "output_per_million": 1.0,
                },
            }
        ],
    }


def test_parse_catalog_builds_registry_and_aliases() -> None:
    catalog = parse_catalog(sample_catalog())

    assert catalog.metadata.version == "2026-08-17"
    assert catalog.metadata.pricing_as_of == "2026-08-17"
    profile = catalog.registry().get("fast")
    assert profile.name == "small-model"
    assert profile.execution_classes == {ExecutionClass.LIGHT_REASONING}
    assert profile.capabilities == {Requirement.SEMANTIC_REASONING}
    assert profile.input_cost_per_million == 0.25


def test_json_catalog_loads_from_disk(tmp_path) -> None:
    path = tmp_path / "models.json"
    path.write_text(json.dumps(sample_catalog()), encoding="utf-8")

    catalog = load_catalog(path)

    assert catalog.registry().get("small-model").provider == "provider-a"


def test_unknown_alias_target_is_rejected() -> None:
    data = sample_catalog()
    data["aliases"] = {"fast": "missing-model"}

    with pytest.raises(CatalogError, match="unknown model"):
        parse_catalog(data)


def test_invalid_enum_value_is_rejected() -> None:
    data = sample_catalog()
    models = data["models"]
    assert isinstance(models, list)
    model = models[0]
    assert isinstance(model, dict)
    model["execution_classes"] = ["warp_speed"]

    with pytest.raises(CatalogError, match="invalid models\[0\].execution_classes value"):
        parse_catalog(data)


def test_duplicate_model_names_are_rejected() -> None:
    data = sample_catalog()
    models = data["models"]
    assert isinstance(models, list)
    models.append(dict(models[0]))

    with pytest.raises(CatalogError, match="model names must be unique"):
        parse_catalog(data)
