import json
from pathlib import Path

from agent_router.cli import main

CATALOG = {
    "version": "1",
    "pricing_as_of": "2026-08-01",
    "models": [
        {
            "name": "small-model",
            "provider": "openai",
            "execution_classes": ["light_reasoning"],
            "capabilities": ["semantic_reasoning"],
            "reliability": 0.9,
            "context_window": 100000,
            "pricing": {
                "input_per_million": 1.0,
                "output_per_million": 2.0,
            },
        }
    ],
}


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_catalog_check(tmp_path, capsys) -> None:
    catalog = tmp_path / "models.json"
    _write_json(catalog, CATALOG)

    code = main(["catalog", "check", str(catalog)])

    assert code == 0
    assert "OK" in capsys.readouterr().out


def test_catalog_diff_returns_one_when_changed(tmp_path, capsys) -> None:
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    _write_json(before, CATALOG)
    changed = json.loads(json.dumps(CATALOG))
    changed["models"][0]["pricing"]["input_per_million"] = 0.5
    _write_json(after, changed)

    code = main(["catalog", "diff", str(before), str(after)])

    assert code == 1
    assert "input_cost_per_million" in capsys.readouterr().out


def test_catalog_sync_writes_candidate_not_source(tmp_path) -> None:
    source = tmp_path / "models.json"
    snapshots = tmp_path / "snapshots.json"
    candidate = tmp_path / "candidate.json"
    _write_json(source, CATALOG)
    original = source.read_text(encoding="utf-8")
    _write_json(
        snapshots,
        [
            {
                "provider": "openai",
                "name": "small-model",
                "context_window": 200000,
                "input_cost_per_million": 0.5,
                "output_cost_per_million": 1.5,
            }
        ],
    )

    code = main(
        [
            "catalog",
            "sync",
            str(source),
            str(snapshots),
            "--output",
            str(candidate),
            "--pricing-as-of",
            "2026-08-17",
        ]
    )

    assert code == 0
    assert source.read_text(encoding="utf-8") == original
    assert candidate.exists()


def test_catalog_reconcile_writes_state_and_snapshots(tmp_path) -> None:
    # Regression: the CLI passed ``expected=`` while the function accepts ``expected_models=``,
    # so this command raised TypeError on every invocation and had zero test coverage.
    catalog = tmp_path / "models.json"
    inventory = tmp_path / "inventory.json"
    state_out = tmp_path / "state.json"
    snaps_out = tmp_path / "snapshots.json"
    _write_json(catalog, CATALOG)
    _write_json(
        inventory,
        [{"provider": "openai", "model_id": "small-model", "available": True}],
    )

    code = main(
        [
            "catalog",
            "reconcile",
            str(catalog),
            "--inventory",
            str(inventory),
            "--state-output",
            str(state_out),
            "--snapshots-output",
            str(snaps_out),
        ]
    )

    assert code == 0
    assert state_out.exists()
    assert snaps_out.exists()


def test_catalog_sync_refuses_overwrite(tmp_path, capsys) -> None:
    source = tmp_path / "models.json"
    snapshots = tmp_path / "snapshots.json"
    _write_json(source, CATALOG)
    _write_json(snapshots, [])

    code = main(
        [
            "catalog",
            "sync",
            str(source),
            str(snapshots),
            "--output",
            str(source),
        ]
    )

    assert code == 2
    assert "refuses to overwrite" in capsys.readouterr().err
