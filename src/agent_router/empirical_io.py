from __future__ import annotations

import json
from pathlib import Path

from .empirical import EmpiricalSuccessModel


class EmpiricalModelIOError(ValueError):
    pass


def write_empirical_model(path: str | Path, model: EmpiricalSuccessModel) -> None:
    payload = {
        "version": 1,
        "prior_alpha": model.prior_alpha,
        "prior_beta": model.prior_beta,
        "feature_weight": model.feature_weight,
        "global": [
            {"model": name, "successes": counts[0], "trials": counts[1]}
            for name, counts in sorted(model._global.items())
        ],
        "feature": [
            {
                "model": model_name,
                "feature_key": feature_key,
                "successes": counts[0],
                "trials": counts[1],
            }
            for (model_name, feature_key), counts in sorted(model._feature.items())
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_empirical_model(path: str | Path) -> EmpiricalSuccessModel:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EmpiricalModelIOError(f"failed to load empirical model {path}: {exc}") from exc

    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise EmpiricalModelIOError("unsupported empirical model format")
    try:
        model = EmpiricalSuccessModel(
            prior_alpha=float(payload["prior_alpha"]),
            prior_beta=float(payload["prior_beta"]),
            feature_weight=float(payload["feature_weight"]),
        )
        global_counts: dict[str, tuple[int, int]] = {}
        for item in payload.get("global", []):
            if not isinstance(item, dict):
                raise TypeError
            name = str(item["model"])
            successes = int(item["successes"])
            trials = int(item["trials"])
            _validate_counts(successes, trials)
            global_counts[name] = (successes, trials)

        feature_counts: dict[tuple[str, str], tuple[int, int]] = {}
        for item in payload.get("feature", []):
            if not isinstance(item, dict):
                raise TypeError
            key = (str(item["model"]), str(item["feature_key"]))
            successes = int(item["successes"])
            trials = int(item["trials"])
            _validate_counts(successes, trials)
            feature_counts[key] = (successes, trials)
        model._global = global_counts
        model._feature = feature_counts
        return model
    except (KeyError, TypeError, ValueError) as exc:
        raise EmpiricalModelIOError("invalid empirical model payload") from exc


def _validate_counts(successes: int, trials: int) -> None:
    if successes < 0 or trials < 0 or successes > trials:
        raise ValueError("invalid success/trial counts")
