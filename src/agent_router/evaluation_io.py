from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .evaluation import EvaluationCase, EvaluationRun


class EvaluationIOError(ValueError):
    pass


def load_cases(path: str | Path) -> tuple[EvaluationCase, ...]:
    data = _load_list(path)
    cases: list[EvaluationCase] = []
    seen: set[str] = set()
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise EvaluationIOError(f"cases[{index}] must be an object")
        try:
            case = EvaluationCase(
                id=str(item["id"]),
                task_kind=str(item["task_kind"]),
                minimum_quality=float(item.get("minimum_quality", 1.0)),
                metadata=dict(item.get("metadata", {})),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EvaluationIOError(f"invalid cases[{index}]: {item!r}") from exc
        if case.id in seen:
            raise EvaluationIOError(f"duplicate evaluation case id {case.id!r}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise EvaluationIOError("benchmark must contain at least one case")
    return tuple(cases)


def write_runs(path: str | Path, runs: Iterable[EvaluationRun]) -> None:
    payload = [
        {
            "case_id": run.case_id,
            "strategy": run.strategy,
            "model": run.model,
            "quality": run.quality,
            "cost_usd": run.cost_usd,
            "latency_seconds": run.latency_seconds,
            "escalations": run.escalations,
            "success": run.success,
            "metadata": run.metadata,
        }
        for run in runs
    ]
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_runs(path: str | Path) -> tuple[EvaluationRun, ...]:
    data = _load_list(path)
    runs: list[EvaluationRun] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise EvaluationIOError(f"runs[{index}] must be an object")
        try:
            runs.append(
                EvaluationRun(
                    case_id=str(item["case_id"]),
                    strategy=str(item["strategy"]),
                    model=item.get("model") if isinstance(item.get("model"), str) else None,
                    quality=float(item["quality"]),
                    cost_usd=float(item["cost_usd"]),
                    latency_seconds=float(item["latency_seconds"]),
                    escalations=int(item.get("escalations", 0)),
                    success=item.get("success") if isinstance(item.get("success"), bool) else None,
                    metadata=dict(item.get("metadata", {})),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EvaluationIOError(f"invalid runs[{index}]: {item!r}") from exc
    return tuple(runs)


def _load_list(path: str | Path) -> list[object]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationIOError(f"failed to load evaluation file {path}: {exc}") from exc
    if not isinstance(data, list):
        raise EvaluationIOError("evaluation file root must be a JSON list")
    return data
