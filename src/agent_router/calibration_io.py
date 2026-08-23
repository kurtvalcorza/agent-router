"""Read and write calibration proposals, and apply accepted ones to a candidate catalog.

Applying is a separate, explicit act from proposing. This module never writes to the
catalog it is given: like ``catalog sync``, it produces a *candidate* for review.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path

from .calibration import CalibrationProposal
from .catalog import ModelCatalog

__all__ = [
    "AppliedCalibration",
    "CalibrationIOError",
    "apply_proposals",
    "load_proposals",
    "write_proposals",
]

# The review state stamped into the catalog. It records that a human ran the apply step;
# it does not claim anyone read the evidence.
APPLIED_REVIEW_STATE = "applied-by-explicit-action"


class CalibrationIOError(RuntimeError):
    pass


def write_proposals(path: str | Path, proposals: Iterable[CalibrationProposal]) -> None:
    payload = [proposal.as_dict() for proposal in proposals]
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_proposals(path: str | Path) -> tuple[dict, ...]:
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationIOError(f"failed to load proposals {source}: {exc}") from exc
    if not isinstance(data, list):
        raise CalibrationIOError("proposal file must contain a JSON array")
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise CalibrationIOError(f"proposals[{index}] must be an object")
        for key in ("model", "proposed_reliability", "status", "evidence_ref"):
            if key not in item:
                raise CalibrationIOError(f"proposals[{index}] is missing {key!r}")
        value = item["proposed_reliability"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CalibrationIOError(f"proposals[{index}].proposed_reliability must be a number")
        if not 0.0 <= float(value) <= 1.0:
            raise CalibrationIOError(
                f"proposals[{index}].proposed_reliability must be between 0 and 1"
            )
    return tuple(data)


class AppliedCalibration:
    """Result of applying accepted proposals, with what was skipped and why."""

    def __init__(
        self,
        catalog: ModelCatalog,
        applied: Sequence[tuple[str, float, float]],
        skipped: Sequence[tuple[str, str]],
    ) -> None:
        self.catalog = catalog
        self.applied = tuple(applied)
        self.skipped = tuple(skipped)


def apply_proposals(
    catalog: ModelCatalog,
    proposals: Iterable[dict],
    *,
    accept: Sequence[str] | None = None,
    accept_all: bool = False,
    allow_insufficient_evidence: bool = False,
) -> AppliedCalibration:
    """Return a NEW catalog with accepted reliabilities applied.

    Acceptance is never implicit: a proposal is applied only if its model is named in
    ``accept`` or ``accept_all`` is set. Proposals whose status is not
    ``REVIEW_REQUIRED`` are skipped unless explicitly allowed, so thin evidence cannot
    move a policy value by default.
    """
    accepted_names = set(accept or ())
    by_name = {profile.name: profile for profile in catalog.profiles}

    applied: list[tuple[str, float, float]] = []
    skipped: list[tuple[str, str]] = []
    updates: dict[str, tuple[float, dict]] = {}

    for proposal in proposals:
        name = proposal["model"]
        if not (accept_all or name in accepted_names):
            skipped.append((name, "not accepted"))
            continue
        if name not in by_name:
            skipped.append((name, "not present in the catalog"))
            continue
        status = proposal.get("status")
        if status != "REVIEW_REQUIRED" and not allow_insufficient_evidence:
            skipped.append((name, f"status {status!r}"))
            continue

        proposed = float(proposal["proposed_reliability"])
        current = by_name[name].reliability
        evidence = {
            "proposed_by": "agent-router-calibration",
            "method": proposal.get("method"),
            "method_version": proposal.get("method_version"),
            "evidence_ref": proposal["evidence_ref"],
            "successes": proposal.get("successes"),
            "trials": proposal.get("trials"),
            "credible_interval": proposal.get("credible_interval"),
            "posterior_mean": proposal.get("posterior_mean"),
            "previous_reliability": current,
            "review_state": APPLIED_REVIEW_STATE,
        }
        if proposal.get("warnings"):
            evidence["warnings"] = list(proposal["warnings"])
        updates[name] = (proposed, {k: v for k, v in evidence.items() if v is not None})
        applied.append((name, current, proposed))

    if not updates:
        return AppliedCalibration(catalog, applied, skipped)

    profiles = []
    for profile in catalog.profiles:
        if profile.name not in updates:
            profiles.append(profile)
            continue
        proposed, evidence = updates[profile.name]
        metadata = dict(profile.metadata)
        metadata["reliability_evidence"] = evidence
        profiles.append(replace(profile, reliability=proposed, metadata=metadata))

    return AppliedCalibration(
        replace(catalog, profiles=tuple(profiles)),
        applied,
        skipped,
    )
