from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace

from .catalog import ModelCatalog
from .models import ModelProfile
from .pricing import PricingProfile


@dataclass(frozen=True, slots=True)
class ProviderModelSnapshot:
    provider: str
    name: str
    context_window: int | None = None
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
    pricing: PricingProfile | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider or not self.name:
            raise ValueError("provider and name must be non-empty")
        if self.context_window is not None and self.context_window < 1:
            raise ValueError("context_window must be positive")
        for value, field_name in (
            (self.input_cost_per_million, "input_cost_per_million"),
            (self.output_cost_per_million, "output_cost_per_million"),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must be non-negative")


@dataclass(frozen=True, slots=True)
class CatalogChange:
    model: str
    field: str
    before: object
    after: object


@dataclass(frozen=True, slots=True)
class CatalogDiff:
    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    changed: tuple[CatalogChange, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.added and not self.removed and not self.changed


@dataclass(frozen=True, slots=True)
class SyncResult:
    candidate: ModelCatalog
    diff: CatalogDiff
    warnings: tuple[str, ...] = ()


class CatalogPromotionError(RuntimeError):
    pass


def synchronize_catalog(
    current: ModelCatalog,
    snapshots: Iterable[ProviderModelSnapshot],
    *,
    pricing_as_of: str | None = None,
    pricing_source: str | None = None,
) -> SyncResult:
    """Build a candidate catalog without changing locally governed policy fields."""

    by_key = {(profile.provider, profile.name): profile for profile in current.profiles}
    snapshot_by_key: dict[tuple[str, str], ProviderModelSnapshot] = {}
    warnings: list[str] = []

    for snapshot in snapshots:
        key = (snapshot.provider, snapshot.name)
        if key in snapshot_by_key:
            raise ValueError(f"duplicate provider snapshot for {snapshot.provider}/{snapshot.name}")
        snapshot_by_key[key] = snapshot
        if key not in by_key:
            warnings.append(
                f"unmanaged provider model discovered: {snapshot.provider}/{snapshot.name}"
            )

    profiles: list[ModelProfile] = []
    for profile in current.profiles:
        snapshot = snapshot_by_key.get((profile.provider, profile.name))
        if snapshot is None:
            profiles.append(profile)
            warnings.append(
                "catalog model missing from provider snapshot: "
                f"{profile.provider}/{profile.name}"
            )
            continue

        metadata = dict(profile.metadata)
        if snapshot.metadata:
            metadata["provider_snapshot"] = dict(snapshot.metadata)

        pricing = snapshot.pricing
        standard_input = (
            pricing.standard_input
            if pricing is not None
            else snapshot.input_cost_per_million
        )
        standard_output = (
            pricing.standard_output
            if pricing is not None
            else snapshot.output_cost_per_million
        )

        # A flat-cost-only snapshot must also update the structured ``pricing`` profile,
        # otherwise ``pricing_profile`` and serialization keep the stale rate while the diff
        # advertises a change that never takes effect. Preserve the profile's other pricing
        # dimensions (cache/batch/long-context) and update only the standard rates.
        if pricing is None and (
            snapshot.input_cost_per_million is not None
            or snapshot.output_cost_per_million is not None
        ):
            base_pricing = profile.pricing or profile.pricing_profile
            pricing = replace(
                base_pricing,
                standard_input=(
                    standard_input
                    if standard_input is not None
                    else base_pricing.standard_input
                ),
                standard_output=(
                    standard_output
                    if standard_output is not None
                    else base_pricing.standard_output
                ),
            )

        profiles.append(
            replace(
                profile,
                context_window=(
                    snapshot.context_window
                    if snapshot.context_window is not None
                    else profile.context_window
                ),
                input_cost_per_million=(
                    standard_input
                    if standard_input is not None
                    else profile.input_cost_per_million
                ),
                output_cost_per_million=(
                    standard_output
                    if standard_output is not None
                    else profile.output_cost_per_million
                ),
                pricing=pricing if pricing is not None else profile.pricing,
                metadata=metadata,
            )
        )

    metadata = replace(
        current.metadata,
        pricing_as_of=pricing_as_of or current.metadata.pricing_as_of,
        pricing_source=pricing_source or current.metadata.pricing_source,
    )
    candidate = ModelCatalog(
        metadata=metadata,
        profiles=tuple(profiles),
        aliases=dict(current.aliases),
    )
    return SyncResult(
        candidate=candidate,
        diff=diff_catalogs(current, candidate),
        warnings=tuple(warnings),
    )


def diff_catalogs(before: ModelCatalog, after: ModelCatalog) -> CatalogDiff:
    old = {profile.name: profile for profile in before.profiles}
    new = {profile.name: profile for profile in after.profiles}
    added = tuple(sorted(new.keys() - old.keys()))
    removed = tuple(sorted(old.keys() - new.keys()))
    changed: list[CatalogChange] = []

    fields = (
        "provider",
        "context_window",
        "input_cost_per_million",
        "output_cost_per_million",
        "pricing",
        "reliability",
        "execution_classes",
        "capabilities",
    )
    for name in sorted(old.keys() & new.keys()):
        left = old[name]
        right = new[name]
        for field_name in fields:
            left_value = getattr(left, field_name)
            right_value = getattr(right, field_name)
            if left_value != right_value:
                changed.append(CatalogChange(name, field_name, left_value, right_value))

    if before.aliases != after.aliases:
        changed.append(CatalogChange("<catalog>", "aliases", before.aliases, after.aliases))
    if before.metadata.pricing_as_of != after.metadata.pricing_as_of:
        changed.append(
            CatalogChange(
                "<catalog>",
                "pricing_as_of",
                before.metadata.pricing_as_of,
                after.metadata.pricing_as_of,
            )
        )
    if before.metadata.pricing_source != after.metadata.pricing_source:
        changed.append(
            CatalogChange(
                "<catalog>",
                "pricing_source",
                before.metadata.pricing_source,
                after.metadata.pricing_source,
            )
        )

    return CatalogDiff(added=added, removed=removed, changed=tuple(changed))


def validate_promotion(current: ModelCatalog, candidate: ModelCatalog) -> None:
    old = {profile.name: profile for profile in current.profiles}
    new = {profile.name: profile for profile in candidate.profiles}

    if old.keys() != new.keys():
        raise CatalogPromotionError("promotion cannot add or remove models without local review")
    if current.aliases != candidate.aliases:
        raise CatalogPromotionError("promotion cannot change aliases without local review")

    for name, before in old.items():
        after = new[name]
        if before.provider != after.provider:
            raise CatalogPromotionError(f"promotion cannot change provider for {name}")
        if before.execution_classes != after.execution_classes:
            raise CatalogPromotionError(f"promotion cannot change execution classes for {name}")
        if before.capabilities != after.capabilities:
            raise CatalogPromotionError(f"promotion cannot change capabilities for {name}")
        if before.reliability != after.reliability:
            raise CatalogPromotionError(f"promotion cannot change reliability for {name}")


def promote_candidate(current: ModelCatalog, candidate: ModelCatalog) -> ModelCatalog:
    validate_promotion(current, candidate)
    return candidate
