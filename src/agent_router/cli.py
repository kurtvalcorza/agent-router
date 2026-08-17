from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .catalog import CatalogError, load_catalog
from .catalog_sync import diff_catalogs, synchronize_catalog
from .pricing_io import write_pricing_records
from .pricing_sources import AnthropicPricingSource, PricingSourceError
from .serialize import write_catalog
from .snapshot import SnapshotError, load_snapshots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-router")
    subparsers = parser.add_subparsers(dest="command", required=True)

    catalog = subparsers.add_parser("catalog", help="inspect and synchronize model catalogs")
    catalog_sub = catalog.add_subparsers(dest="catalog_command", required=True)

    check = catalog_sub.add_parser("check", help="validate a catalog")
    check.add_argument("catalog")

    diff = catalog_sub.add_parser("diff", help="compare two catalogs")
    diff.add_argument("before")
    diff.add_argument("after")

    sync = catalog_sub.add_parser("sync", help="build a candidate catalog from provider snapshots")
    sync.add_argument("catalog")
    sync.add_argument("snapshots")
    sync.add_argument("--output", required=True)
    sync.add_argument("--pricing-as-of")
    sync.add_argument("--pricing-source")

    pricing = subparsers.add_parser("pricing", help="fetch normalized provider pricing")
    pricing_sub = pricing.add_subparsers(dest="pricing_command", required=True)
    pricing_fetch = pricing_sub.add_parser("fetch", help="fetch an authoritative pricing source")
    pricing_fetch.add_argument("provider", choices=["anthropic"])
    pricing_fetch.add_argument("--model-map", required=True)
    pricing_fetch.add_argument("--output", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "catalog" and args.catalog_command == "check":
            catalog = load_catalog(args.catalog)
            print(
                f"OK {args.catalog}: {len(catalog.profiles)} models, "
                f"{len(catalog.aliases)} aliases"
            )
            return 0

        if args.command == "catalog" and args.catalog_command == "diff":
            before = load_catalog(args.before)
            after = load_catalog(args.after)
            return _print_diff(diff_catalogs(before, after))

        if args.command == "catalog" and args.catalog_command == "sync":
            current = load_catalog(args.catalog)
            snapshots = load_snapshots(args.snapshots)
            result = synchronize_catalog(
                current,
                snapshots,
                pricing_as_of=args.pricing_as_of,
                pricing_source=args.pricing_source,
            )
            output = Path(args.output)
            if output.resolve() == Path(args.catalog).resolve():
                raise ValueError(
                    "catalog sync refuses to overwrite the pinned catalog; use a candidate path"
                )
            write_catalog(output, result.candidate)
            for warning in result.warnings:
                print(f"WARNING {warning}", file=sys.stderr)
            _print_diff(result.diff)
            print(f"candidate written to {output}")
            return 0

        if args.command == "pricing" and args.pricing_command == "fetch":
            mapping = _load_anthropic_model_map(args.model_map)
            source = AnthropicPricingSource(
                mapping["models"],
                long_context_thresholds=mapping["long_context_thresholds"],
            )
            records = source.fetch()
            write_pricing_records(args.output, records)
            print(f"wrote {len(records)} pricing records to {args.output}")
            return 0
    except (
        CatalogError,
        SnapshotError,
        PricingSourceError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 2

    parser.error("unsupported command")
    return 2


def _load_anthropic_model_map(path: str | Path) -> dict[str, dict[str, object]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("model map must be a JSON object")
    models = data.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("model map 'models' must be a non-empty object")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in models.items()):
        raise ValueError("model map 'models' must map display names to model IDs")

    thresholds = data.get("long_context_thresholds", {})
    if not isinstance(thresholds, dict):
        raise ValueError("long_context_thresholds must be an object")
    normalized_thresholds: dict[str, int] = {}
    for key, value in thresholds.items():
        if not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("long_context_thresholds must map display names to positive integers")
        if key not in models:
            raise ValueError(f"long-context model {key!r} is not present in the model map")
        normalized_thresholds[key] = value

    return {"models": dict(models), "long_context_thresholds": normalized_thresholds}


def _print_diff(diff) -> int:
    if diff.is_empty:
        print("no changes")
        return 0

    for name in diff.added:
        print(f"+ {name}")
    for name in diff.removed:
        print(f"- {name}")
    for change in diff.changed:
        print(f"~ {change.model} {change.field}: {change.before!r} -> {change.after!r}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
