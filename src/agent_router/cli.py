from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .catalog import CatalogError, load_catalog
from .catalog_sync import diff_catalogs, synchronize_catalog
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
                raise ValueError("catalog sync refuses to overwrite the pinned catalog; use a candidate path")
            write_catalog(output, result.candidate)
            for warning in result.warnings:
                print(f"WARNING {warning}", file=sys.stderr)
            _print_diff(result.diff)
            print(f"candidate written to {output}")
            return 0
    except (CatalogError, SnapshotError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 2

    parser.error("unsupported command")
    return 2


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
