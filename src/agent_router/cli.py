from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .catalog import CatalogError, load_catalog
from .catalog_sync import diff_catalogs, synchronize_catalog
from .empirical import EmpiricalSuccessModel
from .empirical_io import EmpiricalModelIOError, write_empirical_model
from .evaluation import compare_strategies, evaluate_gate, summarize_strategy
from .evaluation_io import EvaluationIOError, load_cases, load_runs
from .inventory import AnthropicInventoryFetcher, OpenAIInventoryFetcher
from .pricing_io import write_pricing_records
from .pricing_sources import AnthropicPricingSource, OpenAIModelPricingSource, PricingSourceError
from .reconcile import reconcile_records
from .records_io import (
    RecordIOError,
    load_availability_state,
    load_inventory,
    load_pricing,
    write_availability_state,
    write_inventory,
)
from .serialize import write_catalog
from .snapshot import SnapshotError, load_snapshots, write_snapshots


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

    reconcile = catalog_sub.add_parser(
        "reconcile",
        help="combine provider inventory, pricing, and prior availability state into snapshots",
    )
    reconcile.add_argument("catalog")
    reconcile.add_argument("--inventory", required=True)
    reconcile.add_argument("--pricing")
    reconcile.add_argument("--previous-state")
    reconcile.add_argument("--state-output", required=True)
    reconcile.add_argument("--snapshots-output", required=True)
    reconcile.add_argument("--missing-threshold", type=int, default=2)

    pricing = subparsers.add_parser("pricing", help="fetch normalized provider pricing")
    pricing_sub = pricing.add_subparsers(dest="pricing_command", required=True)
    pricing_fetch = pricing_sub.add_parser("fetch", help="fetch an authoritative pricing source")
    pricing_fetch.add_argument("provider", choices=["anthropic", "openai"])
    pricing_fetch.add_argument("--model-map", required=True)
    pricing_fetch.add_argument("--output", required=True)

    provider = subparsers.add_parser("provider", help="fetch provider inventory metadata")
    provider_sub = provider.add_subparsers(dest="provider_command", required=True)
    provider_fetch = provider_sub.add_parser("fetch", help="fetch provider model inventory")
    provider_fetch.add_argument("provider", choices=["openai", "anthropic"])
    provider_fetch.add_argument("--output", required=True)

    evaluation = subparsers.add_parser("evaluation", help="train, summarize, and gate benchmark results")
    evaluation_sub = evaluation.add_subparsers(dest="evaluation_command", required=True)
    report = evaluation_sub.add_parser("report", help="compare a strategy with a baseline")
    report.add_argument("--cases", required=True)
    report.add_argument("--runs", required=True)
    report.add_argument("--strategy", default="router")
    report.add_argument("--baseline", default="always-strong")
    report.add_argument("--minimum-cost-savings", type=float, default=0.0)
    report.add_argument("--maximum-quality-loss", type=float, default=0.0)
    report.add_argument("--maximum-success-rate-loss", type=float, default=0.0)

    train = evaluation_sub.add_parser(
        "train-empirical",
        help="fit an empirical success model from historical benchmark runs",
    )
    train.add_argument("--cases", required=True)
    train.add_argument("--runs", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--prior-alpha", type=float, default=1.0)
    train.add_argument("--prior-beta", type=float, default=1.0)
    train.add_argument("--feature-weight", type=float, default=3.0)

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

        if args.command == "catalog" and args.catalog_command == "reconcile":
            catalog = load_catalog(args.catalog)
            inventory = load_inventory(args.inventory)
            pricing = load_pricing(args.pricing) if args.pricing else ()
            previous = load_availability_state(args.previous_state) if args.previous_state else ()
            expected = [(profile.provider, profile.name) for profile in catalog.profiles]
            result = reconcile_records(
                inventory,
                pricing,
                previous=previous,
                expected=expected,
                missing_threshold=args.missing_threshold,
            )
            write_availability_state(args.state_output, result.observations)
            write_snapshots(args.snapshots_output, result.snapshots)
            for warning in result.warnings:
                print(f"WARNING {warning}", file=sys.stderr)
            print(
                f"wrote {len(result.observations)} availability observations to "
                f"{args.state_output}"
            )
            print(f"wrote {len(result.snapshots)} provider snapshots to {args.snapshots_output}")
            return 0

        if args.command == "pricing" and args.pricing_command == "fetch":
            mapping = _load_model_map(args.model_map)
            if args.provider == "anthropic":
                source = AnthropicPricingSource(
                    mapping["models"],
                    long_context_thresholds=mapping.get("long_context_thresholds", {}),
                )
            else:
                source = OpenAIModelPricingSource(mapping["models"])
            records = source.fetch()
            write_pricing_records(args.output, records)
            print(f"wrote {len(records)} pricing records to {args.output}")
            return 0

        if args.command == "provider" and args.provider_command == "fetch":
            fetcher = (
                OpenAIInventoryFetcher.from_env()
                if args.provider == "openai"
                else AnthropicInventoryFetcher.from_env()
            )
            records = fetcher.fetch()
            write_inventory(args.output, records)
            print(f"wrote {len(records)} inventory records to {args.output}")
            return 0

        if args.command == "evaluation" and args.evaluation_command == "train-empirical":
            cases = load_cases(args.cases)
            runs = load_runs(args.runs)
            model = EmpiricalSuccessModel.fit(
                cases,
                runs,
                prior_alpha=args.prior_alpha,
                prior_beta=args.prior_beta,
                feature_weight=args.feature_weight,
            )
            write_empirical_model(args.output, model)
            print(f"wrote empirical routing model to {args.output}")
            return 0

        if args.command == "evaluation" and args.evaluation_command == "report":
            cases = load_cases(args.cases)
            runs = load_runs(args.runs)
            strategy = summarize_strategy(cases, runs, strategy=args.strategy)
            baseline = summarize_strategy(cases, runs, strategy=args.baseline)
            comparison = compare_strategies(strategy, baseline)
            passed, failures = evaluate_gate(
                comparison,
                minimum_cost_savings=args.minimum_cost_savings,
                maximum_quality_loss=args.maximum_quality_loss,
                maximum_success_rate_loss=args.maximum_success_rate_loss,
            )
            print(
                f"{strategy.strategy}: success={strategy.success_rate:.3f} "
                f"quality={strategy.mean_quality:.3f} cost=${strategy.total_cost_usd:.6f} "
                f"latency={strategy.mean_latency_seconds:.3f}s "
                f"escalation_rate={strategy.escalation_rate:.3f}"
            )
            print(
                f"vs {baseline.strategy}: savings={comparison.cost_savings_fraction:.3f} "
                f"quality_delta={comparison.quality_delta:+.3f} "
                f"success_delta={comparison.success_rate_delta:+.3f} "
                f"latency_delta={comparison.latency_delta_seconds:+.3f}s"
            )
            if not passed:
                for failure in failures:
                    print(f"FAIL {failure}", file=sys.stderr)
                return 1
            print("PASS evaluation gate")
            return 0
    except (
        CatalogError,
        SnapshotError,
        RecordIOError,
        EvaluationIOError,
        EmpiricalModelIOError,
        PricingSourceError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 2

    parser.error("unsupported command")
    return 2


def _load_model_map(path: str | Path) -> dict[str, dict[str, object]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("model map must be a JSON object")
    models = data.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("model map 'models' must be a non-empty object")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in models.items()):
        raise ValueError("model map 'models' must map string keys to string values")

    thresholds = data.get("long_context_thresholds", {})
    if not isinstance(thresholds, dict):
        raise ValueError("long_context_thresholds must be an object")
    normalized_thresholds: dict[str, int] = {}
    for key, value in thresholds.items():
        if not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("long_context_thresholds must map names to positive integers")
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
