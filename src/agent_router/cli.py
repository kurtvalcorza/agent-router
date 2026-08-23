from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adaptive import AdaptivePolicy, PolicyMode
from .catalog import CatalogError, load_catalog
from .catalog_sync import diff_catalogs, synchronize_catalog
from .delegation import (
    DEFAULT_DELEGATION_THRESHOLD_TOKENS,
    estimate_tokens,
    parse_requirements,
    parse_risk,
    plan_delegation,
)
from .empirical import EmpiricalSuccessModel
from .empirical_io import EmpiricalModelIOError, write_empirical_model
from .evaluation import compare_strategies, evaluate_gate, summarize_strategy
from .evaluation_io import EvaluationIOError, load_cases, load_runs
from .inventory import AnthropicInventoryFetcher, OpenAIInventoryFetcher
from .model_executor import RoutedModelExecutor
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
from .runtime import RouterRuntime
from .serialize import write_catalog
from .snapshot import SnapshotError, load_snapshots, write_snapshots
from .types import Budget, ExecutionClass, Task


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-router")
    subparsers = parser.add_subparsers(dest="command", required=True)

    route = subparsers.add_parser(
        "route",
        help="decide which model should handle a task, and optionally run it",
        description=(
            "Host-agnostic delegation entry point. Defaults to --plan, which decides "
            "and prints without calling any provider. --execute makes real, billed "
            "provider calls."
        ),
    )
    route.add_argument("prompt", nargs="?", help="task prompt; omit or pass - to read stdin")
    route.add_argument("--catalog", required=True, help="path to a reviewed model catalog")
    mode_group = route.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--plan",
        dest="execute",
        action="store_false",
        help="decide only; no provider call, no spend (default)",
    )
    mode_group.add_argument(
        "--execute",
        dest="execute",
        action="store_true",
        help="run the task through the selected model; issues REAL billed provider calls",
    )
    route.set_defaults(execute=False)
    route.add_argument(
        "--requirements",
        default="semantic_reasoning",
        help=(
            "comma-separated, e.g. semantic_reasoning,long_context. Defaults to "
            "semantic_reasoning because a prose prompt handed to the router is "
            "semantic work; passing this replaces the default rather than adding to it"
        ),
    )
    route.add_argument("--risk", default="low", help="low, medium, or high (default: low)")
    route.add_argument(
        "--mode",
        default="balanced",
        choices=[m.value for m in PolicyMode],
        help="adaptive policy mode (default: balanced)",
    )
    route.add_argument("--kind", default="subtask", help="task kind label (default: subtask)")
    route.add_argument("--input-tokens", type=int, help="override the estimated input tokens")
    route.add_argument("--output-tokens", type=int, help="override the estimated output tokens")
    route.add_argument(
        "--threshold-tokens",
        type=int,
        default=DEFAULT_DELEGATION_THRESHOLD_TOKENS,
        help=(
            "below this estimated total, report delegate=false because answering "
            f"directly is cheaper (default: {DEFAULT_DELEGATION_THRESHOLD_TOKENS})"
        ),
    )
    route.add_argument("--max-cost-usd", type=float, help="per-run cost ceiling")
    route.add_argument(
        "--max-model-calls",
        type=int,
        default=3,
        help="per-run model-call ceiling",
    )
    route.add_argument("--json", action="store_true", help="emit machine-readable JSON")

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

    evaluation = subparsers.add_parser(
        "evaluation", help="train, summarize, and gate benchmark results"
    )
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
        if args.command == "route":
            return _route_command(args)

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
                expected_models=expected,
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
        if (
            not isinstance(key, str)
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
        ):
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


def _route_command(args) -> int:
    prompt = args.prompt
    if prompt is None or prompt == "-":
        prompt = sys.stdin.read()
    prompt = prompt.strip()
    if not prompt:
        print("route: empty prompt", file=sys.stderr)
        return 2

    requirements = parse_requirements(args.requirements)
    risk = parse_risk(args.risk)

    estimated_input, estimated_output = estimate_tokens(prompt)
    if args.input_tokens is not None:
        estimated_input = args.input_tokens
    if args.output_tokens is not None:
        estimated_output = args.output_tokens

    task = Task(
        kind=args.kind,
        payload={"prompt": prompt},
        requirements=requirements,
        risk=risk,
        metadata={
            "estimated_input_tokens": estimated_input,
            "estimated_output_tokens": estimated_output,
        },
    )

    catalog = load_catalog(args.catalog)
    adaptive_policy = AdaptivePolicy(PolicyMode(args.mode))
    decision = plan_delegation(
        task,
        registry=catalog.registry(),
        adaptive_policy=adaptive_policy,
        threshold_tokens=args.threshold_tokens,
        max_cost_usd=args.max_cost_usd,
    )

    payload = decision.as_dict()
    payload["executed"] = False

    # A plan never invokes a provider, and neither does an --execute run that the
    # threshold or eligibility check already rejected.
    if args.execute and decision.delegate:
        payload.update(_execute_routed(task, catalog, adaptive_policy, args))
        payload["executed"] = True

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        _print_route(payload)
    return 0


def _execute_routed(task, catalog, adaptive_policy, args) -> dict:
    from .live_evaluation import provider_invoker_from_catalog

    executor = RoutedModelExecutor(
        registry=catalog.registry(),
        invoke=provider_invoker_from_catalog(catalog),
        adaptive_policy=adaptive_policy,
    )
    runtime = RouterRuntime()
    for execution_class in (ExecutionClass.LIGHT_REASONING, ExecutionClass.DEEP_REASONING):
        runtime.register_executor(execution_class, executor)

    budget = Budget(max_cost_usd=args.max_cost_usd, max_model_calls=args.max_model_calls)
    result = runtime.execute(task, budget=budget)
    return {
        "output": result.output,
        "actual_cost_usd": budget.cost_usd,
        "model_calls": budget.model_calls,
        "executed_model": result.metadata.get("model"),
        "executed_provider": result.metadata.get("provider"),
    }


def _print_route(payload: dict) -> None:
    verdict = "DELEGATE" if payload["delegate"] else "DO NOT DELEGATE"
    print(f"{verdict}: {payload['reason']}")
    print(f"  execution class  : {payload['execution_class']} ({payload['routing_reason']})")
    print(f"  reliability floor: {payload['reliability_floor']:.2f}")
    print(
        f"  estimated tokens : {payload['estimated_input_tokens']} in / "
        f"{payload['estimated_output_tokens']} out"
    )
    if payload.get("model"):
        print(
            f"  selected         : {payload['provider']}/{payload['model']} "
            f"(est. ${payload['estimated_cost_usd']:.6f})"
        )
    for alt in payload.get("alternatives", ()):
        print(
            f"  alternative      : {alt['provider']}/{alt['model']} "
            f"(est. ${alt['estimated_cost_usd']:.6f})"
        )
    if payload.get("executed"):
        print(
            f"  ACTUAL           : {payload['executed_provider']}/{payload['executed_model']} "
            f"${payload['actual_cost_usd']:.6f} in {payload['model_calls']} call(s)"
        )
        print()
        print(payload["output"])

if __name__ == "__main__":
    raise SystemExit(main())
