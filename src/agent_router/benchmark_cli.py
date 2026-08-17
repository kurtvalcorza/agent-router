from __future__ import annotations

import argparse
import sys

from .adaptive import PolicyMode
from .benchmark_runtime import BenchmarkSpecError
from .catalog import CatalogError, load_catalog
from .evaluation_io import EvaluationIOError, load_cases, write_runs
from .live_evaluation import run_fixed_baseline, run_router_strategy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-router-benchmark")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--cheap", required=True, help="catalog model name or alias")
    parser.add_argument("--strong", required=True, help="catalog model name or alias")
    parser.add_argument("--mode", choices=[mode.value for mode in PolicyMode], default="balanced")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["router", "always-cheap", "always-strong"],
        default=["router", "always-cheap", "always-strong"],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cases = load_cases(args.cases)
        catalog = load_catalog(args.catalog)
        runs = []
        if "router" in args.strategies:
            runs.extend(
                run_router_strategy(
                    cases,
                    catalog=catalog,
                    mode=PolicyMode(args.mode),
                )
            )
        if "always-cheap" in args.strategies:
            runs.extend(
                run_fixed_baseline(
                    cases,
                    strategy="always-cheap",
                    catalog=catalog,
                    model=args.cheap,
                )
            )
        if "always-strong" in args.strategies:
            runs.extend(
                run_fixed_baseline(
                    cases,
                    strategy="always-strong",
                    catalog=catalog,
                    model=args.strong,
                )
            )
        write_runs(args.output, runs)
        print(
            f"wrote {len(runs)} runs across {len(args.strategies)} strategies to {args.output}"
        )
        return 0
    except (
        BenchmarkSpecError,
        CatalogError,
        EvaluationIOError,
        ImportError,
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
