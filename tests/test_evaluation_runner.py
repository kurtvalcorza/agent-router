from agent_router import (
    EvaluationCase,
    StrategyResult,
    compare_strategies,
    run_strategy,
    summarize_strategy,
)


def test_run_strategy_and_compare_baseline():
    cases = (
        EvaluationCase(id="a", task_kind="simple", minimum_quality=0.8),
        EvaluationCase(id="b", task_kind="hard", minimum_quality=0.8),
    )

    router = run_strategy(
        cases,
        strategy="router",
        execute=lambda case: StrategyResult(
            model="small" if case.id == "a" else "strong",
            quality=0.9,
            cost_usd=0.01 if case.id == "a" else 0.05,
            escalations=0 if case.id == "a" else 1,
        ),
    )
    baseline = run_strategy(
        cases,
        strategy="always-strong",
        execute=lambda case: StrategyResult(
            model="strong",
            quality=0.95,
            cost_usd=0.05,
        ),
    )

    router_summary = summarize_strategy(cases, router + baseline, strategy="router")
    strong_summary = summarize_strategy(cases, router + baseline, strategy="always-strong")
    comparison = compare_strategies(router_summary, strong_summary)

    assert router_summary.escalation_rate == 0.5
    assert comparison.cost_savings_fraction == 0.4
    assert comparison.quality_delta < 0
