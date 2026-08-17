from agent_router.evaluation import (
    EvaluationCase,
    EvaluationRun,
    compare_strategies,
    evaluate_gate,
    summarize_strategy,
)


CASES = [
    EvaluationCase(id="a", task_kind="simple", minimum_quality=0.8),
    EvaluationCase(id="b", task_kind="hard", minimum_quality=0.9),
]

RUNS = [
    EvaluationRun(
        case_id="a",
        strategy="router",
        model="small",
        quality=0.9,
        cost_usd=0.01,
        latency_seconds=1.0,
    ),
    EvaluationRun(
        case_id="b",
        strategy="router",
        model="strong",
        quality=0.92,
        cost_usd=0.04,
        latency_seconds=3.0,
        escalations=1,
    ),
    EvaluationRun(
        case_id="a",
        strategy="always-strong",
        model="strong",
        quality=0.95,
        cost_usd=0.04,
        latency_seconds=2.0,
    ),
    EvaluationRun(
        case_id="b",
        strategy="always-strong",
        model="strong",
        quality=0.96,
        cost_usd=0.04,
        latency_seconds=3.0,
    ),
]


def test_summary_computes_success_cost_latency_and_escalation():
    summary = summarize_strategy(CASES, RUNS, strategy="router")

    assert summary.success_rate == 1.0
    assert summary.total_cost_usd == 0.05
    assert summary.mean_cost_usd == 0.025
    assert summary.mean_latency_seconds == 2.0
    assert summary.escalation_rate == 0.5


def test_comparison_measures_cost_savings_and_quality_delta():
    router = summarize_strategy(CASES, RUNS, strategy="router")
    baseline = summarize_strategy(CASES, RUNS, strategy="always-strong")
    comparison = compare_strategies(router, baseline)

    assert comparison.cost_savings_fraction == 0.375
    assert round(comparison.quality_delta, 3) == -0.045
    assert comparison.success_rate_delta == 0.0


def test_gate_can_allow_small_quality_loss_for_material_cost_savings():
    router = summarize_strategy(CASES, RUNS, strategy="router")
    baseline = summarize_strategy(CASES, RUNS, strategy="always-strong")
    comparison = compare_strategies(router, baseline)

    passed, failures = evaluate_gate(
        comparison,
        minimum_cost_savings=0.30,
        maximum_quality_loss=0.05,
        maximum_success_rate_loss=0.0,
    )

    assert passed
    assert failures == ()
