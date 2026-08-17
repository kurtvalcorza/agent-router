from agent_router import (
    EmpiricalSuccessModel,
    EvaluationCase,
    EvaluationRun,
    load_empirical_model,
    write_empirical_model,
)
from agent_router.benchmark_runtime import case_to_task


def test_empirical_model_round_trip(tmp_path):
    cases = [
        EvaluationCase(
            id="c1",
            task_kind="qa",
            metadata={
                "requirements": ["semantic_reasoning"],
                "payload": {"prompt": "x"},
            },
        )
    ]
    runs = [
        EvaluationRun(
            case_id="c1",
            strategy="always-cheap",
            model="cheap",
            quality=1.0,
            cost_usd=0.01,
            latency_seconds=0.1,
            success=True,
        )
    ]
    trained = EmpiricalSuccessModel.fit(cases, runs)
    path = tmp_path / "empirical.json"

    write_empirical_model(path, trained)
    loaded = load_empirical_model(path)

    before = trained.estimate("cheap", case_to_task(cases[0]))
    after = loaded.estimate("cheap", case_to_task(cases[0]))
    assert after.probability == before.probability
    assert after.successes == before.successes
    assert after.trials == before.trials
