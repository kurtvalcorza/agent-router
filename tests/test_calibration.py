import json
import math

import pytest

from agent_router.calibration import (
    CALIBRATION_METHOD_VERSION,
    _beta_quantile,
    _regularized_incomplete_beta,
    calibrate_reliability,
)
from agent_router.calibration_io import (
    CalibrationIOError,
    StaleProposalError,
    apply_proposals,
    load_proposals,
    write_proposals,
)
from agent_router.catalog import parse_catalog
from agent_router.cli import main
from agent_router.evaluation import EvaluationCase, EvaluationRun

# --- Beta maths ---------------------------------------------------------------
#
# Checked against closed forms rather than another library, so the tests stand alone:
#   Beta(1,1) CDF(x) = x        Beta(2,1) CDF(x) = x^2      Beta(1,2) CDF(x) = 1-(1-x)^2
#   Beta(2,2) CDF(x) = 3x^2 - 2x^3


@pytest.mark.parametrize(
    ("a", "b", "x", "expected"),
    [
        (1.0, 1.0, 0.3, 0.3),
        (2.0, 1.0, 0.5, 0.25),
        (1.0, 2.0, 0.5, 0.75),
        (2.0, 2.0, 0.25, 3 * 0.25**2 - 2 * 0.25**3),
        (5.0, 3.0, 0.0, 0.0),
        (5.0, 3.0, 1.0, 1.0),
    ],
)
def test_incomplete_beta_matches_closed_forms(a, b, x, expected) -> None:
    assert _regularized_incomplete_beta(a, b, x) == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize(
    ("a", "b", "p", "expected"),
    [
        (1.0, 1.0, 0.05, 0.05),
        (2.0, 1.0, 0.25, math.sqrt(0.25)),
        (1.0, 2.0, 0.75, 1 - math.sqrt(0.25)),
        (2.0, 2.0, 0.5, 0.5),
    ],
)
def test_beta_quantile_matches_closed_forms(a, b, p, expected) -> None:
    assert _beta_quantile(a, b, p) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(("a", "b"), [(48, 10), (3, 1), (1, 50), (200, 300)])
@pytest.mark.parametrize("p", [0.05, 0.5, 0.95])
def test_quantile_round_trips_through_the_cdf(a, b, p) -> None:
    assert _regularized_incomplete_beta(a, b, _beta_quantile(a, b, p)) == pytest.approx(p, abs=1e-9)


def test_beta_quantile_rejects_out_of_range_probability() -> None:
    with pytest.raises(ValueError, match="probability"):
        _beta_quantile(2.0, 2.0, 1.5)


# --- calibration --------------------------------------------------------------


def _evidence(model_plan):
    """Build cases/runs from {model: [(task_kind, trials, successes)]}."""
    cases, runs, seen = [], [], set()
    for model, groups in model_plan.items():
        for kind, trials, wins in groups:
            for index in range(trials):
                case_id = f"{kind}-{index}"
                if case_id not in seen:
                    cases.append(EvaluationCase(id=case_id, task_kind=kind, minimum_quality=1.0))
                    seen.add(case_id)
                runs.append(
                    EvaluationRun(
                        case_id=case_id,
                        strategy="router",
                        model=model,
                        quality=1.0 if index < wins else 0.0,
                        cost_usd=0.0,
                        latency_seconds=0.1,
                    )
                )
    return cases, runs


def test_proposal_is_conservative_relative_to_the_posterior_mean() -> None:
    cases, runs = _evidence({"m": [("classify", 40, 36), ("summarize", 16, 11)]})

    (proposal,) = calibrate_reliability(cases, runs, evidence_ref="run-1")

    assert proposal.successes == 47
    assert proposal.trials == 56
    # The whole point: never propose the raw posterior mean.
    assert proposal.proposed_reliability < proposal.posterior_mean
    assert proposal.proposed_reliability == min(
        proposal.pooled_lower_bound, proposal.task_class_balanced_mean
    )
    assert proposal.credible_interval[0] <= proposal.posterior_mean <= proposal.credible_interval[1]


def test_class_balancing_resists_a_corpus_skewed_toward_easy_cases() -> None:
    """A model that aces 200 easy cases and fails a small hard class must not be
    promoted on the strength of the easy ones."""
    cases, runs = _evidence({"m": [("easy", 200, 200), ("hard", 20, 4)]})

    (proposal,) = calibrate_reliability(cases, runs, evidence_ref="run-1")

    pooled_rate = proposal.successes / proposal.trials
    assert proposal.task_class_balanced_mean < pooled_rate
    assert proposal.proposed_reliability <= proposal.task_class_balanced_mean
    assert any("skewed" in warning for warning in proposal.warnings)


def test_thin_evidence_is_marked_insufficient() -> None:
    cases, runs = _evidence({"m": [("classify", 5, 5)]})

    (proposal,) = calibrate_reliability(cases, runs, evidence_ref="run-1")

    assert proposal.status == "INSUFFICIENT_EVIDENCE"
    assert any("trial" in warning for warning in proposal.warnings)


def test_single_task_class_is_warned_about() -> None:
    cases, runs = _evidence({"m": [("classify", 40, 30)]})

    (proposal,) = calibrate_reliability(cases, runs, evidence_ref="run-1")

    assert any("single task class" in warning for warning in proposal.warnings)


def test_threshold_crossings_are_reported() -> None:
    """The audit event that matters: a measured score crossing a policy floor."""
    cases, runs = _evidence({"m": [("classify", 60, 40), ("summarize", 60, 40)]})

    (proposal,) = calibrate_reliability(
        cases, runs, current_reliability={"m": 0.90}, evidence_ref="run-1"
    )

    crossed = {crossing.mode for crossing in proposal.threshold_crossings}
    assert {"balanced", "quality"} <= crossed
    assert all(not c.eligible_after for c in proposal.threshold_crossings)


def test_raising_reliability_is_flagged_for_extra_scrutiny() -> None:
    cases, runs = _evidence({"m": [("classify", 60, 59), ("summarize", 60, 59)]})

    (proposal,) = calibrate_reliability(
        cases, runs, current_reliability={"m": 0.50}, evidence_ref="run-1"
    )

    assert proposal.proposed_reliability > 0.50
    assert any("RAISE" in warning for warning in proposal.warnings)


def test_evidence_ref_is_required() -> None:
    cases, runs = _evidence({"m": [("classify", 30, 25)]})

    with pytest.raises(ValueError, match="evidence_ref"):
        calibrate_reliability(cases, runs, evidence_ref="")


def test_unknown_case_is_rejected() -> None:
    cases = [EvaluationCase(id="a", task_kind="qa")]
    runs = [
        EvaluationRun(
            case_id="ghost", strategy="router", model="m", quality=1.0,
            cost_usd=0.0, latency_seconds=0.0,
        )
    ]

    with pytest.raises(ValueError, match="unknown case"):
        calibrate_reliability(cases, runs, evidence_ref="run-1")


def test_explicit_success_flag_wins_over_quality_threshold() -> None:
    cases = [EvaluationCase(id="a", task_kind="qa", minimum_quality=1.0)]
    runs = [
        EvaluationRun(
            case_id="a", strategy="router", model="m", quality=1.0,
            cost_usd=0.0, latency_seconds=0.0, success=False,
        )
    ]

    (proposal,) = calibrate_reliability(cases, runs, evidence_ref="run-1")

    assert (proposal.successes, proposal.trials) == (0, 1)


# --- applying -----------------------------------------------------------------

CATALOG = {
    "version": "1",
    "pricing_as_of": "2026-08-23",
    "models": [
        {
            "name": "m",
            "provider": "local",
            "execution_classes": ["light_reasoning"],
            "capabilities": ["semantic_reasoning"],
            "reliability": 0.80,
            "context_window": 8192,
            "pricing": {"input_per_million": 0.0, "output_per_million": 0.0},
        }
    ],
}


def _proposal_dict(**overrides):
    base = {
        "model": "m",
        "current_reliability": 0.80,
        "proposed_reliability": 0.741,
        "status": "REVIEW_REQUIRED",
        "evidence_ref": "run-1",
        "successes": 47,
        "trials": 56,
        "credible_interval": [0.74, 0.90],
        "method": "beta-posterior-conservative",
        "method_version": CALIBRATION_METHOD_VERSION,
    }
    base.update(overrides)
    return base


def test_nothing_is_applied_without_explicit_acceptance() -> None:
    catalog = parse_catalog(CATALOG)

    result = apply_proposals(catalog, [_proposal_dict()])

    assert result.applied == ()
    assert result.skipped == (("m", "not accepted"),)
    assert result.catalog.registry().get("m").reliability == 0.80


def test_accepted_proposal_updates_a_new_catalog_and_records_provenance() -> None:
    catalog = parse_catalog(CATALOG)

    result = apply_proposals(catalog, [_proposal_dict()], accept=["m"])

    assert result.applied == (("m", 0.80, 0.741),)
    # The input catalog object is untouched.
    assert catalog.registry().get("m").reliability == 0.80

    updated = result.catalog.registry().get("m")
    assert updated.reliability == 0.741
    evidence = updated.metadata["reliability_evidence"]
    assert evidence["evidence_ref"] == "run-1"
    assert evidence["previous_reliability"] == 0.80
    assert evidence["method_version"] == CALIBRATION_METHOD_VERSION
    assert evidence["review_state"] == "applied-by-explicit-action"


def test_insufficient_evidence_is_skipped_unless_allowed() -> None:
    catalog = parse_catalog(CATALOG)
    proposal = _proposal_dict(status="INSUFFICIENT_EVIDENCE")

    skipped = apply_proposals(catalog, [proposal], accept=["m"])
    assert skipped.applied == ()
    assert "INSUFFICIENT_EVIDENCE" in skipped.skipped[0][1]

    forced = apply_proposals(
        catalog, [proposal], accept=["m"], allow_insufficient_evidence=True
    )
    assert forced.applied == (("m", 0.80, 0.741),)


def test_proposal_for_an_unknown_model_is_skipped() -> None:
    result = apply_proposals(
        parse_catalog(CATALOG), [_proposal_dict(model="ghost")], accept_all=True
    )

    assert result.applied == ()
    assert result.skipped == (("ghost", "not present in the catalog"),)


def test_proposal_file_round_trips(tmp_path) -> None:
    cases, runs = _evidence({"m": [("classify", 40, 36), ("summarize", 16, 11)]})
    proposals = calibrate_reliability(cases, runs, evidence_ref="run-1")
    path = tmp_path / "proposals.json"

    write_proposals(path, proposals)
    loaded = load_proposals(path)

    assert loaded[0]["model"] == "m"
    assert loaded[0]["status"] == "REVIEW_REQUIRED"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"model": "m"}, "missing"),
        (
            [{"model": "m", "proposed_reliability": 2.0, "status": "x", "evidence_ref": "r"}],
            "between",
        ),
        (
            [{"model": "m", "proposed_reliability": "x", "status": "s", "evidence_ref": "r"}],
            "number",
        ),
        ({}, "array"),
    ],
)
def test_malformed_proposal_files_are_rejected(tmp_path, payload, message) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload if isinstance(payload, list) else [payload]
                               if payload else payload), encoding="utf-8")

    with pytest.raises(CalibrationIOError, match=message):
        load_proposals(path)


# --- CLI ----------------------------------------------------------------------


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _cli_fixtures(tmp_path):
    cases, runs = _evidence({"m": [("classify", 40, 36), ("summarize", 16, 11)]})
    cases_file = _write(
        tmp_path, "cases.json",
        [{"id": c.id, "task_kind": c.task_kind, "minimum_quality": c.minimum_quality}
         for c in cases],
    )
    runs_file = _write(
        tmp_path, "runs.json",
        [{"case_id": r.case_id, "strategy": r.strategy, "model": r.model, "quality": r.quality,
          "cost_usd": r.cost_usd, "latency_seconds": r.latency_seconds} for r in runs],
    )
    catalog_file = _write(tmp_path, "catalog.json", CATALOG)
    return cases_file, runs_file, catalog_file


def test_calibrate_command_emits_proposals_without_touching_the_catalog(tmp_path, capsys) -> None:
    cases_file, runs_file, catalog_file = _cli_fixtures(tmp_path)
    before = catalog_file.read_text(encoding="utf-8")

    code = main([
        "evaluation", "calibrate",
        "--cases", str(cases_file), "--runs", str(runs_file),
        "--catalog", str(catalog_file),
        "--evidence-ref", "run-1", "--json",
    ])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["model"] == "m"
    assert payload[0]["status"] == "REVIEW_REQUIRED"
    # Measurement must never mutate policy.
    assert catalog_file.read_text(encoding="utf-8") == before


def test_apply_calibration_refuses_without_accept(tmp_path, capsys) -> None:
    cases_file, runs_file, catalog_file = _cli_fixtures(tmp_path)
    proposals = tmp_path / "proposals.json"
    main([
        "evaluation", "calibrate", "--cases", str(cases_file), "--runs", str(runs_file),
        "--catalog", str(catalog_file), "--evidence-ref", "run-1", "--output", str(proposals),
    ])
    capsys.readouterr()

    code = main([
        "catalog", "apply-calibration", str(catalog_file), str(proposals),
        "--output", str(tmp_path / "candidate.json"),
    ])

    assert code == 1
    assert "nothing applied" in capsys.readouterr().out
    assert not (tmp_path / "candidate.json").exists()


def test_apply_calibration_writes_a_candidate_and_leaves_the_source_alone(
    tmp_path, capsys
) -> None:
    cases_file, runs_file, catalog_file = _cli_fixtures(tmp_path)
    proposals = tmp_path / "proposals.json"
    main([
        "evaluation", "calibrate", "--cases", str(cases_file), "--runs", str(runs_file),
        "--catalog", str(catalog_file), "--evidence-ref", "run-1", "--output", str(proposals),
    ])
    capsys.readouterr()
    before = catalog_file.read_text(encoding="utf-8")
    candidate = tmp_path / "candidate.json"

    code = main([
        "catalog", "apply-calibration", str(catalog_file), str(proposals),
        "--output", str(candidate), "--accept", "all",
    ])

    assert code == 0
    assert catalog_file.read_text(encoding="utf-8") == before
    written = json.loads(candidate.read_text(encoding="utf-8"))
    model = next(item for item in written["models"] if item["name"] == "m")
    assert model["reliability"] < 0.80
    assert model["metadata"]["reliability_evidence"]["evidence_ref"] == "run-1"


# --- optimistic concurrency ---------------------------------------------------
#
# A proposal records the reliability it was reviewed against. Applying it to a catalog
# that has since moved would overwrite a newer value with a decision made about an older
# one -- a lost update. These fail against the pre-guard implementation.


def test_stale_proposal_is_refused_rather_than_overwriting_a_newer_value() -> None:
    """The catalog moved 0.80 -> 0.92 after the proposal was reviewed."""
    moved = json.loads(json.dumps(CATALOG))
    moved["models"][0]["reliability"] = 0.92
    catalog = parse_catalog(moved)

    with pytest.raises(StaleProposalError) as excinfo:
        apply_proposals(catalog, [_proposal_dict()], accept=["m"])

    assert excinfo.value.mismatches == (("m", 0.80, 0.92),)
    assert "0.92" in str(excinfo.value)
    assert catalog.registry().get("m").reliability == 0.92


def test_a_proposal_without_a_recorded_baseline_cannot_be_verified() -> None:
    """calibrate without --catalog records no baseline; applying it blind is refused."""
    catalog = parse_catalog(CATALOG)

    with pytest.raises(StaleProposalError) as excinfo:
        apply_proposals(catalog, [_proposal_dict(current_reliability=None)], accept=["m"])

    assert excinfo.value.mismatches == (("m", None, 0.80),)
    assert "no recorded baseline" in str(excinfo.value)


def test_one_stale_proposal_aborts_the_whole_set() -> None:
    """The set was reviewed together against one catalog state, so a partial apply would
    build a candidate from a mix of fresh and stale decisions."""
    two_models = json.loads(json.dumps(CATALOG))
    two_models["models"].append(
        {
            "name": "n",
            "provider": "local",
            "execution_classes": ["light_reasoning"],
            "capabilities": ["semantic_reasoning"],
            "reliability": 0.55,
            "context_window": 8192,
            "pricing": {"input_per_million": 0.0, "output_per_million": 0.0},
        }
    )
    catalog = parse_catalog(two_models)
    proposals = [
        _proposal_dict(),  # fresh: baseline 0.80 matches
        _proposal_dict(model="n", current_reliability=0.40),  # stale: catalog holds 0.55
    ]

    with pytest.raises(StaleProposalError) as excinfo:
        apply_proposals(catalog, proposals, accept_all=True)

    assert [name for name, _, _ in excinfo.value.mismatches] == ["n"]
    # Nothing applied, including the fresh one.
    assert catalog.registry().get("m").reliability == 0.80


def test_staleness_is_not_reported_for_proposals_that_were_not_accepted() -> None:
    """An unaccepted proposal is irrelevant, stale or not."""
    moved = json.loads(json.dumps(CATALOG))
    moved["models"][0]["reliability"] = 0.92

    result = apply_proposals(parse_catalog(moved), [_proposal_dict()])

    assert result.applied == ()
    assert result.skipped == (("m", "not accepted"),)


def test_staleness_tolerates_float_representation_noise() -> None:
    """A value that round-tripped through JSON must still match."""
    catalog = parse_catalog(CATALOG)
    baseline = json.loads(json.dumps(0.80))

    result = apply_proposals(
        catalog, [_proposal_dict(current_reliability=baseline)], accept=["m"]
    )

    assert result.applied == (("m", 0.80, 0.741),)


def test_apply_calibration_command_reports_staleness_and_writes_nothing(
    tmp_path, capsys
) -> None:
    cases_file, runs_file, catalog_file = _cli_fixtures(tmp_path)
    proposals = tmp_path / "proposals.json"
    main([
        "evaluation", "calibrate", "--cases", str(cases_file), "--runs", str(runs_file),
        "--catalog", str(catalog_file), "--evidence-ref", "run-1",
        "--output", str(proposals),
    ])
    capsys.readouterr()

    # Someone else raises reliability between calibration and application.
    moved = json.loads(catalog_file.read_text(encoding="utf-8"))
    moved["models"][0]["reliability"] = 0.92
    catalog_file.write_text(json.dumps(moved), encoding="utf-8")
    candidate = tmp_path / "candidate.json"

    code = main([
        "catalog", "apply-calibration", str(catalog_file), str(proposals),
        "--output", str(candidate), "--accept", "all",
    ])

    assert code == 1
    assert not candidate.exists()
    err = capsys.readouterr().err
    assert "STALE" in err
    assert "0.92" in err
