"""The graduation watcher decides when to interrupt a maintainer; these are its guardrails.

The expensive mistake is a false alarm: an issue claiming a provider beat random when one jackpot
carried a thin ledger. Most of what follows pins down the cases that must *not* fire.
"""

from __future__ import annotations

import json

import pytest

from scripts.check_graduation import (
    CONTROL_PROVIDER,
    MET,
    UNKNOWN,
    UNMET,
    Gate,
    Thresholds,
    build_notifications,
    evaluate_provider,
    exported_names,
    gate_no_critical_issues,
    gate_pypi_install,
    gate_release_history,
    gate_settled_observations,
    is_prerelease,
    notification_key,
    paired_draws,
    sign_test_p_value,
)

THRESHOLDS = Thresholds()
KEY = ("euromillions", "EUR", "frequency", "1.0.0", "config-1")


def _record(draw: str, payout: float, control_payout: float, stake: float = 10.0) -> dict:
    return {
        "schema_version": 1,
        "lottobench_version": "0.1.0a1",
        "provider_name": "frequency",
        "provider_version": "1.0.0",
        "provider_config_sha256": "config-1",
        "benchmark_id": "cohort-1",
        "game": "euromillions",
        "draw_key": draw,
        "currency": "EUR",
        "n_sets": 4,
        "stake": stake,
        "m_portfolio_prize": payout,
        "m_net_return": payout - stake,
        "realized_roi": (payout - stake) / stake,
        "c_portfolio_prize": control_payout,
        "c_net_return": control_payout - stake,
        "control_realized_roi": (control_payout - stake) / stake,
        "realized_roi_lift": (payout - control_payout) / stake,
        "outcome_source": "self_reported",
        "purchase_proof_hash_present": 0,
        "record_sha256": f"hash-{draw}",
    }


def _pairs(records: list[dict]) -> list:
    pairs, problems = paired_draws(records)
    assert not problems
    return pairs[KEY]


def test_sign_test_matches_the_exact_binomial_tail():
    assert sign_test_p_value(10, 10) == pytest.approx(1 / 1024)
    assert sign_test_p_value(0, 10) == pytest.approx(1.0)
    assert sign_test_p_value(5, 10) == pytest.approx(0.623046875)
    # No decisive draws is no information, never a pass.
    assert sign_test_p_value(0, 0) == pytest.approx(1.0)


def test_one_jackpot_in_a_losing_ledger_does_not_qualify():
    """The failure mode the gate exists for: positive cumulative ROI, no repeatable edge."""
    records = [_record(f"draw-{index}", 0.0, 2.0) for index in range(39)]
    records.append(_record("draw-jackpot", 5000.0, 2.0))

    scored = evaluate_provider(KEY, _pairs(records), THRESHOLDS)

    assert scored["net_lift"] > 0
    assert scored["net_lift_excluding_best_draw"] < 0
    assert not scored["qualifies"]
    assert "collapses without its single best draw" in " ".join(scored["blocking_reasons"])


def test_consistent_wins_over_the_control_qualify():
    records = [_record(f"draw-{index}", 12.0, 2.0) for index in range(35)]

    scored = evaluate_provider(KEY, _pairs(records), THRESHOLDS)

    assert scored["settled_draws"] == 35
    assert scored["draws_won"] == 35
    assert scored["sign_test_p_value"] < THRESHOLDS.alpha
    assert scored["roi_lift"] == pytest.approx(1.0)
    assert scored["qualifies"], scored["blocking_reasons"]


def test_a_short_ledger_never_qualifies_however_good_it_looks():
    records = [_record(f"draw-{index}", 500.0, 0.0) for index in range(5)]

    scored = evaluate_provider(KEY, _pairs(records), THRESHOLDS)

    assert not scored["qualifies"]
    assert any("settled matched draws" in reason for reason in scored["blocking_reasons"])


def test_ties_are_not_evidence():
    """Draws where nobody won anything are the common case and must not count as wins."""
    records = [_record(f"draw-{index}", 0.0, 0.0) for index in range(40)]

    scored = evaluate_provider(KEY, _pairs(records), THRESHOLDS)

    assert scored["settled_draws"] == 40
    assert scored["decisive_draws"] == 0
    assert not scored["qualifies"]
    assert any("decisive draws" in reason for reason in scored["blocking_reasons"])


def test_records_without_a_matched_control_are_excluded():
    matched = _record("draw-1", 12.0, 2.0)
    unmatched = _record("draw-2", 12.0, 2.0)
    for field in ("c_portfolio_prize", "c_net_return", "control_realized_roi", "realized_roi_lift"):
        unmatched[field] = None

    pairs, problems = paired_draws([matched, unmatched])

    assert not problems
    assert [pair.draw_key for pair in pairs[KEY]] == ["draw-1"]


def test_the_control_is_not_scored_against_itself():
    record = _record("draw-1", 12.0, 2.0)
    record["provider_name"] = CONTROL_PROVIDER

    pairs, problems = paired_draws([record])

    assert pairs == {}
    assert not problems


def test_a_corrupt_ledger_row_is_reported_rather_than_counted():
    broken = _record("draw-bad", 12.0, 2.0)
    broken["m_net_return"] = 999.0

    pairs, problems = paired_draws([_record("draw-1", 12.0, 2.0), broken])

    assert len(problems) == 1
    assert "draw-bad" in problems[0]
    assert [pair.draw_key for pair in pairs[KEY]] == ["draw-1"]
    assert gate_no_critical_issues(0, problems).status == UNMET


def test_unsupplied_inputs_report_unknown_rather_than_passing():
    assert gate_pypi_install(None, ("3.10",)).status == UNKNOWN
    assert gate_pypi_install({"results": {"3.10": True}}, ("3.10", "3.11")).status == UNKNOWN
    assert gate_no_critical_issues(None, []).status == UNKNOWN


def test_pypi_gate_reads_both_report_shapes():
    versions = ("3.10", "3.11")
    assert gate_pypi_install({"3.10": True, "3.11": True}, versions).status == MET
    assert gate_pypi_install({"results": {"3.10": True, "3.11": True}}, versions).status == MET
    assert gate_pypi_install({"results": {"3.10": True, "3.11": False}}, versions).status == UNMET


def test_release_history_counts_only_prereleases():
    assert is_prerelease("v0.1.0a1")
    assert is_prerelease("0.2.0rc3")
    assert not is_prerelease("v1.0.0")

    tags = [((0, 1, 0, 0, 1), "v0.1.0a1"), ((1, 0, 0, 3, 0), "v1.0.0")]
    assert gate_release_history(tags, THRESHOLDS).status == UNMET
    assert gate_release_history([], THRESHOLDS).status == UNKNOWN

    tags.append(((0, 2, 0, 1, 1), "v0.2.0b1"))
    assert gate_release_history(tags, THRESHOLDS).status == MET


def test_settled_observation_gate_counts_distinct_draws():
    records = [_record(f"draw-{index}", 0.0, 0.0) for index in range(30)]
    pairs, _ = paired_draws(records)

    assert gate_settled_observations(pairs, THRESHOLDS).status == MET
    assert gate_settled_observations({}, THRESHOLDS).status == UNMET


def test_exported_names_reads_all_without_importing():
    assert exported_names("__all__ = ['A', 'B']\nraise SystemExit(1)\n") == {"A", "B"}
    assert exported_names("x = 1\n") is None
    assert exported_names("def broken(\n") is None


def test_notification_keys_are_stable_and_filename_safe():
    key = notification_key("evidence-frequency-1.0.0-abc/def")
    assert key == "evidence-frequency-1.0.0-abc-def"
    assert notification_key("graduation-0.1.0a1") == "graduation-0.1.0a1"


def test_notifications_are_emitted_only_for_reached_milestones():
    gates = [Gate("pypi_install", MET, "fine")]
    report = {
        "graduation": {"ready": False},
        "provider_evidence": {"qualifying": []},
    }

    assert build_notifications(report, gates, THRESHOLDS, "0.1.0a1") == []

    scored = evaluate_provider(KEY, _pairs([_record(f"d-{i}", 12.0, 2.0) for i in range(35)]), THRESHOLDS)
    report = {
        "graduation": {"ready": True},
        "provider_evidence": {"qualifying": [scored]},
    }
    notifications = build_notifications(report, gates, THRESHOLDS, "0.1.0a1")

    assert [entry["key"] for entry in notifications] == [
        "graduation-0.1.0a1",
        "evidence-frequency-1.0.0-config-1",
    ]
    for entry in notifications:
        # The workflow deduplicates on this marker; without it every run reopens the issue.
        assert f"<!-- lottobench-watch: {entry['key']} -->" in entry["body"]
        assert json.dumps(THRESHOLDS.as_dict(), sort_keys=True) in entry["body"]
    assert "not betting advice" in notifications[1]["body"]
