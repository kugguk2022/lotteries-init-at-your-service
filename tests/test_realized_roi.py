from __future__ import annotations

import json

import pandas as pd
import pytest

from lotteries_core.outcome_tracker import main as tracker_main
from lotteries_core.realized_roi import (
    LEGACY_VERSION,
    comparison,
    export_bundle,
    load_records,
    validate_bundle,
)


def _record(version: str = "1.0.0", payout: float = 4.0) -> dict:
    stake = 10.0
    return {
        "schema_version": 1,
        "lottobench_version": "0.1.0a1",
        "provider_name": "frequency",
        "provider_version": version,
        "provider_config_sha256": f"config-{version}",
        "benchmark_id": "cohort-1",
        "game": "euromillions",
        "draw_key": f"draw-{version}",
        "currency": "EUR",
        "n_sets": 4,
        "stake": stake,
        "m_portfolio_prize": payout,
        "m_net_return": payout - stake,
        "realized_roi": (payout - stake) / stake,
        "c_portfolio_prize": 1.0,
        "c_net_return": 1.0 - stake,
        "control_realized_roi": (1.0 - stake) / stake,
        "realized_roi_lift": (payout - 1.0) / stake,
        "outcome_source": "self_reported",
        "purchase_proof_hash_present": 0,
        "record_sha256": "prediction-hash",
    }


def test_comparison_separates_model_versions():
    rows = comparison([_record("1.0.0", 2.0), _record("2.0.0", 7.0)])
    assert {row["provider_version"] for row in rows} == {"1.0.0", "2.0.0"}
    assert rows[0]["provider_version"] == "2.0.0"
    assert rows[0]["realized_roi"] == pytest.approx(-0.3)
    assert rows[0]["realized_roi_lift"] == pytest.approx(0.6)


def test_bundle_is_deterministic_private_and_tamper_evident(tmp_path):
    first = export_bundle([_record()])
    second = export_bundle([_record()])
    assert first == second
    encoded = json.dumps(first)
    assert "tickets" not in encoded
    assert "machine" not in encoded
    path = tmp_path / "roi.json"
    path.write_text(json.dumps(first), encoding="utf-8")
    assert load_records(path)[0]["provider_version"] == "1.0.0"
    first["records"][0]["stake"] = 999
    with pytest.raises(ValueError, match="bundle integrity"):
        validate_bundle(first)


def test_legacy_csv_is_labeled_not_silently_mixed(tmp_path):
    path = tmp_path / "results.csv"
    pd.DataFrame(
        [{
            "method": "gingerm", "game": "euromillions", "draw_key": "old-1",
            "currency": "EUR", "stake": 10, "m_portfolio_prize": 0,
            "m_net_return": -10,
        }]
    ).to_csv(path, index=False)
    row = load_records(path)[0]
    assert row["provider_version"] == LEGACY_VERSION
    assert comparison([row])[0]["realized_roi"] == -1.0


def test_tracker_writes_versioned_roi_provenance(tmp_path):
    history = tmp_path / "history.csv"
    ledger = tmp_path / "ledger"
    payouts = tmp_path / "payouts.json"
    rows = []
    for index in range(12):
        rows.append(
            {
                "draw_date": f"2026-01-{index + 1:02d}",
                "ball_1": index % 8 + 1,
                "ball_2": (index + 2) % 8 + 1,
                "ball_3": (index + 5) % 8 + 1,
                "star_1": index % 4 + 1,
            }
        )
    pd.DataFrame(rows).to_csv(history, index=False)
    payouts.write_text(json.dumps({"tiers": {}}), encoding="utf-8")
    tracker_main(
        [
            "record", "--history", str(history), "--draw-key", "2026-03-01",
            "--ledger", str(ledger), "--methods", "frequency", "--n-sets", "4",
            "--ticket-price", "2.50", "--main-n", "8", "--main-k", "3",
            "--star-n", "4", "--star-k", "1",
        ]
    )
    tracker_main(
        [
            "settle", "--ledger", str(ledger), "--draw-key", "2026-03-01",
            "--actual-main", "1,3,6", "--actual-stars", "1",
            "--payout-table", str(payouts), "--outcome-source", "operator_verified",
        ]
    )
    row = load_records(ledger)[0]
    assert row["provider_name"] == "frequency"
    assert row["provider_version"] == "1.0.0"
    assert row["provider_config_sha256"]
    assert row["benchmark_id"]
    assert row["result_sha256"]
    assert row["realized_roi"] == -1.0
    assert row["outcome_source"] == "operator_verified"
    assert row["control_realized_roi"] == -1.0
