from __future__ import annotations

import json

import pandas as pd
import pytest

from lotteries_core.poi_g_artifacts import (
    CANDIDATES_FILE,
    MANIFEST_FILE,
    PREDICTION_FILE,
    SELECTION_FILE,
    SETTLEMENT_FILE,
    build_poi_g_artifacts,
    settle_poi_g_artifacts,
    validate_poi_g_artifacts,
)
from lotteries_core.protocol import GameSpec
from lotteries_core.roi import JackpotModel


def _history(rows: int = 18) -> pd.DataFrame:
    values = []
    for index in range(rows):
        main = sorted({(index + offset * 3) % 8 + 1 for offset in range(3)})
        if len(main) < 3:
            main = [1, 2, 3]
        values.append(
            {
                "draw_date": f"2026-01-{index + 1:02d}",
                "ball_1": main[0],
                "ball_2": main[1],
                "ball_3": main[2],
                "star_1": index % 3 + 1,
            }
        )
    return pd.DataFrame(values)


def _bundle():
    return build_poi_g_artifacts(
        _history(),
        GameSpec("tiny", main_n=8, main_k=3, star_n=3, star_k=1),
        draw_key="2026-02-01",
        subset_size=30,
        budget=5,
        window=6,
        seed=2026,
        created_utc="2026-01-31T12:00:00+00:00",
        jackpot=JackpotModel(jackpot=100.0, ticket_price=2.0, n_other_tickets=100.0),
    )


def test_candidate_subset_and_fixed_budget_are_separate_verifiable_artifacts(tmp_path):
    bundle = _bundle()
    paths = bundle.write(tmp_path)
    manifest = validate_poi_g_artifacts(tmp_path)

    assert set(paths) == {"candidates", "selection", "prediction", "manifest"}
    assert len(pd.read_csv(tmp_path / CANDIDATES_FILE)) == 30
    assert len(pd.read_csv(tmp_path / SELECTION_FILE)) == 5
    assert json.loads((tmp_path / PREDICTION_FILE).read_text())["budget"] == 5
    assert manifest["candidate_rows"] == 30
    assert manifest["selection_rows"] == 5
    assert manifest["roi_boundary"].startswith("ROI applies only to the fixed-budget selection")


def test_candidate_tampering_is_rejected_before_settlement(tmp_path):
    _bundle().write(tmp_path)
    candidates = pd.read_csv(tmp_path / CANDIDATES_FILE)
    candidates.loc[0, "main_numbers"] = "[6,7,8]"
    candidates.to_csv(tmp_path / CANDIDATES_FILE, index=False)

    with pytest.raises(ValueError, match="artifact integrity"):
        validate_poi_g_artifacts(tmp_path)


def test_settlement_scores_only_selection_and_matched_equal_budget_control(tmp_path):
    bundle = _bundle()
    bundle.write(tmp_path)
    winning_main, winning_auxiliary = bundle.prediction.tickets[0]
    tier = f"{len(winning_main)}+{len(winning_auxiliary)}"

    result = settle_poi_g_artifacts(
        tmp_path,
        actual_main=winning_main,
        actual_auxiliary=winning_auxiliary,
        payout_table={tier: 100.0},
        ticket_price=2.0,
        currency="TST",
        settled_utc="2026-02-01T21:00:00+00:00",
    )

    assert result["budget"] == 5
    assert result["candidate_subset_size"] == 30
    assert sum(result["method"]["tier_counts"].values()) == 5
    assert sum(result["control"]["tier_counts"].values()) == 5
    assert result["method_prize"] >= 100.0
    assert result["realized_roi"] is not None
    assert (tmp_path / SETTLEMENT_FILE).exists()


def test_missing_payout_evidence_never_becomes_zero_roi(tmp_path):
    bundle = _bundle()
    bundle.write(tmp_path)
    actual_main, actual_auxiliary = bundle.prediction.tickets[0]

    result = settle_poi_g_artifacts(
        tmp_path,
        actual_main=actual_main,
        actual_auxiliary=actual_auxiliary,
        ticket_price=2.0,
        settled_utc="2026-02-01T21:00:00+00:00",
    )

    assert result["payout_table_present"] is False
    assert result["method_prize"] is None
    assert result["realized_roi"] is None
    assert result["control_realized_roi"] is None


def test_prospective_bundle_requires_a_seal_time():
    with pytest.raises(ValueError, match="require created_utc"):
        build_poi_g_artifacts(
            _history(),
            GameSpec("tiny", 8, 3, 3, 1),
            draw_key="2026-02-01",
            subset_size=20,
            budget=5,
        )


def test_manifest_is_written_last_and_present(tmp_path):
    _bundle().write(tmp_path)
    assert (tmp_path / MANIFEST_FILE).exists()
