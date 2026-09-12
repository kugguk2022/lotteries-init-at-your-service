from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from lotteries_core.crowd_escape_forecast import (
    CROWD_ESCAPE_FILE,
    build_crowd_escape_forecast,
)
from lotteries_core.protocol import GameSpec
from lotteries_core.roi import JackpotModel


def _prospective() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "agent": "unpopularity",
                "rank_within_agent": rank,
                "main_draw": main,
                "auxiliary_draw": "1 2",
                "backtest_consistency_pct": 75.0,
                "backtest_mean_roi_alpha_pp": 0.25,
                "score_status": "PENDING",
                "commitment_sha256": hashlib.sha256(f"ticket-{rank}".encode()).hexdigest(),
            }
            for rank, main in ((2, "32 33 34 35 36"), (1, "40 42 44 46 48"))
        ]
        + [
            {
                "agent": "uniform_random",
                "rank_within_agent": 1,
                "main_draw": "1 2 3 4 5",
                "auxiliary_draw": "1 2",
                "backtest_consistency_pct": 50.0,
                "backtest_mean_roi_alpha_pp": 0.0,
                "score_status": "PENDING",
                "commitment_sha256": "0" * 64,
            }
        ]
    )


def test_writes_ranked_sealed_crowd_escape_draws(tmp_path: Path):
    result = build_crowd_escape_forecast(
        _prospective(),
        GameSpec.euromillions(),
        output_directory=tmp_path,
        history_cutoff="2026-09-11",
        target_draw_date="2026-09-15",
        snapshot_sha256="a" * 64,
        jackpot=JackpotModel(jackpot=100_000_000, ticket_price=2.5, n_other_tickets=10_000_000),
    )

    assert result.draws["rank"].tolist() == [1, 2]
    assert result.draws["target_draw_date"].eq("2026-09-15").all()
    assert result.draws["score_status"].eq("PENDING").all()
    assert result.draws["crowd_popularity_share_vs_average"].lt(1).all()
    assert result.draws["crowd_escape_lift_vs_average"].gt(1).all()
    assert result.summary["ticket_count"] == 2
    assert result.summary["full_purchase_stake"] == 5.0
    assert result.summary["mechanical_jackpot_coverage_pct"] == pytest.approx(
        2 / GameSpec.euromillions().n_tickets() * 100
    )
    artifact = tmp_path / CROWD_ESCAPE_FILE
    assert result.summary["artifacts"][CROWD_ESCAPE_FILE] == hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()


def test_rejects_settled_rows(tmp_path: Path):
    prospective = _prospective()
    prospective.loc[0, "score_status"] = "SCORED"
    with pytest.raises(ValueError, match="before settlement"):
        build_crowd_escape_forecast(
            prospective,
            GameSpec.euromillions(),
            output_directory=tmp_path,
            history_cutoff="2026-09-11",
            target_draw_date="2026-09-15",
            snapshot_sha256="a" * 64,
            jackpot=JackpotModel(),
        )
