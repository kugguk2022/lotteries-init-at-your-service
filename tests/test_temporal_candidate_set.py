from __future__ import annotations

import gzip
from itertools import combinations

import pandas as pd
import pytest

from lotteries_core.protocol import GameSpec
from lotteries_core.roi import JackpotModel
from lotteries_core.temporal_candidate_set import (
    ARCHIVE_FILE,
    build_temporal_candidate_set,
    ticket_global_index,
)


def _history(rows: int = 8) -> pd.DataFrame:
    main = list(combinations(range(1, 7), 3))
    stars = list(combinations(range(1, 4), 2))
    return pd.DataFrame(
        [
            {
                "draw_date": f"2026-01-{index + 1:02d}",
                "ball_1": main[index][0],
                "ball_2": main[index][1],
                "ball_3": main[index][2],
                "star_1": stars[index % len(stars)][0],
                "star_2": stars[index % len(stars)][1],
            }
            for index in range(rows)
        ]
    )


def test_ticket_global_index_matches_legal_lexicographic_enumeration():
    spec = GameSpec("tiny", 6, 3, 3, 2)
    tickets = [
        (main, star)
        for main in combinations(range(1, 7), 3)
        for star in combinations(range(1, 4), 2)
    ]
    assert [ticket_global_index(ticket, spec) for ticket in tickets] == list(
        range(spec.n_tickets())
    )


def test_exact_hybrid_archive_keeps_coverage_and_profitability_separate(
    tmp_path, monkeypatch
):
    spec = GameSpec("tiny", 6, 3, 3, 2)
    history = _history()
    monkeypatch.setattr(
        "lotteries_core.temporal_candidate_set._forecasts",
        lambda _history, _spec, _epochs: (
            11.25,
            14.75,
            {"probability_upper": 0.6},
            {"epochs": 1, "training_loss": 0.5},
        ),
    )
    result = build_temporal_candidate_set(
        history,
        spec,
        output_directory=tmp_path,
        history_cutoff="2026-01-08",
        target_draw_date="2026-01-09",
        snapshot_sha256="a" * 64,
        jackpot=JackpotModel(jackpot=100.0, ticket_price=2.0, n_other_tickets=100.0),
        candidate_size=17,
        holdout=2,
        transformer_epochs=1,
        batch_size=4,
    )

    summary = result.summary
    assert summary["candidate_size"] == 17
    assert summary["universe_size"] == 60
    assert summary["mechanical_jackpot_coverage_pct"] == pytest.approx(17 / 60 * 100)
    assert summary["full_purchase_stake"] == 34.0
    assert summary["jackpot_containment_alone_proves_profit"] is False
    assert summary["profitability_status"] == "UNASSESSED_NO_PURCHASE_OR_SETTLED_PAYOUT"
    assert summary["forward_only_gate"]["holdout_draws"] == 2
    assert len(result.preview) == 17
    assert result.preview["rank"].tolist() == list(range(1, 18))
    assert result.preview["score_status"].eq("PENDING").all()

    with gzip.open(tmp_path / ARCHIVE_FILE, "rt", encoding="utf-8") as handle:
        archived = pd.read_csv(handle)
    assert len(archived) == 17
    assert archived[["main_draw", "auxiliary_draw"]].drop_duplicates().shape[0] == 17
    assert summary["artifacts"][ARCHIVE_FILE]


def test_candidate_size_is_bounded_by_the_exact_universe(tmp_path, monkeypatch):
    spec = GameSpec("tiny-no-aux", 6, 3)
    history = _history().iloc[:5].drop(columns=["star_1", "star_2"])
    monkeypatch.setattr(
        "lotteries_core.temporal_candidate_set._forecasts",
        lambda _history, _spec, _epochs: (4.0, 5.0, {}, {}),
    )
    result = build_temporal_candidate_set(
        history,
        spec,
        output_directory=tmp_path,
        history_cutoff="2026-01-05",
        target_draw_date="2026-01-06",
        snapshot_sha256="b" * 64,
        jackpot=JackpotModel(jackpot=100.0, ticket_price=2.0, n_other_tickets=100.0),
        candidate_size=1_000_000,
        holdout=1,
        transformer_epochs=1,
        batch_size=2,
    )
    assert result.summary["candidate_size"] == spec.n_tickets() == 20
    assert result.summary["mechanical_jackpot_coverage_pct"] == 100.0
