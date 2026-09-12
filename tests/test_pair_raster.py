from __future__ import annotations

import hashlib
from itertools import combinations

import pandas as pd

from lotteries_core.pair_raster import build_pair_raster
from lotteries_core.protocol import GameSpec


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"draw_date": "2026-01-01", "ball_1": 1, "ball_2": 2, "ball_3": 3,
             "star_1": 1, "star_2": 2},
            {"draw_date": "2026-01-08", "ball_1": 1, "ball_2": 2, "ball_3": 4,
             "star_1": 1, "star_2": 2},
            {"draw_date": "2026-01-15", "ball_1": 1, "ball_2": 2, "ball_3": 5,
             "star_1": 1, "star_2": 3},
        ]
    )


def _brute_score(history: pd.DataFrame, main: tuple[int, ...], stars: tuple[int, ...]) -> int:
    score = 0
    for row in history.itertuples(index=False):
        observed_main = {row.ball_1, row.ball_2, row.ball_3}
        observed_stars = {row.star_1, row.star_2}
        score += sum(set(pair) <= observed_main for pair in combinations(main, 2))
        score += sum(set(pair) <= observed_stars for pair in combinations(stars, 2))
        score += sum(
            main_number in observed_main and star in observed_stars
            for main_number in main
            for star in stars
        )
    return score


def test_pair_raster_scores_the_exact_full_universe_and_matches_brute_force():
    history = _history()
    spec = GameSpec("tiny", main_n=5, main_k=3, star_n=3, star_k=2)
    result = build_pair_raster(
        history,
        spec,
        snapshot_sha256="a" * 64,
        history_cutoff="2026-01-15",
        target_draw_date="2026-01-22",
        top_n=30,
        batch_size=4,
    )

    expected = sorted(
        (
            _brute_score(history, main, stars),
            main,
            stars,
        )
        for main in combinations(range(1, 6), 3)
        for stars in combinations(range(1, 4), 2)
    )
    expected.sort(key=lambda row: (-row[0], row[1], row[2]))

    assert result.summary["main_combinations_scored"] == 10
    assert result.summary["ticket_combinations_scored"] == 30
    assert result.distribution["ticket_combinations"].sum() == 30
    assert result.top.iloc[0]["main_draw"] == " ".join(map(str, expected[0][1]))
    assert result.top.iloc[0]["auxiliary_draw"] == " ".join(map(str, expected[0][2]))
    assert result.top.iloc[0]["historical_pair_density_score"] == expected[0][0]
    assert result.top["commitment_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert result.top["score_status"].eq("PENDING").all()


def test_pair_raster_commitment_is_bound_to_the_forward_history_cutoff():
    spec = GameSpec("tiny-no-aux", main_n=5, main_k=3)
    history = _history().drop(columns=["star_1", "star_2"])
    first = build_pair_raster(
        history,
        spec,
        snapshot_sha256=hashlib.sha256(b"first").hexdigest(),
        history_cutoff="2026-01-15",
        target_draw_date="2026-01-22",
        top_n=3,
        batch_size=3,
    )
    second = build_pair_raster(
        history,
        spec,
        snapshot_sha256=hashlib.sha256(b"second").hexdigest(),
        history_cutoff="2026-01-22",
        target_draw_date="2026-01-29",
        top_n=3,
        batch_size=3,
    )

    assert first.top["commitment_sha256"].tolist() != second.top[
        "commitment_sha256"
    ].tolist()
    assert first.summary["history_cutoff"] == "2026-01-15"
    assert first.summary["interpretation"].startswith("Descriptive past co-occurrence")
