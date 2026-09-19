from __future__ import annotations

import copy
import json
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from lotteries_core import forecast_ranges as ranges
from lotteries_core.pair_raster import _history_matrices
from lotteries_core.protocol import GameSpec
from lotteries_core.temporal_candidate_set import _walk_forward_backtest


def history():
    return pd.DataFrame([
        {"draw_date": f"2026-01-{i + 1:02}", "ball_1": a, "ball_2": b}
        for i, (a, b) in enumerate(((1, 2), (1, 3), (2, 4), (3, 4)))
    ])


def fixture_pending():
    frame, spec = history().iloc[:3], GameSpec("tiny", 4, 2)
    measures = ranges.measure_ranges(
        [{"range_id": "test", "intervals": [[1, 1]]}], {0: 3, 1: 3}, 6, 2.0,
    )
    return ranges.pending_record(frame, spec, measures, history_cutoff="2026-01-03",
                                 target_draw_date="2026-01-04", snapshot_sha256="a" * 64)


def test_exact_membership_and_union_count_match_independent_ticket_enumeration():
    frame = history()
    observed = [set((row.ball_1, row.ball_2)) for row in frame.itertuples()]
    scores = {ticket: sum(set(ticket).issubset(draw) for draw in observed)
              for ticket in combinations(range(1, 5), 2)}
    bands = [{"range_id": "overlap", "intervals": [[1, 1], [0.5, 1.5]]}]
    measured = ranges.measure_ranges(bands, Counter(scores.values()), 6, 2, actual_score=1)[0]
    assert measured["candidate_count"] == sum(value == 1 for value in scores.values()) == 4
    assert measured["full_winner_in_range"] is True
    assert measured["fair_containment_probability"] == pytest.approx(4 / 6)
    assert measured["full_purchase_stake_eur"] == 8
    assert measured["cash_roi"] is None
    assert not ranges.contains([[1, 1]], 0)


def test_normal_bands_are_declared_nested_and_missing_variance_is_not_invented():
    assert ranges.forecast_ranges(10, 20, {}) == []
    bands = ranges.forecast_ranges(10, 20, {"variance_next": 4})
    assert bands[0]["intervals"][0] == pytest.approx([7.436896869, 12.563103131])
    assert bands[2]["intervals"][0][0] < bands[0]["intervals"][0][0]
    with pytest.raises(ValueError):
        ranges.forecast_ranges(10, 20, {"variance_next": float("nan")})
    with pytest.raises(ValueError, match="complete legal"):
        ranges.measure_ranges(bands, {1: 2}, 3, 2)


def test_settlement_uses_frozen_counts_and_rejects_modified_commitment():
    pending, spec = fixture_pending(), GameSpec("tiny", 4, 2)
    settled = ranges.settle(pending, history(), spec)
    assert settled["actual_g_score"] == 0  # (3,4) never occurred before the target
    assert settled["ranges"][0]["full_winner_in_range"] is False
    modified_past = history()
    modified_past.loc[:2, ["ball_1", "ball_2"]] = [3, 4]
    assert ranges.settle(pending, modified_past, spec) == settled
    corrupted = copy.deepcopy(pending)
    corrupted["forecast"]["ranges"][0]["intervals"] = [[0, 10]]
    with pytest.raises(ValueError, match="commitment"):
        ranges.settle(corrupted, history(), spec)
    with pytest.raises(ValueError, match="unique official"):
        ranges.settle(pending, pd.concat([history(), history().tail(1)]), spec)


def test_no_future_target_enters_forecast_or_range_construction(monkeypatch):
    seen = []

    def forecast(training, _spec, _epochs):
        seen.append(training["draw_date"].tolist())
        return 0.0, 1.0, {"variance_next": 1.0}, {}

    monkeypatch.setattr("lotteries_core.temporal_candidate_set._forecasts", forecast)
    rows = []
    _walk_forward_backtest(history(), GameSpec("tiny", 4, 2), 2, holdout=2,
                           transformer_epochs=1, batch_size=2, seed=1,
                           range_rows=rows, ticket_price=2)
    assert seen == [["2026-01-01", "2026-01-02"],
                    ["2026-01-01", "2026-01-02", "2026-01-03"]]
    assert len(rows) == 8
    assert all(row["history_cutoff"] < row["draw_date"] for row in rows)
    assert all(row["cash_roi"] is None for row in rows)


@pytest.mark.parametrize("published,eligible", [("2026-01-03T23:59:59Z", True),
                                                 ("2026-01-04T00:00:00Z", False),
                                                 ("2026-01-05T00:00:00Z", False)])
def test_only_exact_dated_public_forecasts_count_as_prospective(published, eligible):
    pending = fixture_pending()

    def read(url):
        if "/tree/" in url:
            return json.dumps([{"path": f"data/profiles/euromillions/{ranges.PENDING_FILE}",
                                "lastCommit": {"id": "b" * 40, "date": published}}]).encode()
        return json.dumps(pending).encode()

    receipt = ranges.publication_receipt(pending, "euromillions", read_url=read)
    assert receipt["verified_before_draw_date"] is eligible


def test_ledger_is_idempotent_preserves_first_forecast_and_never_counts_unverified(tmp_path):
    pending, spec = fixture_pending(), GameSpec("tiny", 4, 2)
    first = ranges.write_benchmark(tmp_path, pending, [], history(), spec)
    assert first["prospective"] == []
    later = copy.deepcopy(pending)
    later["forecast"]["generated_utc"] = "2026-01-04T23:00:00Z"
    later["commitment_sha256"] = ranges.digest(later["forecast"])
    ranges.write_benchmark(tmp_path, later, [], history(), spec)
    ledger = json.loads((tmp_path / ranges.LEDGER_FILE).read_text())
    assert len(ledger["entries"]) == 1
    assert json.loads((tmp_path / ranges.PENDING_FILE).read_text()) == pending

    def read(url):
        if "/tree/" in url:
            return json.dumps([{"path": f"data/profiles/euromillions/{ranges.PENDING_FILE}",
                                "lastCommit": {"id": "b" * 40,
                                               "date": "2026-01-03T23:00:00Z"}}]).encode()
        return json.dumps(pending).encode()

    verified = ranges.write_benchmark(tmp_path, later, [], history(), spec,
                                      profile_key="euromillions", read_url=read)
    assert verified["prospective"][0]["draws"] == 1
    assert verified["prospective"][0]["full_winner_hits"] == 0
    assert verified["prospective"][0]["fair_expected_hits"] == 0.5


def test_second_prize_is_exact_euromillions_5_plus_1_and_nl_5_plus_reserve():
    eu = GameSpec.euromillions()
    matrices = (np.zeros((51, 51), int), np.zeros((13, 13), int), np.zeros((51, 13), int))
    row = ranges.measure_ranges([{"range_id": "all", "intervals": [[0, 0]]}],
                                {0: eu.n_tickets()}, eu.n_tickets(), 2.5, 0)
    result = ranges.prize_measures(row, eu, ((1, 2, 3, 4, 5), (1, 2)), matrices)[0]
    assert result["second_prize_winning_tickets_in_range"] == 20
    assert result["fair_second_or_better_probability"] == 1.0
    nl = GameSpec("nl-lotto", 45, 6)
    matrices = (np.zeros((46, 46), int), np.zeros((1, 1), int), np.zeros((46, 1), int))
    row = ranges.measure_ranges([{"range_id": "all", "intervals": [[0, 0]]}],
                                {0: nl.n_tickets()}, nl.n_tickets(), 2, 0)
    result = ranges.prize_measures(row, nl, ((1, 2, 3, 4, 5, 6), ()), matrices, 7)[0]
    assert result["second_prize_winning_tickets_in_range"] == 6
    with pytest.raises(ValueError, match="reserve"):
        ranges.prize_measures(row, nl, ((1, 2, 3, 4, 5, 6), ()), matrices, 1)
    result = ranges.prize_measures(row, nl, ((1, 2, 3, 4, 5, 6), ()), matrices)[0]
    assert result["second_prize_winning_tickets_in_range"] is None


def test_replay_match_uses_same_frozen_scoring_definition():
    frame, spec = history().iloc[:3], GameSpec("tiny", 4, 2)
    pending = fixture_pending()
    matrices = _history_matrices(frame, spec)
    assert ranges.score_ticket(matrices, (3, 4), ()) == ranges.settle(pending, history(), spec)["actual_g_score"]


def test_second_prize_can_be_inside_while_jackpot_is_outside():
    eu = GameSpec.euromillions()
    matrices = (np.zeros((51, 51), int), np.zeros((13, 13), int), np.zeros((51, 13), int))
    matrices[1][1, 3] = matrices[1][3, 1] = 10
    count = eu.n_tickets() // 66  # precisely one star pair, across every main combination
    rows = ranges.measure_ranges([{"range_id": "star13", "intervals": [[10, 10]]}],
                                  {10: count, 0: eu.n_tickets() - count}, eu.n_tickets(), 2.5, 0)
    result = ranges.prize_measures(rows, eu, ((1, 2, 3, 4, 5), (1, 2)), matrices)[0]
    assert result["full_winner_in_range"] is False
    assert result["second_prize_winning_tickets_in_range"] == 1
    assert result["second_or_better_hit"] is True
    assert 0.27 < result["fair_second_or_better_probability"] < 0.28
