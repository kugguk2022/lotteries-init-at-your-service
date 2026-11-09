from __future__ import annotations

import importlib.util
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lotteries_core import poi_benchmark as pb
from lotteries_core.likely_set_generator import GameConfig, build_comatrices, score_ticket
from lotteries_core.protocol import GameSpec

SPEC = GameSpec("tiny", 8, 4, 3, 1)


def history(n=12):
    rng = np.random.default_rng(18)
    rows = []
    for i in range(n):
        numbers = sorted(rng.choice(np.arange(1, 9), 4, replace=False))
        rows.append({"draw_date": str((pd.Timestamp("2025-01-01")+pd.Timedelta(days=i)).date()),
                     **{f"ball_{j+1}": int(v) for j, v in enumerate(numbers)}, "star_1": i % 3+1})
    return pd.DataFrame(rows)


def test_exact_ranking_matches_exhaustive_original_pair_statistic_and_prefixes():
    train = history()
    mc, sc = [f"ball_{i+1}" for i in range(4)], ["star_1"]
    cfg = GameConfig("tiny", 8, 4, 3, 1)
    w = build_comatrices(train, cfg, mc, sc)
    target = np.mean([score_ticket(tuple(train.iloc[i][mc].astype(int)-1), (), w, "main")
                      for i in range(len(train)-4, len(train))])
    expected = []
    for index, main in enumerate(combinations(range(8), 4)):
        g = score_ticket(main, (), w, "main")
        for aux in range(3):
            ticket_id = index*3+aux
            key = int(pb.tie_keys(np.array([ticket_id]), 41)[0])
            expected.append((abs(g-target), -g, key, ticket_id))
    expected.sort()
    ranked = pb.rank_poi(train, SPEC, 50, seed=41, window=4)
    assert ranked.target == target
    assert list(ranked.ids) == [row[3] for row in expected[:50]]
    assert list(pb.rank_poi(train, SPEC, 12, seed=41, window=4).ids) == list(ranked.ids[:12])
    assert len(set(ranked.ids)) == 50
    mains, aux = ranked.decode()
    for m, s in zip(mains, aux):
        SPEC.validate_ticket((tuple(m), tuple(s)))


def test_uniform_reference_is_exact_without_replacement():
    universe, favourable, budget = 10, 3, 4
    sets = list(combinations(range(universe), budget))
    expected = sum(bool(set(s) & {0, 1, 2}) for s in sets)/len(sets)
    assert pb.null_any(universe, favourable, budget) == pytest.approx(expected)
    assert pb.null_any(139838160, 1, 1000000) == pytest.approx(1000000/139838160)
    assert pb.null_any(10, 3, 10) == 1
    assert pb.null_any(10, 0, 4) == 0
    assert pb.favourable_tickets(SPEC, 4) == 3
    assert pb.favourable_tickets(SPEC, None) == 1
    assert math.comb(8, 4)*3 == SPEC.n_tickets()


def build(tmp_path, frame=None, target="2025-01-13"):
    return pb.build_poi_benchmark(history() if frame is None else frame, SPEC,
                                 output_directory=tmp_path, target_draw_date=target,
                                 snapshot_sha256="a"*64, holdout=4, budgets=(4, 12, 50),
                                 window=4, created_utc="2025-01-12T12:00:00Z")


def test_future_mutations_cannot_change_earlier_forecasts(tmp_path):
    frame = history()
    build(tmp_path / "first", frame)
    changed = frame.copy()
    changed.loc[11, [f"ball_{i+1}" for i in range(4)]] = [1, 2, 3, 4]
    changed.loc[11, "star_1"] = 2
    build(tmp_path / "second", changed)
    a = pd.read_csv(tmp_path / "first" / pb.FILES[0])
    b = pd.read_csv(tmp_path / "second" / pb.FILES[0])
    assert list(a.candidate_ids_sha256) == list(b.candidate_ids_sha256)
    # Changing even the target result must not affect its own candidate set.
    assert list(a.history_sha256) == list(b.history_sha256)
    assert list(a[a.draw_date < "2025-01-12"].matching_tickets) == list(
        b[b.draw_date < "2025-01-12"].matching_tickets)


def test_manifest_full_tickets_cache_and_tamper_detection(tmp_path, monkeypatch):
    summary = build(tmp_path)
    manifest = json.loads((tmp_path / pb.FILES[4]).read_text())
    assert summary["realized_roi"] is None
    assert summary["evidence_kind"] == "historical_forward_replay_not_predraw_publication"
    assert manifest["publication_status"] == "AWAITING_PUBLIC_TIMESTAMP_VERIFICATION"
    tickets = pd.read_csv(tmp_path / pb.FILES[3])
    assert len(tickets) == 50
    assert not tickets.drop(columns=["rank", "g_score"]).duplicated().any()
    calls = []
    original = pb.rank_poi

    def count(*args, **kwargs):
        calls.append(len(args[0]))
        return original(*args, **kwargs)

    monkeypatch.setattr(pb, "rank_poi", count)
    assert build(tmp_path)["results"] == summary["results"]
    assert calls == [len(history())]  # only next set; replay cache is verified
    path = Path(__file__).resolve().parents[1]/"publishing/huggingface-space/poi_view.py"
    module_spec = importlib.util.spec_from_file_location("poi_view", path)
    view = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(view)
    loaded, _, replay = view.load_evidence(tmp_path)
    assert len(view.result_table(loaded)) == 3
    trend = view.trend_table(replay, 12, "full_winner")
    assert len(trend) == 8
    with (tmp_path / pb.FILES[3]).open("ab") as out:
        out.write(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        view.load_evidence(tmp_path)


def test_full_winner_requires_both_pools_on_one_ticket():
    # Correct mains on one ticket, correct star on a different main combination.
    ranked = pb.RankedPoi(np.array([0, 4]), np.zeros(math.comb(8, 4)), 0, SPEC, 1)
    outcome = pd.Series({"ball_1": 1, "ball_2": 2, "ball_3": 3, "ball_4": 4, "star_1": 2})
    rows = pb.evaluate_draw(ranked, outcome, (2,), "a"*64, "2025-01-13")
    assert next(r for r in rows if r["metric"] == "full_winner")["draw_hit"] == 0
    assert next(r for r in rows if r["metric"] == "all_main")["draw_hit"] == 1


def test_invalid_history_and_targets_fail_closed(tmp_path):
    bad = history().astype({"ball_1": float})
    bad.loc[0, "ball_1"] = 1.5
    with pytest.raises(ValueError, match="Noninteger"):
        build(tmp_path, bad)
    with pytest.raises(ValueError, match="Target must follow"):
        build(tmp_path, target="2025-01-12")
    with pytest.raises(ValueError, match="Duplicate"):
        build(tmp_path, pd.concat([history(), history().iloc[:1]]))
    with pytest.raises(ValueError, match="Invalid size"):
        pb.rank_poi(history(), SPEC, SPEC.n_tickets()+1, window=4)
