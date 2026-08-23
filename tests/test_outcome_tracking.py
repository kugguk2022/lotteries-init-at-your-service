from __future__ import annotations

import json

import pandas as pd
import pytest

from lotteries_core.likely_set_generator import CooccurrenceLevelSetProvider
from lotteries_core.outcome_tracker import METHOD_CHOICES, PENDING, RESULTS
from lotteries_core.outcome_tracker import main as tracker_main
from lotteries_core.protocol import GameSpec


def _history() -> pd.DataFrame:
    rows = []
    for i in range(18):
        mains = sorted({(i % 8) + 1, ((i + 2) % 8) + 1, ((i + 5) % 8) + 1})
        while len(mains) < 3:
            mains.append(max(mains) % 8 + 1)
            mains = sorted(set(mains))
        rows.append(
            {
                "draw_date": f"2026-01-{i + 1:02d}",
                "ball_1": mains[0],
                "ball_2": mains[1],
                "ball_3": mains[2],
                "star_1": (i % 4) + 1,
            }
        )
    return pd.DataFrame(rows)


def test_named_versions_are_available_to_three_year_tracker():
    assert {
        "gingerm", "spectral_contrarian", "parallax", "garch_markov_branch",
        "sequence_transformer",
    } <= set(METHOD_CHOICES)


def test_named_versions_keep_their_public_identity():
    from lotteries_core import registry

    for name in ("gingerm", "spectral_contrarian", "parallax"):
        assert registry.create(name).name == name


def test_provider_integrates_with_common_protocol():
    history = _history()
    spec = GameSpec("small", main_n=8, main_k=3, star_n=4, star_k=1)
    provider = CooccurrenceLevelSetProvider(window=5).fit(history, spec)
    result = provider.propose(spec, budget=6, rng=None)  # type: ignore[arg-type]
    assert len(result.tickets) == 6
    assert len(set(result.tickets)) == 6
    assert result.diagnostics["history_rows"] == len(history)
    for ticket in result.tickets:
        spec.validate_ticket(ticket)


def test_record_settle_tracks_tiers_and_official_payouts(tmp_path):
    history_path = tmp_path / "history.csv"
    ledger = tmp_path / "ledger"
    payout_path = tmp_path / "payout.json"
    _history().to_csv(history_path, index=False)
    payout_path.write_text(json.dumps({"tiers": {"3+1": 1000, "2+1": 10}}), encoding="utf-8")

    tracker_main(
        [
            "record",
            "--history",
            str(history_path),
            "--draw-key",
            "2026-02-01",
            "--ledger",
            str(ledger),
            "--n-sets",
            "6",
            "--main-n",
            "8",
            "--main-k",
            "3",
            "--star-n",
            "4",
            "--star-k",
            "1",
        ]
    )
    pending = [json.loads(line) for line in (ledger / PENDING).read_text().splitlines()]
    assert len(pending) == 1
    assert pending[0]["generated_from_rows"] == len(_history())
    assert pending[0]["history_sha256"]
    assert pending[0]["record_sha256"]
    controls = [(tuple(m), tuple(s)) for m, s in pending[0]["control_tickets"]]
    assert len(controls) == len(set(controls)) == 6

    tracker_main(
        [
            "settle",
            "--ledger",
            str(ledger),
            "--draw-key",
            "2026-02-01",
            "--actual-main",
            "1,3,6",
            "--actual-stars",
            "1",
            "--payout-table",
            str(payout_path),
        ]
    )
    results = pd.read_csv(ledger / RESULTS)
    assert len(results) == 1
    assert results.loc[0, "record_sha256"] == pending[0]["record_sha256"]
    assert json.loads(results.loc[0, "m_tier_counts"])
    assert results.loc[0, "m_portfolio_prize"] >= 0


def test_settlement_rejects_tampered_prediction(tmp_path):
    history_path = tmp_path / "history.csv"
    ledger = tmp_path / "ledger"
    _history().to_csv(history_path, index=False)
    tracker_main(
        [
            "record",
            "--history",
            str(history_path),
            "--draw-key",
            "2026-02-02",
            "--ledger",
            str(ledger),
            "--n-sets",
            "4",
            "--main-n",
            "8",
            "--main-k",
            "3",
            "--star-n",
            "4",
            "--star-k",
            "1",
        ]
    )
    path = ledger / PENDING
    rec = json.loads(path.read_text().strip())
    rec["tickets"][0][0][0] = 8
    path.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        tracker_main(
            [
                "settle",
                "--ledger",
                str(ledger),
                "--draw-key",
                "2026-02-02",
                "--actual-main",
                "1,3,6",
                "--actual-stars",
                "1",
            ]
        )
