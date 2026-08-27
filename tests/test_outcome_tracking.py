from __future__ import annotations

import json

import pandas as pd
import pytest

from lotteries_core.likely_set_generator import CooccurrenceLevelSetProvider
from lotteries_core.outcome_tracker import (
    METHOD_CHOICES,
    PENDING,
    RESULTS,
    SETTLED,
    _validate_record,
    _validate_settlement,
)
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
        "uniform_random",
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


def _record(tmp_path, draw_key: str, ledger, extra: list[str] | None = None):
    history_path = tmp_path / "history.csv"
    if not history_path.exists():
        _history().to_csv(history_path, index=False)
    tracker_main(
        [
            "record", "--history", str(history_path), "--draw-key", draw_key,
            "--ledger", str(ledger), "--n-sets", "6", "--ticket-price", "2.50",
            "--main-n", "8", "--main-k", "3", "--star-n", "4", "--star-k", "1",
            *(extra or []),
        ]
    )


def _settle(ledger, draw_key: str, extra: list[str] | None = None):
    tracker_main(
        [
            "settle", "--ledger", str(ledger), "--draw-key", draw_key,
            "--actual-main", "1,3,6", "--actual-stars", "1", *(extra or []),
        ]
    )


def test_missing_payout_table_is_recorded_as_missing_not_as_zero(tmp_path):
    """A draw with no official breakdown must not settle as a EUR 0 prize.

    Recording zero would understate ROI for every such draw, and the bias would be invisible in
    `report` -- which is exactly the number the three-year experiment turns on.
    """
    ledger = tmp_path / "ledger"
    _record(tmp_path, "2026-02-01", ledger)
    _settle(ledger, "2026-02-01")

    row = pd.read_csv(ledger / RESULTS).iloc[0]
    assert row["payout_table_present"] == 0
    assert row["payout_source"] == "none"
    assert pd.isna(row["m_portfolio_prize"])
    assert pd.isna(row["m_net_return"])
    assert pd.isna(row["realized_roi"])
    assert row["stake"] == 15.0  # the stake is known even when the prize is not
    assert pd.isna(row["result_sha256"]) or row["result_sha256"] == ""


def test_settle_is_idempotent_and_skips_settled_draws(tmp_path):
    """Settlement runs unattended, so a repeat or a retried step must not duplicate evidence."""
    ledger = tmp_path / "ledger"
    _record(tmp_path, "2026-02-01", ledger)
    _settle(ledger, "2026-02-01")
    first = pd.read_csv(ledger / RESULTS)

    _settle(ledger, "2026-02-01")
    again = pd.read_csv(ledger / RESULTS)

    assert len(first) == len(again) == 1
    assert first.loc[0, "settled_utc"] == again.loc[0, "settled_utc"]
    assert (ledger / PENDING).read_text().strip() == ""


def test_force_rescores_a_settled_draw_in_place(tmp_path):
    """Correcting a mistyped result replaces the row rather than appending a second one."""
    ledger = tmp_path / "ledger"
    payouts = tmp_path / "payouts.json"
    payouts.write_text(json.dumps({"tiers": {"3+1": 1000}}), encoding="utf-8")
    _record(tmp_path, "2026-02-01", ledger)
    _settle(ledger, "2026-02-01")

    _settle(ledger, "2026-02-01", ["--payout-table", str(payouts), "--force"])
    results = pd.read_csv(ledger / RESULTS)
    assert len(results) == 1
    assert results.loc[0, "payout_table_present"] == 1
    assert results.loc[0, "payout_source"] == "official"
    settled = [json.loads(line) for line in (ledger / SETTLED).read_text().splitlines()]
    assert len(settled) == 1


def test_settled_records_pass_both_integrity_digests(tmp_path):
    """Preregistration and settlement are hashed separately, so scoring cannot look like tampering."""
    ledger = tmp_path / "ledger"
    _record(tmp_path, "2026-02-01", ledger)
    _settle(ledger, "2026-02-01")

    record = json.loads((ledger / SETTLED).read_text().strip())
    _validate_record(record)      # the preregistered portfolio is untouched by settlement
    _validate_settlement(record)  # and the scoring inputs are hashed in their own right
    assert record["settlement"]["actual_main"] == [1, 3, 6]
    assert record["settlement"]["actual_stars"] == [1]

    record["settlement"]["actual_main"] = [2, 4, 8]
    with pytest.raises(ValueError, match="settlement failed integrity"):
        _validate_settlement(record)


def test_report_counts_money_only_over_draws_with_payouts(tmp_path, capsys):
    ledger = tmp_path / "ledger"
    payouts = tmp_path / "payouts.json"
    payouts.write_text(json.dumps({"tiers": {"3+1": 1000}}), encoding="utf-8")
    _record(tmp_path, "2026-02-01", ledger)
    _record(tmp_path, "2026-02-04", ledger)
    _settle(ledger, "2026-02-01", ["--payout-table", str(payouts)])
    _settle(ledger, "2026-02-04")

    tracker_main(["report", "--ledger", str(ledger)])
    out = capsys.readouterr().out
    assert "settled draws              : 2" in out
    assert "with official payouts    : 1" in out
    # 6 tickets at 2.50 for the one draw that has prize data -- not 30.00 across both.
    assert "tracked stake              : 15.00 EUR" in out


def test_euromillions_defaults_need_no_flags(tmp_path, monkeypatch):
    """A scheduled run should not have to restate the game's price or its ledger path."""
    history_path = tmp_path / "history.csv"
    frame = pd.DataFrame(
        {
            "draw_date": [f"2026-01-{i + 1:02d}" for i in range(40)],
            **{f"ball_{b}": [(i * b) % 50 + 1 for i in range(40)] for b in range(1, 6)},
            **{f"star_{s}": [(i * s) % 12 + 1 for i in range(40)] for s in range(1, 3)},
        }
    )
    frame.to_csv(history_path, index=False)
    monkeypatch.chdir(tmp_path)

    tracker_main(["record", "--history", str(history_path), "--draw-key", "2026-08-18",
                  "--n-sets", "3"])

    ledger = tmp_path / "ledger" / "euromillions"
    record = json.loads((ledger / PENDING).read_text().strip())
    assert record["game"] == "euromillions"
    assert record["ticket_price"] == 2.50
