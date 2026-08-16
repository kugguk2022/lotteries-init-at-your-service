"""Tests for canonical dataset metadata -- provenance and staleness for the draw histories.

The failure these guard against is not a crash: it is a history quietly going stale or drifting while
documented results and sealed ledger entries keep being computed against it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lotteries_core import dataset


def _history(path, n: int = 60, last: str = "2026-08-14", seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(last)
    rows = []
    for i in range(n):
        mains = sorted(int(v) for v in rng.choice(np.arange(1, 51), 5, replace=False))
        stars = sorted(int(v) for v in rng.choice(np.arange(1, 13), 2, replace=False))
        rows.append(
            {
                "draw_date": (end - pd.Timedelta(days=4 * (n - 1 - i))).date().isoformat(),
                **{f"ball_{j+1}": mains[j] for j in range(5)},
                **{f"star_{j+1}": stars[j] for j in range(2)},
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def test_describe_captures_span_and_schema(tmp_path):
    csv = tmp_path / "euromillions.csv"
    _history(csv, n=60, last="2026-08-14")
    meta = dataset.describe(csv, game="euromillions")

    assert meta.rows == 60
    assert meta.last_draw == "2026-08-14"
    assert meta.first_draw < meta.last_draw
    assert meta.columns[:3] == ["draw_date", "ball_1", "ball_2"]
    assert len(meta.content_sha256) == 64
    assert meta.game == "euromillions"


def test_metadata_roundtrips_through_the_sidecar(tmp_path):
    csv = tmp_path / "euromillions.csv"
    _history(csv)
    written = dataset.write(csv, game="euromillions")
    assert dataset.meta_path(csv).name == "euromillions.meta.json"

    loaded = dataset.read(csv)
    assert loaded == written


def test_read_returns_none_when_never_written(tmp_path):
    csv = tmp_path / "euromillions.csv"
    _history(csv)
    assert dataset.read(csv) is None
    ok, reason = dataset.verify(csv)
    assert not ok
    assert "no metadata" in reason


def test_digest_ignores_column_order_but_not_content(tmp_path):
    """Re-saving with reordered columns is not a data change; editing a draw is."""
    csv = tmp_path / "euromillions.csv"
    df = _history(csv)
    original = dataset.content_digest(df)

    reordered = df.reindex(sorted(df.columns, reverse=True), axis=1)
    assert dataset.content_digest(reordered) == original

    edited = df.copy()
    edited.loc[0, "ball_1"] = int(edited.loc[0, "ball_1"]) % 50 + 1
    assert dataset.content_digest(edited) != original


def test_verify_detects_drift(tmp_path):
    csv = tmp_path / "euromillions.csv"
    df = _history(csv)
    dataset.write(csv, game="euromillions")
    assert dataset.verify(csv)[0]

    df.loc[0, "star_1"] = int(df.loc[0, "star_1"]) % 12 + 1
    df.to_csv(csv, index=False)
    ok, reason = dataset.verify(csv)
    assert not ok
    assert "content changed" in reason


def test_staleness_is_measured_from_the_newest_draw(tmp_path):
    csv = tmp_path / "euromillions.csv"
    _history(csv, last="2026-08-14")
    dataset.write(csv, game="euromillions")

    assert dataset.staleness_days(csv, today="2026-08-16") == 2
    assert not dataset.is_stale(csv, today="2026-08-16")
    # Two draws a week, so a fortnight past the newest draw means completed draws are missing.
    assert dataset.staleness_days(csv, today="2026-08-28") == 14
    assert dataset.is_stale(csv, today="2026-08-28")


def test_staleness_works_without_a_sidecar(tmp_path):
    csv = tmp_path / "euromillions.csv"
    _history(csv, last="2026-08-14")
    assert dataset.read(csv) is None
    assert dataset.staleness_days(csv, today="2026-08-16") == 2


def test_empty_or_dateless_history_is_rejected(tmp_path):
    empty = tmp_path / "empty.csv"
    pd.DataFrame(columns=["draw_date", "ball_1"]).to_csv(empty, index=False)
    with pytest.raises(ValueError, match="no rows"):
        dataset.describe(empty, game="euromillions")

    dateless = tmp_path / "dateless.csv"
    pd.DataFrame({"ball_1": [1, 2]}).to_csv(dateless, index=False)
    with pytest.raises(ValueError, match="no draw_date/date column"):
        dataset.describe(dateless, game="euromillions")


def test_the_committed_history_is_described_and_current():
    """The real dataset the ledger and documented results depend on."""
    csv = "data/euromillions.csv"
    meta = dataset.read(csv)
    if meta is None:
        pytest.skip("data/euromillions.csv has no sidecar yet; run scripts/refresh_history.py")
    ok, reason = dataset.verify(csv)
    assert ok, reason
    assert meta.rows > 1000
    assert meta.game == "euromillions"
