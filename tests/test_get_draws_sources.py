import pandas as pd

from euromillions import lottology as lottology_mod
from euromillions.get_draws import PEDRO_URL, PRIMARY_URL, FetchError, fetch_and_normalize


def test_fetch_and_normalize_lottology(monkeypatch, tmp_path):
    def fake_fetch(session=None):
        return [
            lottology_mod.EMRow("2024-01-02", 1, 2, 3, 4, 5, 1, 2),
            lottology_mod.EMRow("2024-01-09", 6, 7, 8, 9, 10, 3, 4),
        ]

    monkeypatch.setenv("EUROMILLIONS_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr("euromillions.get_draws.fetch_euromillions_lottology", fake_fetch)

    result = fetch_and_normalize(
        source="lottology",
        cache_dir=tmp_path,
        use_cache=False,
        date_from="2024-01-05",
    )

    assert list(result.dataframe.draw_date.astype(str)) == ["2024-01-09"]
    assert result.dataframe.iloc[0]["ball_5"] == 10
    assert "draw_date" in result.raw_csv.splitlines()[0]


def test_fetch_and_normalize_archive(monkeypatch, tmp_path):
    def fake_fetch(session=None, start_year=None, end_year=None):
        assert start_year == 2026
        assert end_year == 2026
        return [
            lottology_mod.EMRow("2026-05-01", 3, 9, 42, 46, 47, 1, 11),
            lottology_mod.EMRow("2026-05-05", 3, 4, 8, 20, 31, 6, 8),
        ]

    monkeypatch.setenv("EUROMILLIONS_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr("euromillions.get_draws.fetch_euromillions_archive", fake_fetch)

    result = fetch_and_normalize(
        source="archive",
        cache_dir=tmp_path,
        use_cache=False,
        date_from="2026-01-01",
        date_to="2026-12-31",
    )

    assert list(result.dataframe.draw_date.astype(str)) == ["2026-05-01", "2026-05-05"]
    assert result.dataframe.iloc[-1]["star_2"] == 8
    assert "draw_date" in result.raw_csv.splitlines()[0]


def test_fetch_and_normalize_auto_falls_back(monkeypatch, tmp_path):
    calls = []

    def fake_fetch_raw_csv(*, urls, **kwargs):
        calls.append(tuple(urls))
        if urls[0] == PRIMARY_URL:
            raise FetchError("primary down")
        return (
            "Date,Ball1,Ball2,Ball3,Ball4,Ball5,Lucky Star1,Lucky Star2\n"
            "2024-01-02,1,2,3,4,5,1,2\n"
        )

    monkeypatch.setattr("euromillions.get_draws.fetch_raw_csv", fake_fetch_raw_csv)
    result = fetch_and_normalize(source="auto", cache_dir=tmp_path, use_cache=False)

    assert calls[0] == (PRIMARY_URL,)
    assert calls[1] == (PEDRO_URL,)
    assert len(calls) == 2
    assert pd.to_datetime(result.dataframe.iloc[0]["draw_date"]).date().isoformat() == "2024-01-02"
