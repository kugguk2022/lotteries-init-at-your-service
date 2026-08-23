import pandas as pd
import pytest
import requests

from euromillions.get_draws import (
    FetchError,
    NormalizationError,
    _cache_key,  # type: ignore[attr-defined]
    fetch_raw_csv,
    normalize,
)


class _FailingSession:
    def get(self, *args, **kwargs):
        raise requests.ConnectionError("boom")


def test_fetch_uses_cache_on_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("EUROMILLIONS_CACHE_DIR", str(tmp_path))
    from euromillions.get_draws import PRIMARY_URL  # local import to avoid cycle

    cache_file = _cache_key(PRIMARY_URL, {}, tmp_path)
    cache_file.write_text(
        "Date,Ball1,Ball2,Ball3,Ball4,Ball5,Lucky Star1,Lucky Star2\n", encoding="utf-8"
    )

    text = fetch_raw_csv(session=_FailingSession(), min_rows_full_history=1, allow_partial=True)

    assert "Ball1" in text


def test_fetch_raises_when_no_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("EUROMILLIONS_CACHE_DIR", str(tmp_path))
    with pytest.raises(FetchError):
        fetch_raw_csv(session=_FailingSession(), use_cache=False)


def test_normalize_rejects_bad_shape():
    bad_csv = "Date,Ball1,Ball2\n2024-01-01,1,2"
    with pytest.raises(NormalizationError):
        normalize(bad_csv)


def test_normalize_dedupes_duplicates():
    csv_text = (
        "Date,Ball1,Ball2,Ball3,Ball4,Ball5,Lucky Star1,Lucky Star2\n"
        "2024-01-02,1,2,3,4,5,1,2\n"
        "2024-01-02,1,2,3,4,5,1,2\n"
    )

    df = normalize(csv_text)

    assert len(df) == 1
    assert pd.to_datetime(df.iloc[0]["draw_date"]).date().isoformat() == "2024-01-02"


def test_fetch_falls_back_when_first_source_short(tmp_path, monkeypatch):
    from euromillions.get_draws import PRIMARY_URL, SECONDARY_URL, fetch_raw_csv

    short_csv = (
        "Date,Ball1,Ball2,Ball3,Ball4,Ball5,Lucky Star1,Lucky Star2\n2024-01-01,1,2,3,4,5,1,2\n"
    )
    long_csv = short_csv + short_csv + short_csv  # 3 rows -> still small but shows switch

    class _Response:
        def __init__(self, text: str):
            self.text = text
            self.headers = {"Content-Type": "text/csv"}

        def raise_for_status(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class _Session:
        def __init__(self):
            self.calls = []

        def get(self, url, params=None, timeout=5):
            self.calls.append(url)
            if url == PRIMARY_URL:
                return _Response(short_csv)
            return _Response(long_csv)

    monkeypatch.setenv("EUROMILLIONS_CACHE_DIR", str(tmp_path))
    session = _Session()
    text = fetch_raw_csv(
        session=session,
        urls=[PRIMARY_URL, SECONDARY_URL],
        min_rows_full_history=4,
        allow_partial=False,
    )

    assert SECONDARY_URL in session.calls
    assert text.count("\n") + 1 >= 4
