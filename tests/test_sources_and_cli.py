"""The documented journey -- fetch, store, benchmark -- exercised against mocked retrieval.

Synthetic-data tests prove the maths. They do not prove a user can obtain data, which is the step
that was broken: retrieval lived outside the shipped package, so `pip install lottobench` produced
a benchmark with no way to feed it. These tests drive the real CLI over a stubbed HTTP layer, so
the fetch path itself is covered without touching a public archive.
"""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path

import pandas as pd
import pytest

from lotteries_core import storage
from lotteries_core.sources import euromillions as em
from lotteries_core.sources import netherlands as nl
from lotteries_core.sources.errors import ContentTypeError, FetchError, NormalizationError
from lottobench import cli, games


def _archive_csv(rows: int = 400) -> str:
    """A payload shaped like the published archive: header plus `rows` plausible draws."""
    lines = ["DrawDate,Ball1,Ball2,Ball3,Ball4,Ball5,Lucky Star1,Lucky Star2"]
    for index in range(rows):
        day = pd.Timestamp("2019-01-01") + pd.Timedelta(days=index * 3)
        base = index % 40
        lines.append(
            f"{day.date().isoformat()},{base + 1},{base + 2},{base + 4},"
            f"{base + 7},{base + 10},{index % 12 + 1},{(index + 5) % 12 + 1}"
        )
    return "\n".join(lines) + "\n"


class _Response(io.BytesIO):
    """Minimal stand-in for the object urlopen returns as a context manager."""

    def __init__(self, body: str, content_type: str = "text/csv") -> None:
        super().__init__(body.encode("utf-8"))
        self.headers = _Headers(content_type)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Headers:
    def __init__(self, content_type: str) -> None:
        self._content_type = content_type

    def get(self, key, default=""):
        return self._content_type if key == "Content-Type" else default

    def get_content_charset(self):
        return "utf-8"


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Point the on-disk payload cache at tmp_path so tests never share state."""
    monkeypatch.setenv("LOTTOBENCH_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path


def _stub_urlopen(monkeypatch, handler):
    monkeypatch.setattr(em.urllib.request, "urlopen", lambda request, timeout=None: handler(request))


def test_fetch_normalizes_the_published_column_spellings(isolated_cache, monkeypatch):
    _stub_urlopen(monkeypatch, lambda request: _Response(_archive_csv()))

    frame = em.fetch_euromillions()

    assert list(frame.columns) == em.CANONICAL_COLUMNS
    assert len(frame) == 400
    assert frame["draw_date"].is_monotonic_increasing
    assert frame["star_1"].between(1, 12).all()
    assert frame["ball_1"].between(1, 50).all()


def test_a_truncated_payload_falls_through_to_the_next_source(isolated_cache, monkeypatch):
    """Three rows is an error page, not a short history. It must not become the dataset."""
    seen: list[str] = []

    def handler(request):
        seen.append(request.full_url)
        if em.PRIMARY_URL.split("?")[0] in request.full_url:
            return _Response(_archive_csv(rows=3))
        return _Response(_archive_csv())

    _stub_urlopen(monkeypatch, handler)
    frame = em.fetch_euromillions()

    assert len(seen) == 2, "the truncated primary should not have satisfied the request"
    assert len(frame) == 400


def test_an_html_error_page_served_as_200_is_rejected(isolated_cache, monkeypatch):
    _stub_urlopen(
        monkeypatch, lambda request: _Response("<html>blocked</html>", content_type="text/html")
    )

    with pytest.raises(FetchError):
        em.fetch_euromillions(source="national-lottery")


def test_a_non_text_content_type_is_refused(isolated_cache, monkeypatch):
    _stub_urlopen(
        monkeypatch,
        lambda request: _Response(_archive_csv(), content_type="application/octet-stream"),
    )

    with pytest.raises((FetchError, ContentTypeError)):
        em.fetch_euromillions(source="national-lottery")


def test_every_source_failing_reports_each_attempt(isolated_cache, monkeypatch):
    def handler(request):
        raise urllib.error.URLError("name resolution failed")

    _stub_urlopen(monkeypatch, handler)
    with pytest.raises(FetchError) as excinfo:
        em.fetch_euromillions(source="national-lottery", timeout=0.01)

    message = str(excinfo.value)
    assert "national-lottery" in message


def test_unparseable_payload_raises_normalization_error():
    with pytest.raises(NormalizationError):
        em.normalize("not,a,draw\nfile\x00\x01")


def test_draw_validation_rejects_fractional_and_duplicate_numbers():
    fractional = (
        "DrawDate,Ball1,Ball2,Ball3,Ball4,Ball5,Lucky Star1,Lucky Star2\n"
        "2026-01-01,1.9,2,3,4,5,1,2\n"
    )
    duplicate = (
        "DrawDate,Ball1,Ball2,Ball3,Ball4,Ball5,Lucky Star1,Lucky Star2\n"
        "2026-01-01,1,1,3,4,5,1,2\n"
    )
    with pytest.raises(ValueError, match="non-integer"):
        em.normalize(fractional)
    with pytest.raises(ValueError, match="duplicate main"):
        em.normalize(duplicate)


def test_headerless_payloads_are_recovered():
    body = "\n".join(
        f"2020-01-{day:02d},{day},{day + 5},{day + 9},{day + 14},{day + 20},"
        f"{day % 12 + 1},{(day + 5) % 12 + 1}"
        for day in range(1, 20)
    )
    frame = em.normalize(body)
    assert list(frame.columns) == em.CANONICAL_COLUMNS
    assert len(frame) == 19


def test_cli_fetch_then_benchmark_is_the_whole_journey(isolated_cache, monkeypatch, capsys, tmp_path):
    """Item under test: `lottobench fetch` followed by `lottobench benchmark`, no CSV in sight."""
    _stub_urlopen(monkeypatch, lambda request: _Response(_archive_csv()))
    db = tmp_path / "lotteries.db"

    assert cli.main(["fetch", "--game", "euromillions", "--db", str(db)]) == 0
    fetch_output = capsys.readouterr().out
    assert "400 draws" in fetch_output
    assert "lottobench benchmark" in fetch_output, "fetch should name the next step"

    stored = storage.read_history(db, game="euromillions")
    assert len(stored) == 400

    summary_path = tmp_path / "summary.json"
    assert (
        cli.main(
            [
                "benchmark", "--game", "euromillions", "--db", str(db),
                "--budget", "10", "--holdout", "5", "--out", str(summary_path),
            ]
        )
        == 0
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["providers"], "the benchmark must score at least one provider"
    assert "uniform_random" in summary["providers"], "the control must always be present"


def test_benchmark_without_a_database_says_what_to_run(tmp_path, capsys):
    missing = tmp_path / "absent.db"
    with pytest.raises(SystemExit):
        cli.main(["benchmark", "--game", "euromillions", "--db", str(missing)])
    assert "lottobench fetch" in capsys.readouterr().err


def test_backlog_games_are_refused_with_a_reason(capsys):
    with pytest.raises(SystemExit):
        cli.main(["fetch", "--game", "uk-lotto"])
    error = capsys.readouterr().err
    assert "not supported end to end" in error
    assert "Backlog.md" in error


def test_unknown_game_is_distinguished_from_a_backlog_game():
    with pytest.raises(KeyError, match="unknown game"):
        games.game("not-a-lottery")
    with pytest.raises(KeyError, match="not supported end to end"):
        games.game("se-lotto")


def test_catalogue_advertises_only_what_is_supported():
    """The packaging bug this whole change exists to fix."""
    assert set(games.GAMES) == {"euromillions", "nl-lotto"}
    assert "uk-lotto" in games.BACKLOG_GAMES
    for entry in games.BACKLOG_GAMES.values():
        assert entry.blocked_on, f"{entry.key} must say what it is blocked on"


def test_netherlands_official_result_normalization_keeps_metadata():
    payload = {
        "results": [
            {
                "draw": {"drawDate": "2026-08-22"},
                "isXlDraw": False,
                "winningNumbers": {"numbers": [33, 11, 30, 19, 32, 29], "bonusNumber": 40},
                "jackpotAmountGross": 100000000,
                "isJackpotWon": False,
                "jackpotGuaranteed": True,
            },
            {
                "draw": {"drawDate": "2026-08-22"},
                "isXlDraw": True,
                "winningNumbers": {"numbers": [4, 14, 15, 25, 31, 41], "bonusNumber": 16},
            },
        ]
    }
    rows = nl.normalize_result(payload)
    assert rows[0]["ball_1"] == 11
    assert rows[0]["ball_6"] == 33
    assert rows[0]["reserve_number"] == 40
    assert rows[0]["jackpot_color"] == "yellow"


def test_netherlands_published_dates_only_uses_completed_draw_slider_links():
    html = """
    <script>Christmas 2024-12-17 to 2024-12-27; upcoming 2999-01-01</script>
    <a href="/trekkingsuitslag/2024-12-21" data-test="date-slider-item">result</a>
    <a href="/trekkingsuitslag/2999-01-01" data-test="upcoming-draw">open</a>
    """
    assert nl.published_dates(html) == ["2024-12-21"]


def test_netherlands_cli_fetch_store_and_benchmark(monkeypatch, tmp_path):
    dates = [
        (pd.Timestamp("2025-01-04") + pd.Timedelta(days=index * 7)).date().isoformat()
        for index in range(30)
    ]

    def fake_get(url, *, timeout):
        if url == nl.RESULTS_PAGE:
            return " ".join(
                f'<a href="/trekkingsuitslag/{value}" data-test="date-slider-item">draw</a>'
                for value in dates
            )
        draw_date = url.rsplit("/", 1)[-1]
        index = dates.index(draw_date)
        numbers = sorted({(index + step * 7) % 45 + 1 for step in range(6)})
        return json.dumps(
            {
                "results": [
                    {
                        "draw": {"drawDate": draw_date},
                        "isXlDraw": False,
                        "winningNumbers": {"numbers": numbers, "bonusNumber": 45},
                        "jackpotAmountGross": 100000000,
                        "isJackpotWon": False,
                        "jackpotGuaranteed": False,
                    }
                ]
            }
        )

    monkeypatch.setattr(nl, "_get_text", fake_get)
    db = tmp_path / "lotteries.db"
    assert cli.main(["fetch", "--game", "nl-lotto", "--db", str(db)]) == 0
    stored = storage.read_history(db, game="nl-lotto")
    assert len(stored) == 30
    assert stored.iloc[0]["jackpot_color"] == "black"
    assert cli.main(
        [
            "benchmark", "--game", "nl-lotto", "--db", str(db),
            "--budget", "4", "--holdout", "3", "--out", str(tmp_path / "nl.json"),
        ]
    ) == 0


def test_html_source_dependency_failure_is_actionable(monkeypatch):
    """A damaged base installation should produce a useful repair command."""
    import sys

    from lotteries_core.sources import html_archive

    monkeypatch.setitem(sys.modules, "bs4", None)
    with pytest.raises(ImportError, match=r"force-reinstall lottobench"):
        html_archive._require_scrape_dependencies()


def test_cache_avoids_a_second_request(isolated_cache, monkeypatch):
    calls: list[str] = []

    def handler(request):
        calls.append(request.full_url)
        return _Response(_archive_csv())

    _stub_urlopen(monkeypatch, handler)
    em.fetch_euromillions()
    first = len(calls)
    em.fetch_euromillions()

    assert len(calls) == first, "a cached payload should not re-hit the archive"


def test_no_cache_forces_a_request(isolated_cache, monkeypatch):
    calls: list[str] = []

    def handler(request):
        calls.append(request.full_url)
        return _Response(_archive_csv())

    _stub_urlopen(monkeypatch, handler)
    em.fetch_euromillions()
    em.fetch_euromillions(use_cache=False)

    assert len(calls) == 2


def test_stored_history_round_trips_through_export(isolated_cache, monkeypatch, tmp_path):
    _stub_urlopen(monkeypatch, lambda request: _Response(_archive_csv()))
    db = tmp_path / "lotteries.db"
    cli.main(["fetch", "--game", "euromillions", "--db", str(db)])

    out = tmp_path / "exported.csv"
    assert cli.main(["export-csv", str(out), "--game", "euromillions", "--db", str(db)]) == 0
    assert len(pd.read_csv(out)) == 400


def test_fetch_records_provenance_for_the_stored_rows(isolated_cache, monkeypatch, tmp_path):
    _stub_urlopen(monkeypatch, lambda request: _Response(_archive_csv()))
    db = tmp_path / "lotteries.db"
    cli.main(["fetch", "--game", "euromillions", "--db", str(db)])

    metadata = storage.read_metadata(db, game="euromillions")
    assert metadata is not None
    assert metadata["rows"] == 400
    assert metadata["source"].startswith("lottobench.fetch:")
    assert len(metadata["content_sha256"]) == 64


def test_source_choice_is_validated():
    with pytest.raises(ValueError, match="unknown source"):
        em.fetch_euromillions(source="wikipedia")


def test_wheel_entry_point_is_importable():
    """`lottobench = lottobench.cli:main` must resolve from the installed package."""
    assert callable(cli.main)
    assert Path(cli.__file__).name == "cli.py"
