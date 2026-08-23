"""Smoke test executed from outside the repository against an installed wheel.

This asserts the *published* contract, not the repository's: what a user gets from
``pip install lottobench`` and nothing else. It therefore covers the whole documented journey --
retrieval, normalization, storage, benchmarking -- for **every** game the wheel advertises, with
only the network replaced. A game listed in ``lottobench.GAMES`` that cannot complete that journey
from a clean install is the exact packaging bug this script exists to catch.

It runs with no repository files available, so every input is generated here.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from importlib.metadata import version as distribution_version
from pathlib import Path

import pandas as pd

import lotteries_core
import lottobench
from lotteries_core import storage

# --- the published surface -------------------------------------------------------------------

# Pinned against the installed distribution rather than a literal: the contract is that the
# importable version and the wheel's own metadata agree, and that assertion cannot go stale on a
# version bump the way a hard-coded string does.
assert lotteries_core.__version__ == distribution_version("lottobench"), (
    f"lotteries_core.__version__ is {lotteries_core.__version__!r} but the installed "
    f"distribution is {distribution_version('lottobench')!r}"
)
assert lottobench.__version__ == lotteries_core.__version__
assert "frequency" in lottobench.names()
assert lottobench.create("frequency").name == "frequency"
assert callable(lottobench.compare_realized_roi)
assert callable(lottobench.export_roi_bundle)

# Only what is supported end to end may be in the catalogue. A game the wheel cannot fetch must
# not be returned as if it were usable.
SUPPORTED = {"euromillions", "nl-lotto"}
BACKLOG = {"uk-lotto", "de-lotto-6aus49", "dk-lotto", "se-lotto"}

assert set(lottobench.GAMES) == SUPPORTED, sorted(lottobench.GAMES)
assert lottobench.game("euromillions").spec.main_n == 50
assert lottobench.game("euromillions").spec.main_k == 5
assert lottobench.game("euromillions").spec.star_k == 2
assert lottobench.game("nl-lotto").spec.main_n == 45
assert lottobench.game("nl-lotto").spec.main_k == 6
assert lottobench.game("nl-lotto").spec.star_k == 0

for backlog_key in sorted(BACKLOG):
    try:
        lottobench.game(backlog_key)
    except KeyError as exc:
        assert "not supported end to end" in str(exc), exc
    else:  # pragma: no cover - a regression would take this branch
        raise AssertionError(f"{backlog_key} was advertised as supported")

try:
    lottobench.game("not-a-lottery")
except KeyError as exc:
    assert "unknown game" in str(exc), exc
else:  # pragma: no cover
    raise AssertionError("an unknown key was resolved as a game")

# --- the base install carries the retrieval stack ----------------------------------------------
#
# ``requests`` and ``beautifulsoup4`` are base dependencies, not an extra: automatic retrieval,
# including the HTML archive fallback that ``--source auto`` reaches for when both CSV archives
# fail, has to work on a plain ``pip install lottobench``.

from lotteries_core.sources.html_archive import (  # noqa: E402
    _require_scrape_dependencies,
    fetch_archive,
    fetch_lottology,
)

requests_module, soup_factory = _require_scrape_dependencies()
assert requests_module.__name__ == "requests"
assert soup_factory.__name__ == "BeautifulSoup"
assert callable(fetch_archive) and callable(fetch_lottology)


# --- the documented journey, with only the network replaced ------------------------------------

NL_RESULTS_HOST = "lotto.nederlandseloterij.nl"
NL_RESULT_API_PATH = "/api/draws/results/"

#: Fixed past dates, so ``published_dates`` (which drops anything later than today) keeps all of
#: them no matter when this runs.
NL_DATES = [
    (pd.Timestamp("2024-01-06") + pd.Timedelta(days=index * 7)).date().isoformat()
    for index in range(40)
]


class _Headers:
    def __init__(self, content_type: str) -> None:
        self._content_type = content_type

    def get(self, key, default=""):
        return self._content_type if key == "Content-Type" else default

    def get_content_charset(self):
        return "utf-8"


class _Response(io.BytesIO):
    def __init__(self, body: str, content_type: str = "text/csv") -> None:
        super().__init__(body.encode("utf-8"))
        self.headers = _Headers(content_type)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _published_csv(rows: int = 400) -> str:
    lines = ["DrawDate,Ball1,Ball2,Ball3,Ball4,Ball5,Lucky Star1,Lucky Star2"]
    for index in range(rows):
        day = pd.Timestamp("2019-01-01") + pd.Timedelta(days=index * 3)
        base = index % 40
        lines.append(
            f"{day.date().isoformat()},{base + 1},{base + 2},{base + 4},"
            f"{base + 7},{base + 10},{index % 12 + 1},{(index + 5) % 12 + 1}"
        )
    return "\n".join(lines) + "\n"


def _nl_results_page() -> str:
    listed = "".join(f'<li data-draw-date="{value}">{value}</li>' for value in NL_DATES)
    return f"<html><body><ul>{listed}</ul></body></html>"


def _nl_result_payload(draw_date: str) -> str:
    index = NL_DATES.index(draw_date)
    numbers = sorted({(index + step * 7) % 45 + 1 for step in range(6)})
    assert len(numbers) == 6, numbers
    return json.dumps(
        {
            "results": [
                {
                    "draw": {"drawDate": draw_date},
                    "isXlDraw": False,
                    "winningNumbers": {"numbers": numbers, "bonusNumber": 45},
                    "jackpotAmountGross": 100000000,
                    "isJackpotWon": False,
                    "jackpotGuaranteed": index % 2 == 0,
                },
                {
                    # Lotto XL shares the response and must never enter the primary history.
                    "draw": {"drawDate": draw_date},
                    "isXlDraw": True,
                    "winningNumbers": {"numbers": [4, 14, 15, 25, 31, 41], "bonusNumber": 16},
                },
            ]
        }
    )


def _fake_urlopen(request, timeout=None):
    url = getattr(request, "full_url", request)
    if NL_RESULT_API_PATH in url:
        return _Response(_nl_result_payload(url.rsplit("/", 1)[-1]), "application/json")
    if NL_RESULTS_HOST in url:
        return _Response(_nl_results_page(), "text/html")
    return _Response(_published_csv(), "text/csv")


urllib.request.urlopen = _fake_urlopen

from lottobench.cli import main as cli_main  # noqa: E402  (must follow the transport stub)

#: Per game: how many draws the stubbed source publishes, the provenance label ``fetch`` records,
#: and a benchmark small enough to stay a smoke test. Every entry of ``GAMES`` must appear here.
JOURNEYS = {
    "euromillions": {"rows": 400, "budget": 10, "holdout": 5, "source": "auto"},
    "nl-lotto": {
        "rows": len(NL_DATES), "budget": 4, "holdout": 3, "source": "official-operator-api",
    },
}
assert set(JOURNEYS) == set(lottobench.GAMES), "every advertised game must be exercised here"

with tempfile.TemporaryDirectory(prefix="lottobench-wheel-journey-") as directory:
    root = Path(directory)
    os.environ["LOTTOBENCH_CACHE_DIR"] = str(root / "cache")
    database = root / "lotteries.db"

    for game_key, journey in JOURNEYS.items():
        assert cli_main(["fetch", "--game", game_key, "--db", str(database)]) == 0

        stored = storage.read_history(database, game=game_key)
        assert len(stored) == journey["rows"], f"{game_key}: stored {len(stored)} draws"

        provenance = storage.read_metadata(database, game=game_key)
        assert provenance is not None, game_key
        assert provenance["rows"] == journey["rows"], game_key
        assert provenance["source"] == f"lottobench.fetch:{journey['source']}", provenance["source"]
        assert len(provenance["content_sha256"]) == 64, game_key

        summary_path = root / f"{game_key}-summary.json"
        assert (
            cli_main(
                [
                    "benchmark", "--game", game_key, "--db", str(database),
                    "--budget", str(journey["budget"]), "--holdout", str(journey["holdout"]),
                    "--out", str(summary_path),
                ]
            )
            == 0
        ), game_key
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["providers"], f"{game_key}: the benchmark scored nothing"
        assert "uniform_random" in summary["providers"], (
            f"{game_key}: the matched control must always run"
        )

    # One database, several games: storing a second game must not disturb the first.
    first_key = next(iter(JOURNEYS))
    assert len(storage.read_history(database, game=first_key)) == JOURNEYS[first_key]["rows"]

os.environ.pop("LOTTOBENCH_CACHE_DIR", None)

# --- console entry points ----------------------------------------------------------------------

games = subprocess.run(
    [sys.executable, "-m", "lottobench.cli", "games"],
    check=True, capture_output=True, text=True,
)
for supported_key in sorted(SUPPORTED):
    assert supported_key in games.stdout, supported_key
assert f"{len(SUPPORTED)} games supported end to end" in games.stdout, games.stdout
assert "not yet supported" in games.stdout, "backlog games must be labelled, not hidden"
for backlog_key in sorted(BACKLOG):
    assert backlog_key in games.stdout, backlog_key

providers = subprocess.run(
    [sys.executable, "-m", "lottobench.cli", "providers"],
    check=True, capture_output=True, text=True,
)
assert "frequency" in providers.stdout
assert "uniform_random" in providers.stdout
assert "garch_markov_branch" in providers.stdout
assert "sequence_transformer" in providers.stdout
assert "12 selectable entrants backed by 9 implementation families" in providers.stdout
assert "install [transformer] extra" in providers.stdout
assert "install [ml] extra" in providers.stdout, "ml_ensemble must report its missing extra"
assert "1.0.0" in providers.stdout

subprocess.run(
    [sys.executable, "-m", "lotteries_core.realized_roi", "--help"],
    check=True, capture_output=True, text=True,
)

print(
    "installed LottoBench wheel smoke test passed: fetch -> store -> benchmark for "
    + ", ".join(sorted(SUPPORTED))
)
