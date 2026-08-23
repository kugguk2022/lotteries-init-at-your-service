"""Smoke test executed from outside the repository against an installed wheel.

This asserts the *published* contract, not the repository's: what a user gets from
``pip install lottobench`` and nothing else. It therefore covers the whole documented journey --
retrieval, normalization, storage, benchmarking -- with only the network replaced, because that
journey previously could not be completed from the wheel at all: the retriever lived in
``experiments/``, which is not packaged.

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
from pathlib import Path

import pandas as pd

import lotteries_core
import lottobench
from lotteries_core import storage

# --- the published surface -------------------------------------------------------------------

assert lotteries_core.__version__ == "0.1.0a1"
assert lottobench.game("euromillions").spec.main_n == 50
assert "frequency" in lottobench.names()
assert lottobench.create("frequency").name == "frequency"
assert callable(lottobench.compare_realized_roi)

# Only what is supported end to end may be in the catalogue. A game the wheel cannot fetch must
# not be returned as if it were usable.
assert set(lottobench.GAMES) == {"euromillions"}
for backlog_key in ("uk-lotto", "de-lotto-6aus49", "dk-lotto", "nl-lotto", "se-lotto"):
    try:
        lottobench.game(backlog_key)
    except KeyError as exc:
        assert "not supported end to end" in str(exc), exc
    else:  # pragma: no cover - a regression would take this branch
        raise AssertionError(f"{backlog_key} was advertised as supported")

# --- the base install carries no scraping stack ------------------------------------------------

try:
    from lotteries_core.sources.html_archive import fetch_archive

    fetch_archive()
except ImportError as exc:
    assert "lottobench[scrape]" in str(exc), exc
else:  # pragma: no cover
    raise AssertionError("HTML scraper ran without the scrape extra installed")


# --- the documented journey, with only the network replaced ------------------------------------


class _Headers:
    def get(self, key, default=""):
        return "text/csv" if key == "Content-Type" else default

    def get_content_charset(self):
        return "utf-8"


class _Response(io.BytesIO):
    def __init__(self, body: str) -> None:
        super().__init__(body.encode("utf-8"))
        self.headers = _Headers()

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


urllib.request.urlopen = lambda request, timeout=None: _Response(_published_csv())

from lottobench.cli import main as cli_main  # noqa: E402  (must follow the transport stub)

with tempfile.TemporaryDirectory(prefix="lottobench-wheel-journey-") as directory:
    root = Path(directory)
    os.environ["LOTTOBENCH_CACHE_DIR"] = str(root / "cache")
    database = root / "lotteries.db"

    assert cli_main(["fetch", "--game", "euromillions", "--db", str(database)]) == 0
    stored = storage.read_history(database, game="euromillions")
    assert len(stored) == 400, f"stored {len(stored)} draws"

    provenance = storage.read_metadata(database, game="euromillions")
    assert provenance is not None
    assert provenance["rows"] == 400
    assert provenance["source"].startswith("lottobench.fetch:")
    assert len(provenance["content_sha256"]) == 64

    summary_path = root / "summary.json"
    assert (
        cli_main(
            [
                "benchmark", "--game", "euromillions", "--db", str(database),
                "--budget", "10", "--holdout", "5", "--out", str(summary_path),
            ]
        )
        == 0
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["providers"], "the benchmark scored nothing"
    assert "uniform_random" in summary["providers"], "the matched control must always run"

os.environ.pop("LOTTOBENCH_CACHE_DIR", None)

# --- console entry points ----------------------------------------------------------------------

games = subprocess.run(
    [sys.executable, "-m", "lottobench.cli", "games"],
    check=True, capture_output=True, text=True,
)
assert "euromillions" in games.stdout
assert "1 game supported end to end" in games.stdout
assert "not yet supported" in games.stdout, "backlog games must be labelled, not hidden"

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

print("installed LottoBench wheel smoke test passed: fetch -> store -> benchmark")
