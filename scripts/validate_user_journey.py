"""Offline end-to-end validation of the supported LottoBench user journey."""

from __future__ import annotations

import io
import os
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import lottobench
from lotteries_core import dataset, registry, storage
from lotteries_core.evaluation import evaluate_forward
from lotteries_core.providers import FrequencyProvider, UnpopularityProvider
from lotteries_core.sources import euromillions as em
from lottobench.cli import main as cli_main


def _history(rows: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    records = []
    for index in range(rows):
        main = sorted(int(value) for value in rng.choice(np.arange(1, 51), 5, replace=False))
        stars = sorted(int(value) for value in rng.choice(np.arange(1, 13), 2, replace=False))
        records.append(
            {
                "draw_date": (
                    pd.Timestamp("2026-01-01") + pd.Timedelta(days=index * 4)
                ).date().isoformat(),
                **{f"ball_{position + 1}": value for position, value in enumerate(main)},
                **{f"star_{position + 1}": value for position, value in enumerate(stars)},
            }
        )
    return pd.DataFrame(records)


class _StubResponse(io.BytesIO):
    """What urlopen returns, reduced to what the CSV adapter actually reads."""

    def __init__(self, body: str) -> None:
        super().__init__(body.encode("utf-8"))
        self.headers = _StubHeaders()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _StubHeaders:
    def get(self, key, default=""):
        return "text/csv" if key == "Content-Type" else default

    def get_content_charset(self):
        return "utf-8"


def _published_csv(rows: int = 400) -> str:
    """A payload shaped like the published archive, including its column spellings."""
    lines = ["DrawDate,Ball1,Ball2,Ball3,Ball4,Ball5,Lucky Star1,Lucky Star2"]
    for index in range(rows):
        day = pd.Timestamp("2019-01-01") + pd.Timedelta(days=index * 3)
        base = index % 40
        lines.append(
            f"{day.date().isoformat()},{base + 1},{base + 2},{base + 4},"
            f"{base + 7},{base + 10},{index % 12 + 1},{(index + 5) % 12 + 1}"
        )
    return "\n".join(lines) + "\n"


def _validate_retrieval() -> None:
    """Exercise the shipped fetch path over a stubbed transport.

    Creating synthetic data proves the maths but not that a user can obtain data. Retrieval used
    to live outside the shipped package, so this stage exists to keep that from regressing: it
    drives the real adapter and the real CLI, replacing only the network.
    """
    original = urllib.request.urlopen
    with tempfile.TemporaryDirectory(prefix="lottobench-fetch-") as directory:
        root = Path(directory)
        os.environ["LOTTOBENCH_CACHE_DIR"] = str(root / "cache")
        urllib.request.urlopen = lambda request, timeout=None: _StubResponse(_published_csv())
        try:
            frame = em.fetch_euromillions()
            assert list(frame.columns) == em.CANONICAL_COLUMNS
            assert len(frame) == 400
            assert frame["draw_date"].is_monotonic_increasing

            db = root / "lotteries.db"
            assert cli_main(["fetch", "--game", "euromillions", "--db", str(db)]) == 0
            stored = storage.read_history(db, game="euromillions")
            assert len(stored) == 400

            assert cli_main(
                ["benchmark", "--game", "euromillions", "--db", str(db),
                 "--budget", "4", "--holdout", "3"]
            ) == 0

            provenance = storage.read_metadata(db, game="euromillions")
            assert provenance is not None and provenance["rows"] == 400
            assert provenance["source"].startswith("lottobench.fetch:")
        finally:
            urllib.request.urlopen = original
            os.environ.pop("LOTTOBENCH_CACHE_DIR", None)


def main() -> int:
    definition = lottobench.game("euromillions")
    frame = _history()

    with tempfile.TemporaryDirectory(prefix="lottobench-e2e-") as directory:
        root = Path(directory)
        csv_path = root / "history.csv"
        db_path = root / "lotteries.db"
        exported_path = root / "exported.csv"
        frame.to_csv(csv_path, index=False)

        imported = storage.import_csv(csv_path, db_path, game=definition.key)
        metadata = dataset.write(db_path, game=definition.key, source="offline-e2e-fixture")
        ok, reason = dataset.verify(db_path, game=definition.key)
        exported = storage.export_csv(db_path, exported_path, game=definition.key)
        roundtrip = pd.read_csv(exported_path)

        assert imported == exported == len(frame)
        assert metadata.rows == len(frame)
        assert ok, reason
        pd.testing.assert_frame_equal(frame, roundtrip)

        expected = {
            "gingerm", "spectral_contrarian", "parallax", "frequency", "unpopularity",
            "perron_frobenius_affinity", "perron_frobenius_uniform",
            "parallax_guard_ablation", "ml_ensemble", "garch_markov_branch",
            "sequence_transformer",
            "uniform_random",
        }
        assert set(registry.names()) == expected
        for name in registry.names():
            if not registry.PROVIDERS[name].optional:
                assert registry.create(name).name == name

        summary = evaluate_forward(
            roundtrip,
            definition.spec,
            [FrequencyProvider(), UnpopularityProvider()],
            budget=4,
            holdout=4,
            seed=7,
        )
        assert set(summary["providers"]) == {"frequency", "unpopularity"}
        assert summary["providers"]["frequency"]["expected_roi_per_ticket"] < 0
        assert summary["providers"]["unpopularity"]["expected_roi_per_ticket"] < 0

    _validate_retrieval()

    print(
        "LottoBench E2E passed: mocked retrieval -> 12-provider registry -> "
        "benchmark/ROI -> storage/provenance"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
