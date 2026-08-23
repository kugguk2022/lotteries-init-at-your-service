"""Offline end-to-end validation of the supported LottoBench user journey."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import lottobench
from lotteries_core import dataset, registry, storage
from lotteries_core.evaluation import evaluate_forward
from lotteries_core.providers import FrequencyProvider, UnpopularityProvider


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

    print("LottoBench E2E passed: 12-provider registry -> benchmark/ROI -> storage/provenance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
