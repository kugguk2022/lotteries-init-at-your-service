"""Build redistribution-safe Hugging Face and Kaggle community benchmark artifacts.

Only deterministic synthetic draws and derived metrics are emitted. Operator-sourced histories,
prospective ledgers, tickets, and payout records are deliberately excluded.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from lotteries_core import registry
from lotteries_core.evaluation import evaluate_forward
from lotteries_core.protocol import GameSpec
from lotteries_core.roi import JackpotModel

ROOT = Path(__file__).resolve().parents[1]
PUBLISHING = ROOT / "publishing"
COMMON = PUBLISHING / "common"
TARGETS = (
    PUBLISHING / "huggingface" / "data",
    PUBLISHING / "huggingface-space" / "data",
    PUBLISHING / "kaggle" / "dataset",
)
SEED = 20260828


def synthetic_history(rows: int = 48) -> pd.DataFrame:
    """Create a stable one-main-pool plus auxiliary-pool history with no external data."""
    rng = np.random.default_rng(SEED)
    values = []
    for index in range(rows):
        main = sorted(int(value) for value in rng.choice(np.arange(1, 13), 4, replace=False))
        values.append(
            {
                "draw_date": (pd.Timestamp("2025-01-01") + pd.Timedelta(days=index * 7))
                .date()
                .isoformat(),
                **{f"ball_{position + 1}": value for position, value in enumerate(main)},
                "star_1": int(rng.integers(1, 5)),
            }
        )
    return pd.DataFrame(values)


def build() -> None:
    COMMON.mkdir(parents=True, exist_ok=True)
    history = synthetic_history()
    history_path = COMMON / "synthetic_history.csv"
    history.to_csv(history_path, index=False, lineterminator="\n")
    snapshot_sha256 = hashlib.sha256(history_path.read_bytes()).hexdigest()

    spec = GameSpec("synthetic-community", main_n=12, main_k=4, star_n=4, star_k=1)
    provider_names = ("uniform_random", "frequency", "unpopularity")
    summary = evaluate_forward(
        history,
        spec,
        [registry.create(name) for name in provider_names],
        budget=5,
        holdout=4,
        seed=SEED,
        jackpot=JackpotModel(jackpot=1_000.0, ticket_price=2.0, n_other_tickets=10_000.0),
    )
    result_rows = []
    for provider, metrics in summary["providers"].items():
        result_rows.append(
            {
                "benchmark_version": "1.0.0",
                "dataset_sha256": snapshot_sha256,
                "provider": provider,
                "budget": summary["budget"],
                "holdout": summary["holdout"],
                "seed": summary["seed"],
                **metrics,
            }
        )
    result_rows.append(
        {
            "benchmark_version": "1.0.0",
            "dataset_sha256": snapshot_sha256,
            "provider": "coordinated_aggregation",
            "budget": summary["budget"],
            "holdout": summary["holdout"],
            "seed": summary["seed"],
            **summary["aggregated"],
        }
    )
    pd.DataFrame(result_rows).to_csv(
        COMMON / "benchmark_results.csv", index=False, lineterminator="\n"
    )

    manifest = {
        "benchmark_name": "LottoBench Community Benchmark",
        "benchmark_version": "1.0.0",
        "lottobench_version": "0.1.0a3",
        "data_kind": "deterministic_synthetic",
        "dataset_sha256": snapshot_sha256,
        "seed": SEED,
        "game": {
            "name": spec.name,
            "main_n": spec.main_n,
            "main_k": spec.main_k,
            "auxiliary_n": spec.star_n,
            "auxiliary_k": spec.star_k,
        },
        "evaluation": {"protocol": "forward_only", "budget": 5, "holdout": 4},
        "providers": list(provider_names),
        "primary_metric": "pair_coverage",
        "secondary_metrics": [
            "number_coverage",
            "mean_jaccard_diversity",
            "unpopularity_lift",
            "expected_roi_per_ticket",
            "hit_recall",
        ],
        "claims_boundary": (
            "Synthetic software benchmark only; not evidence of prediction, positive ROI, or "
            "performance on an operated lottery."
        ),
    }
    (COMMON / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    for target in TARGETS:
        target.mkdir(parents=True, exist_ok=True)
        for name in ("synthetic_history.csv", "benchmark_results.csv", "benchmark_manifest.json"):
            shutil.copy2(COMMON / name, target / name)


if __name__ == "__main__":
    build()
