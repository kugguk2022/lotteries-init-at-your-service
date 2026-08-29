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
from lotteries_core.poi_g import generate_poi_g_subset
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
POI_SUBSET_SIZES = (20, 100, 500)


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
                "benchmark_version": "2.0.0",
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
            "benchmark_version": "2.0.0",
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

    # POI-G is evaluated as a candidate-set reducer, not disguised as a five-ticket provider.
    # Each holdout subset is built strictly from earlier rows. ROI belongs only to the fixed-budget
    # downstream selection, while containment belongs to the full candidate subset.
    poi_rows = []
    holdout = 4
    for subset_size in POI_SUBSET_SIZES:
        containments = []
        roi_values = []
        for index in range(len(history) - holdout, len(history)):
            subset = generate_poi_g_subset(history.iloc[:index], spec, subset_size)
            row = history.iloc[index]
            actual = (
                tuple(sorted(int(row[f"ball_{i}"]) for i in range(1, spec.main_k + 1))),
                (int(row["star_1"]),),
            )
            containments.append(float(actual in set(subset.tickets)))
            roi_values.append(
                subset.modeled_portfolio_roi(
                    spec,
                    budget=5,
                    jackpot=JackpotModel(
                        jackpot=1_000.0, ticket_price=2.0, n_other_tickets=10_000.0
                    ),
                )["expected_roi_per_ticket"]
            )
        random_rate = subset_size / spec.n_tickets()
        observed_rate = float(np.mean(containments))
        poi_rows.append(
            {
                "benchmark_version": "2.0.0",
                "dataset_sha256": snapshot_sha256,
                "method": "poi_g_causal",
                "subset_size": subset_size,
                "universe_size": spec.n_tickets(),
                "universe_fraction": random_rate,
                "reduction_factor": spec.n_tickets() / subset_size,
                "holdout": holdout,
                "contained_draws": int(sum(containments)),
                "containment_rate": observed_rate,
                "random_expected_containment_rate": random_rate,
                "containment_lift": observed_rate / random_rate if random_rate else float("nan"),
                "selection_budget": 5,
                "modeled_expected_roi_per_selected_ticket": float(np.mean(roi_values)),
            }
        )
    pd.DataFrame(poi_rows).to_csv(
        COMMON / "poi_g_subset_results.csv", index=False, lineterminator="\n"
    )

    manifest = {
        "benchmark_name": "LottoBench Community Benchmark",
        "benchmark_version": "2.0.0",
        "lottobench_version": "0.1.0a4",
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
        "poi_g_evaluation": {
            "protocol": "forward_only_candidate_subset",
            "subset_sizes": list(POI_SUBSET_SIZES),
            "selection_budget_for_modeled_roi": 5,
            "primary_metric": "containment_lift_over_equal_size_random",
        },
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
        for name in (
            "synthetic_history.csv",
            "benchmark_results.csv",
            "poi_g_subset_results.csv",
            "benchmark_manifest.json",
        ):
            shutil.copy2(COMMON / name, target / name)


if __name__ == "__main__":
    build()
