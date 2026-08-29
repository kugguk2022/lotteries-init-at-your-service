"""Build redistribution-safe Hugging Face and Kaggle community benchmark artifacts.

Only deterministic synthetic draws and derived metrics are emitted. Operator-sourced histories,
prospective ledgers, tickets, and payout records are deliberately excluded.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from math import comb, log
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
BENCHMARK_VERSION = "3.0.0"
# A 48-row history with a 4-draw holdout could not resolve containment at all: a shortlist with
# exactly random containment still shows zero hits 96% of the time at size 20. Five years of weekly
# draws with a 104-draw POI-G holdout makes the containment columns mean something.
HISTORY_ROWS = 260
PROVIDER_HOLDOUT = 52
POI_HOLDOUT = 104


def _normalised_entropy(slots: Counter[int], pool: int) -> float:
    """Shannon entropy of how shortlist slots spread over the auxiliary pool, 1.0 = uniform.

    Low values mean the shortlist barely ranks on the auxiliary axis, which is exactly the
    condition under which full-ticket containment understates a main-pool reducer.
    """
    total = sum(slots.values())
    if total <= 0 or pool < 2:
        return float("nan")
    shares = [slots.get(value, 0) / total for value in range(1, pool + 1)]
    entropy = -sum(share * log(share) for share in shares if share > 0)
    return entropy / log(pool)


def _min_share(slots: Counter[int], pool: int) -> float:
    """Share of shortlist slots taken by the least-represented auxiliary number (uniform = 1/pool)."""
    total = sum(slots.values())
    if total <= 0:
        return float("nan")
    return min(slots.get(value, 0) / total for value in range(1, pool + 1))


def synthetic_history(rows: int = HISTORY_ROWS) -> pd.DataFrame:
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
        holdout=PROVIDER_HOLDOUT,
        seed=SEED,
        jackpot=JackpotModel(jackpot=1_000.0, ticket_price=2.0, n_other_tickets=10_000.0),
    )
    result_rows = []
    for provider, metrics in summary["providers"].items():
        result_rows.append(
            {
                "benchmark_version": BENCHMARK_VERSION,
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
            "benchmark_version": BENCHMARK_VERSION,
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
    #
    # Containment is scored on TWO axes, because scoring only the full ticket measured the wrong
    # thing. POI-G ranks by pair co-occurrence, which is dominated by the main pool; ties in G are
    # broken by enumeration order, which is star-major, so a small shortlist is close to
    # star-degenerate (at size 20 one star can take under 2% of the slots). Full-ticket containment
    # therefore charges the method for an axis it barely ranks on. Main-only containment is reported
    # against the main-only universe C(main_n, main_k) -- using the full-ticket denominator for a
    # main-only hit would compare against the wrong baseline.
    main_universe = comb(spec.main_n, spec.main_k)
    poi_rows = []
    for subset_size in POI_SUBSET_SIZES:
        full_hits: list[float] = []
        main_hits: list[float] = []
        distinct_mains: list[int] = []
        star_slots: Counter[int] = Counter()
        roi_values = []
        for index in range(len(history) - POI_HOLDOUT, len(history)):
            subset = generate_poi_g_subset(history.iloc[:index], spec, subset_size)
            row = history.iloc[index]
            actual_main = tuple(sorted(int(row[f"ball_{i}"]) for i in range(1, spec.main_k + 1)))
            actual = (actual_main, (int(row["star_1"]),))
            shortlist_mains = {ticket[0] for ticket in subset.tickets}
            full_hits.append(float(actual in set(subset.tickets)))
            main_hits.append(float(actual_main in shortlist_mains))
            distinct_mains.append(len(shortlist_mains))
            star_slots.update(star for _mains, stars in subset.tickets for star in stars)
            roi_values.append(
                subset.modeled_portfolio_roi(
                    spec,
                    budget=5,
                    jackpot=JackpotModel(
                        jackpot=1_000.0, ticket_price=2.0, n_other_tickets=10_000.0
                    ),
                )["expected_roi_per_ticket"]
            )

        full_random = subset_size / spec.n_tickets()
        full_rate = float(np.mean(full_hits))
        mean_distinct_mains = float(np.mean(distinct_mains))
        main_random = mean_distinct_mains / main_universe
        main_rate = float(np.mean(main_hits))
        poi_rows.append(
            {
                "benchmark_version": BENCHMARK_VERSION,
                "dataset_sha256": snapshot_sha256,
                "method": "poi_g_causal",
                "subset_size": subset_size,
                "holdout": POI_HOLDOUT,
                "universe_size": spec.n_tickets(),
                "universe_fraction": full_random,
                "reduction_factor": spec.n_tickets() / subset_size,
                "contained_draws_full": int(sum(full_hits)),
                "containment_rate_full": full_rate,
                "random_expected_containment_rate_full": full_random,
                "containment_lift_full": full_rate / full_random if full_random else float("nan"),
                "main_universe_size": main_universe,
                "mean_distinct_main_sets": mean_distinct_mains,
                "contained_draws_main": int(sum(main_hits)),
                "containment_rate_main": main_rate,
                "random_expected_containment_rate_main": main_random,
                "containment_lift_main": main_rate / main_random if main_random else float("nan"),
                "star_share_entropy": _normalised_entropy(star_slots, spec.star_n),
                "min_star_share": _min_share(star_slots, spec.star_n),
                "selection_budget": 5,
                "modeled_expected_roi_per_selected_ticket": float(np.mean(roi_values)),
            }
        )
    pd.DataFrame(poi_rows).to_csv(
        COMMON / "poi_g_subset_results.csv", index=False, lineterminator="\n"
    )

    manifest = {
        "benchmark_name": "LottoBench Community Benchmark",
        "benchmark_version": BENCHMARK_VERSION,
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
        "evaluation": {"protocol": "forward_only", "budget": 5, "holdout": PROVIDER_HOLDOUT},
        "poi_g_evaluation": {
            "protocol": "forward_only_candidate_subset",
            "subset_sizes": list(POI_SUBSET_SIZES),
            "holdout": POI_HOLDOUT,
            "selection_budget_for_modeled_roi": 5,
            "primary_metric": "containment_lift_main",
            "secondary_metrics": [
                "containment_lift_full",
                "star_share_entropy",
                "modeled_expected_roi_per_selected_ticket",
            ],
            "metric_note": (
                "containment_lift_main scores the main-pool combination against the main-only "
                "universe C(main_n, main_k); containment_lift_full scores the whole ticket against "
                "the full universe. They use different denominators on purpose. The full-ticket "
                "figure understates a main-pool reducer whenever star_share_entropy is low, "
                "because POI-G barely ranks on the auxiliary axis at small subset sizes."
            ),
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
