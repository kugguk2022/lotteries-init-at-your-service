from __future__ import annotations

import json

import numpy as np
import pandas as pd

from lotteries_core import registry
from lotteries_core.evaluation import evaluate_forward
from lotteries_core.protocol import GameSpec
from lotteries_core.roi import JackpotModel


def test_base_provider_benchmark_emits_complete_strict_json():
    """Every base-install entrant must survive one real forward-only output pass."""
    spec = GameSpec("output-contract", main_n=10, main_k=3, star_n=3, star_k=1)
    rng = np.random.default_rng(20260823)
    rows = []
    for draw in range(28):
        main = sorted(int(value) for value in rng.choice(np.arange(1, 11), 3, replace=False))
        rows.append(
            {
                "draw_date": f"2026-01-{draw + 1:02d}",
                **{f"ball_{index + 1}": value for index, value in enumerate(main)},
                "star_1": int(rng.integers(1, 4)),
            }
        )
    history = pd.DataFrame(rows)
    names = [name for name, item in registry.PROVIDERS.items() if not item.optional]
    summary = evaluate_forward(
        history,
        spec,
        [registry.create(name) for name in names],
        budget=3,
        holdout=1,
        seed=731,
        jackpot=JackpotModel(jackpot=100.0, ticket_price=2.0, n_other_tickets=100.0),
    )

    assert set(summary["providers"]) == set(names)
    required = {
        "hit_recall",
        "pair_coverage",
        "number_coverage",
        "mean_jaccard_diversity",
        "expected_roi_per_ticket",
        "unpopularity_lift",
    }
    for metrics in summary["providers"].values():
        assert required <= metrics.keys()
        assert all(np.isfinite(float(metrics[key])) for key in required)

    # Reject NaN/Infinity: those values are accepted by Python's default encoder but are not JSON.
    encoded = json.dumps(summary, sort_keys=True, allow_nan=False)
    assert json.loads(encoded)["holdout"] == 1


def test_detailed_benchmark_exposes_forward_only_agent_submissions():
    spec = GameSpec("step-contract", main_n=8, main_k=3, star_n=3, star_k=1)
    rng = np.random.default_rng(91)
    rows = []
    for draw in range(14):
        main = sorted(int(value) for value in rng.choice(np.arange(1, 9), 3, replace=False))
        rows.append(
            {
                "draw_date": f"2026-02-{draw + 1:02d}",
                **{f"ball_{index + 1}": value for index, value in enumerate(main)},
                "star_1": int(rng.integers(1, 4)),
            }
        )

    summary = evaluate_forward(
        pd.DataFrame(rows),
        spec,
        [registry.create("uniform_random"), registry.create("frequency")],
        budget=3,
        holdout=2,
        seed=17,
        jackpot=JackpotModel(jackpot=100.0, ticket_price=2.0, n_other_tickets=100.0),
        include_steps=True,
    )

    assert len(summary["steps"]) == 2
    assert summary["steps"][0]["draw_date"] == "2026-02-13"
    assert set(summary["steps"][0]["agents"]) == {
        "uniform_random",
        "frequency",
        "coordinated_aggregation",
    }
    assert len(summary["steps"][0]["agents"]["frequency"]["tickets"]) == 3
    json.dumps(summary, allow_nan=False)
