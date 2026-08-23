"""Regression guard: the forward benchmark must be byte-for-byte reproducible.

Rather than pinning brittle floating-point values (which drift across numpy/BLAS versions), this
pins the property the framework actually promises: *the same code + same data + same seed yields the
same result*. If a change makes a run non-deterministic, this fails loudly.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from lotteries_core.evaluation import evaluate_forward
from lotteries_core.protocol import GameSpec
from lotteries_core.providers import FrequencyProvider, UnpopularityProvider


def _history(n: int = 150, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        mains = sorted(rng.choice(np.arange(1, 51), size=5, replace=False))
        stars = sorted(rng.choice(np.arange(1, 13), size=2, replace=False))
        rows.append(
            {
                "draw_date": f"2021-{(i % 12) + 1:02d}-01",
                **{f"ball_{j+1}": int(mains[j]) for j in range(5)},
                **{f"star_{j+1}": int(stars[j]) for j in range(2)},
            }
        )
    return pd.DataFrame(rows)


def _run():
    return evaluate_forward(
        _history(),
        GameSpec.euromillions(),
        [FrequencyProvider(), UnpopularityProvider()],
        budget=15,
        holdout=10,
        seed=2024,
    )


def test_benchmark_is_deterministic():
    a = json.dumps(_run(), sort_keys=True)
    b = json.dumps(_run(), sort_keys=True)
    assert a == b


def test_benchmark_reports_all_levers():
    summary = _run()
    for key in ("providers", "aggregated", "headline"):
        assert key in summary
    for lever in ("pair_coverage", "unpopularity_lift", "expected_roi_per_ticket"):
        assert lever in summary["aggregated"]
