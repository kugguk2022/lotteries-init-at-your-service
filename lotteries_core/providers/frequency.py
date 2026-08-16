"""Frequency-weighted baseline provider (preserves the repo's original baseline behaviour).

This is the honest baseline: it samples tickets with numbers weighted by their smoothed historical
frequency. On a fair draw this has *no* predictive edge over uniform sampling (and the repo's own
walk-forward test only claims it beats uniform on a deliberately *biased* synthetic dataset). It is
retained as the reference every other provider and the aggregator must beat on coverage / expected
ROI under equal budget.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..protocol import GameSpec, InferenceProvider, ProviderResult


class FrequencyProvider(InferenceProvider):
    name = "frequency"
    description = (
        "Smoothed historical-frequency weighted sampling. No predictive edge on a fair draw; "
        "kept as the reference baseline."
    )

    def __init__(self, smoothing: float = 1.0) -> None:
        self.smoothing = float(smoothing)
        self._main_probs: np.ndarray | None = None
        self._star_probs: np.ndarray | None = None
        self._spec: GameSpec | None = None

    def fit(self, history: pd.DataFrame, spec: GameSpec | None = None) -> "FrequencyProvider":
        if spec is not None:
            self._spec = spec
            main_cols = [c for c in history.columns if c.lower().startswith(("ball_", "n"))][: spec.main_k]
            self._main_probs = _smoothed_counts(history, main_cols, spec.main_n, self.smoothing)
            if spec.star_k > 0:
                star_cols = [c for c in history.columns if c.lower().startswith(("star_", "dream"))][
                    : spec.star_k
                ]
                self._star_probs = _smoothed_counts(history, star_cols, spec.star_n, self.smoothing)
        return self

    def propose(self, spec: GameSpec, budget: int, rng: np.random.Generator) -> ProviderResult:
        main_probs = self._main_probs
        star_probs = self._star_probs
        tickets = self._sample_distinct_tickets(spec, budget, rng, main_probs, star_probs)
        # Score = product of number frequencies (provider-internal preference).
        scores = np.zeros(len(tickets), dtype=float)
        mp = main_probs if main_probs is not None else np.ones(spec.main_n) / spec.main_n
        for i, (main, _star) in enumerate(tickets):
            scores[i] = float(np.sum(np.log(mp[[v - 1 for v in main]] + 1e-12)))
        return ProviderResult(tickets=tickets, scores=scores, diagnostics={"smoothing": self.smoothing})


def _smoothed_counts(
    history: pd.DataFrame, cols: list[str], pop: int, smoothing: float
) -> np.ndarray:
    counts = np.zeros(pop, dtype=float)
    for c in cols:
        vals = pd.to_numeric(history[c], errors="coerce").dropna().astype(int)
        for v in vals:
            if 1 <= v <= pop:
                counts[v - 1] += 1.0
    probs = counts + smoothing
    return probs / probs.sum()
