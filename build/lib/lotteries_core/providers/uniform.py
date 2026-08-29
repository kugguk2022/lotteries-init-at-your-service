"""Canonical fair-draw null provider."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..protocol import GameSpec, InferenceProvider, ProviderResult


class UniformRandomProvider(InferenceProvider):
    """Sample distinct legal tickets uniformly using only the supplied reproducible RNG."""

    name = "uniform_random"
    description = (
        "Seeded uniform sampling over legal tickets. This is the canonical fair-draw null: it "
        "uses no history and gives every legal combination equal probability."
    )

    def fit(
        self, history: pd.DataFrame, spec: GameSpec | None = None
    ) -> UniformRandomProvider:
        del history, spec
        return self

    def propose(self, spec: GameSpec, budget: int, rng: np.random.Generator) -> ProviderResult:
        tickets = self._sample_distinct_tickets(spec, budget, rng)
        return ProviderResult(
            tickets=tickets,
            scores=np.zeros(len(tickets), dtype=float),
            diagnostics={"null_model": "uniform_over_legal_tickets", "uses_history": False},
        )
