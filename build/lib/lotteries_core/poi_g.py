"""Causal POI-G candidate subsets and honest downstream portfolio diagnostics.

POI-G is a search-space reducer: it ranks legal tickets by distance from the next causal
pair-co-occurrence (G) target. A subset is not a purchased portfolio, so ROI is computed only for
an explicitly bounded selection from that subset.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .likely_set_generator import (
    GameConfig,
    build_comatrices,
    detect_columns,
    generate_sets,
    observed_poi_series,
)
from .protocol import GameSpec, Ticket
from .roi import JackpotModel, portfolio_expected_roi


@dataclass(frozen=True)
class PoiGSubset:
    """A reproducible candidate subset plus its causal provenance."""

    tickets: list[Ticket]
    scores: list[int]
    target_g: float
    requested_size: int
    universe_size: int
    history_rows: int
    window: int
    pairing: str

    @property
    def size(self) -> int:
        return len(self.tickets)

    @property
    def universe_fraction(self) -> float:
        return self.size / self.universe_size

    @property
    def reduction_factor(self) -> float:
        return self.universe_size / self.size if self.size else float("inf")

    def select(self, budget: int) -> list[Ticket]:
        """Return the first ``budget`` ranked tickets; this is the purchasable portfolio."""
        if budget < 1:
            raise ValueError("budget must be positive")
        return self.tickets[:budget]

    def modeled_portfolio_roi(
        self, spec: GameSpec, budget: int, jackpot: JackpotModel
    ) -> dict:
        """Modeled jackpot-tier ROI for a bounded selection, never for the whole shortlist."""
        metrics = portfolio_expected_roi(spec, self.select(budget), jackpot=jackpot)
        return {"selection_budget": min(budget, self.size), **metrics}


def generate_poi_g_subset(
    history: pd.DataFrame,
    spec: GameSpec,
    subset_size: int,
    *,
    window: int = 26,
    pairing: str = "cross",
) -> PoiGSubset:
    """Rank a causal POI-G subset using only the supplied, already-known history.

    This exact implementation is intended for reproducibility and moderate universes. Large
    EuroMillions subsets require enumerating the full universe and can be computationally costly.
    """
    if subset_size < 1:
        raise ValueError("subset_size must be positive")
    if pairing not in {"cross", "main", "pooled"}:
        raise ValueError("pairing must be cross, main, or pooled")
    universe = spec.n_tickets()
    subset_size = min(int(subset_size), universe)
    cfg = GameConfig(spec.name, spec.main_n, spec.main_k, spec.star_n, spec.star_k)
    main_cols, star_cols = detect_columns(history, cfg)
    if len(main_cols) != spec.main_k or len(star_cols) != spec.star_k:
        raise ValueError("history does not contain the required main/star columns")
    matrices = build_comatrices(history, cfg, main_cols, star_cols)
    poi = observed_poi_series(history, cfg, matrices, main_cols, star_cols, pairing)
    target = float(pd.Series(poi[-window:]).mean()) if len(poi) else 0.0
    _level, ranked = generate_sets(
        cfg, matrices, target, pairing=pairing, max_out=0, top_n=subset_size
    )
    tickets = [(mains, stars) for mains, stars, _score in ranked]
    scores = [int(score) for _mains, _stars, score in ranked]
    return PoiGSubset(
        tickets=tickets,
        scores=scores,
        target_g=target,
        requested_size=subset_size,
        universe_size=universe,
        history_rows=len(history),
        window=int(window),
        pairing=pairing,
    )
