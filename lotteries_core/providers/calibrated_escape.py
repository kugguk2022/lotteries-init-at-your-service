"""Calibrated Crowd Escape: low-crowd tickets chosen from a model fitted to real winner counts.

The original :class:`~lotteries_core.providers.unpopularity.UnpopularityProvider` avoids numbers
that a hand-set prior says are popular. This provider does the same job with a
:class:`~lotteries_core.crowd_forecast.CrowdModel` that has been scored out-of-sample against
published winner counts, and it uses the star pool as well as the main pool.

It is deliberately not auto-registered in :mod:`lotteries_core.registry`: it needs a fitted crowd
model, which the frozen draw-only history cannot supply. Construct it with a model, or with
:meth:`CalibratedEscapeProvider.from_winner_history`.
"""

from __future__ import annotations

import numpy as np

from ..crowd_forecast import CrowdModel, WinnerHistory, fit_crowd_model
from ..popularity import PopularityModel
from ..protocol import GameSpec, InferenceProvider, ProviderResult, Ticket


class CalibratedEscapeProvider(InferenceProvider):
    name = "calibrated_escape"
    description = (
        "Prefers combinations a winner-count-calibrated crowd model says few people hold, on both "
        "main numbers and stars. Does not change odds; makes no positive-ROI claim."
    )

    def __init__(
        self,
        crowd: CrowdModel,
        *,
        oversample: int = 400,
        max_shared_main: int = 2,
        temperature: float = 6.0,
    ) -> None:
        self.crowd = crowd
        self.oversample = int(oversample)
        self.max_shared_main = int(max_shared_main)
        self.temperature = float(temperature)
        self._patterns = PopularityModel()

    @classmethod
    def from_winner_history(cls, history: WinnerHistory, **kwargs) -> "CalibratedEscapeProvider":
        return cls(fit_crowd_model(history), **kwargs)

    def propose(self, spec: GameSpec, budget: int, rng: np.random.Generator) -> ProviderResult:
        if spec.name != self.crowd.spec.name:
            raise ValueError(f"crowd model is for {self.crowd.spec.name!r}, not {spec.name!r}")
        main_p = np.exp(-self.temperature * self.crowd.theta_main)
        star_p = np.exp(-self.temperature * self.crowd.theta_star) if spec.star_k > 0 else None
        pool_size = min(max(budget * self.oversample, budget), spec.n_tickets())
        pool: list[Ticket] = self._sample_distinct_tickets(
            spec, pool_size, rng, main_probs=main_p, star_probs=star_p
        )
        # Sequences and other play-slip patterns are crowded for reasons a per-number model
        # cannot see, so they are excluded outright rather than scored.
        pool = [t for t in pool if self._patterns._pattern_multiplier(t[0]) == 1.0]
        scores = np.array([self.crowd.ticket_log_crowding(*t) for t in pool], dtype=float)
        chosen: list[int] = []
        for i in np.argsort(scores, kind="stable"):
            main = set(pool[i][0])
            if all(len(main & set(pool[j][0])) <= self.max_shared_main for j in chosen):
                chosen.append(int(i))
            if len(chosen) == budget:
                break
        tickets = [pool[i] for i in chosen]
        return ProviderResult(
            tickets=tickets,
            scores=-scores[chosen] if chosen else np.array([]),
            diagnostics={
                "pool_size": len(pool),
                "max_shared_main": self.max_shared_main,
                "mean_log_crowding": float(scores[chosen].mean()) if chosen else float("nan"),
                "crowd_model": self.crowd.name,
                "crowd_training_draws": self.crowd.diagnostics.get("training_draws"),
            },
        )
