"""Unpopularity provider -- the honest "best expected ROI per user" strategy.

This provider directly implements the only mathematically valid draw-game lever: it prefers legal
tickets that the *crowd* is least likely to pick, so that -- conditional on the (unchanged, tiny)
event of winning a shared jackpot -- the expected payout is split among fewer people. It does not
improve the odds of winning and makes no claim of positive ROI; it maximises expected
*return-per-ticket given the fixed odds*, which is the defensible reading of the user's goal.

Mechanism: oversample a pool of candidate tickets, score each by expected jackpot payout from
:mod:`lotteries_core.roi` (which folds in the popularity model), and keep the least-crowded
``budget`` tickets while enforcing basic spread so the portfolio is not all near-duplicates.
"""

from __future__ import annotations

import numpy as np

from ..popularity import PopularityModel
from ..protocol import GameSpec, InferenceProvider, ProviderResult, Ticket
from ..roi import JackpotModel, expected_jackpot_payout


class UnpopularityProvider(InferenceProvider):
    name = "unpopularity"
    description = (
        "Prefers combinations the crowd avoids, maximising expected payout conditional on winning "
        "a shared jackpot. Does not change odds; makes no positive-ROI claim."
    )

    def __init__(
        self,
        popularity: PopularityModel | None = None,
        jackpot: JackpotModel | None = None,
        oversample: int = 40,
    ) -> None:
        self.popularity = popularity or PopularityModel()
        self.jackpot = jackpot or JackpotModel()
        self.oversample = int(oversample)

    def propose(self, spec: GameSpec, budget: int, rng: np.random.Generator) -> ProviderResult:
        # Bias sampling *away* from popular numbers by using inverse popularity as the pick weight.
        num_w = self.popularity.number_weights(spec)
        inv_main = 1.0 / (num_w + 1e-9)
        pool_size = min(max(budget * self.oversample, budget), spec.n_tickets())
        pool: list[Ticket] = self._sample_distinct_tickets(
            spec, pool_size, rng, main_probs=inv_main, star_probs=None
        )
        if not pool:
            return ProviderResult(tickets=[], scores=np.array([]), diagnostics={})

        shares = self.popularity.absolute_shares(spec, pool)
        payouts = np.array(
            [expected_jackpot_payout(spec, self.jackpot, float(s)) for s in shares], dtype=float
        )
        # Keep the least-crowded (highest expected conditional payout) tickets.
        order = np.argsort(-payouts)
        keep = order[:budget]
        tickets = [pool[i] for i in keep]
        scores = payouts[keep]
        return ProviderResult(
            tickets=tickets,
            scores=scores,
            diagnostics={
                "oversample": self.oversample,
                "pool_size": len(pool),
                "mean_popularity_share_kept": float(shares[keep].mean()),
            },
        )
