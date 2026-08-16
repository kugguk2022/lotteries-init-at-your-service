"""Expected return-per-ticket -- computed honestly, simulation-only, never a guarantee.

This module answers the user-facing question "which set of tickets has the best expected ROI?"
in the *only* way that is mathematically defensible for a fair draw:

    E[ROI per ticket] = P(win tier) x E[payout | win tier] / ticket_price - 1

The probability terms ``P(win tier)`` are FIXED by the game's combinatorics -- no strategy moves
them. The single term a strategy can influence is ``E[payout | win the shared jackpot]``, because
a pari-mutuel jackpot is *split* among winners: expected payout conditional on winning falls as the
expected number of *co-winners* rises, and co-winners are driven by how popular your combination is
(see :mod:`lotteries_core.popularity`). Choosing unpopular combinations therefore raises expected
ROI **without changing your odds** -- and, because the base game is negative-sum, this typically
*reduces the house/operator's effective edge on your ticket* rather than turning ROI positive.

Everything here is a simulation of expected value. It never implies you should play, and the
framework never buys tickets or moves money.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np

from .popularity import PopularityModel
from .protocol import GameSpec, Ticket


@dataclass(frozen=True)
class JackpotModel:
    """Minimal pari-mutuel jackpot model for expected-ROI comparison of tickets.

    Attributes
    ----------
    jackpot:
        Advertised shared jackpot (currency units).
    ticket_price:
        Price of one ticket.
    n_other_tickets:
        Estimated number of *other* tickets sold this draw (the crowd you may split with).
    """

    jackpot: float = 100_000_000.0
    ticket_price: float = 2.5
    n_other_tickets: float = 50_000_000.0

    def jackpot_match_probability(self, spec: GameSpec) -> float:
        """P(a single ticket matches the full jackpot combination)."""
        return 1.0 / spec.n_tickets()


def expected_cowinners(spec: GameSpec, popularity_share: float, n_other_tickets: float) -> float:
    """Expected number of *other* tickets that also hit the jackpot combination you hold.

    ``popularity_share`` is your combination's share of crowd selection probability relative to a
    uniform pick. If the crowd picked uniformly, expected co-winners would be
    ``n_other_tickets / n_tickets``. Popular combinations multiply that; unpopular ones shrink it.
    """
    uniform_rate = n_other_tickets / spec.n_tickets()
    return uniform_rate * popularity_share


def expected_jackpot_payout(
    spec: GameSpec,
    jackpot: JackpotModel,
    popularity_share: float,
) -> float:
    """E[payout | you hold the winning jackpot combination], accounting for expected splitting.

    Uses the standard pari-mutuel split expectation E[J / (1 + K)] approximated with a Poisson
    number of co-winners K (mean = expected_cowinners). The 1 is you; you always split with your
    own presence in the numerator's "+1".
    """
    lam = expected_cowinners(spec, popularity_share, jackpot.n_other_tickets)
    # E[1/(1+K)] for K ~ Poisson(lam) = (1 - e^{-lam}) / lam  (with limit 1 as lam -> 0).
    if lam < 1e-9:
        share = 1.0
    else:
        share = (1.0 - np.exp(-lam)) / lam
    return jackpot.jackpot * float(share)


def expected_roi_per_ticket(
    spec: GameSpec,
    jackpot: JackpotModel,
    popularity_share: float,
) -> float:
    """Jackpot-tier expected ROI per ticket for a combination with the given popularity share.

    Returns ``E[payout] * P(win) / price - 1``. This is *jackpot-tier only*; lower tiers are added
    by richer prize tables elsewhere. The value is essentially always negative (negative-sum game);
    the point of the framework is to make it *less* negative by lowering ``popularity_share``.
    """
    p_win = jackpot.jackpot_match_probability(spec)
    ev_payout = expected_jackpot_payout(spec, jackpot, popularity_share)
    return (p_win * ev_payout) / jackpot.ticket_price - 1.0


def portfolio_expected_roi(
    spec: GameSpec,
    tickets: list[Ticket],
    jackpot: JackpotModel | None = None,
    popularity: PopularityModel | None = None,
) -> dict:
    """Expected jackpot-tier ROI for a whole portfolio, plus the unpopularity lift it captures.

    ``unpopularity_lift`` is the ratio of the portfolio's expected jackpot payout to that of a
    *uniformly popular* portfolio of the same size -- i.e. how much of the shared-jackpot lever the
    portfolio actually captured. > 1 means the portfolio leans unpopular (good for conditional ROI).
    """
    if not tickets:
        return {"expected_roi_per_ticket": float("nan"), "unpopularity_lift": float("nan")}
    jackpot = jackpot or JackpotModel()
    popularity = popularity or PopularityModel()

    shares = popularity.absolute_shares(spec, tickets)  # 1.0 == average crowding
    rois = np.array(
        [expected_roi_per_ticket(spec, jackpot, float(s)) for s in shares], dtype=float
    )
    payouts = np.array(
        [expected_jackpot_payout(spec, jackpot, float(s)) for s in shares], dtype=float
    )
    baseline_payout = expected_jackpot_payout(spec, jackpot, 1.0)
    return {
        "expected_roi_per_ticket": float(rois.mean()),
        "best_ticket_roi": float(rois.max()),
        "mean_popularity_share": float(shares.mean()),
        "unpopularity_lift": float(payouts.mean() / baseline_payout) if baseline_payout else float("nan"),
    }


# --------------------------------------------------------------------------------------------
# Instant / scratch games: the honest core of the "Joan Ginther" advantage-play story.
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class InstantGamePool:
    """A finite-pool instant/scratch game described by its *published remaining* inventory.

    Unlike a draw game, a scratch game is a finite deck. Once the top prizes are claimed, the
    remaining tickets are worth strictly less -- and lotteries publish remaining-prize counts. When
    (rarely) enough top prizes remain relative to unsold tickets, the *remaining* expected value can
    exceed the ticket price. That is real, public-data advantage play; it is the defensible kernel
    of the Ginther story, not number-prediction.
    """

    ticket_price: float
    tickets_remaining: float
    # prize_tiers: list of (prize_value, count_remaining)
    prize_tiers: tuple[tuple[float, float], ...]

    def remaining_ev(self) -> float:
        """Expected value of buying one *remaining* ticket, given published remaining prizes."""
        if self.tickets_remaining <= 0:
            return 0.0
        total = sum(value * count for value, count in self.prize_tiers)
        return total / self.tickets_remaining

    def remaining_roi(self) -> float:
        """Remaining EV expressed as ROI per ticket (EV/price - 1)."""
        if self.ticket_price <= 0:
            return float("nan")
        return self.remaining_ev() / self.ticket_price - 1.0

    def is_advantage(self, margin: float = 0.0) -> bool:
        """True iff remaining ROI exceeds ``margin`` (favourable to the player right now)."""
        return self.remaining_roi() > margin


def combinations_count(n: int, k: int) -> int:
    """Thin re-export of C(n, k) for callers that only import :mod:`roi`."""
    return comb(n, k)
