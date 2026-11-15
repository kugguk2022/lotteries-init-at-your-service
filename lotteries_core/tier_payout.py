"""All-tier pari-mutuel expected return, driven by a crowd model fitted to real winner counts.

:mod:`lotteries_core.roi` models the jackpot tier only, with a fixed jackpot and a fixed crowd
size. That leaves most of the lever on the table: in EuroMillions *every* tier is pari-mutuel
(each tier receives a fixed share of the prize fund and splits it among that tier's winners), so
a ticket built from under-picked numbers is paid more at all thirteen tiers, including the ones
that actually happen. This module prices that.

For a ticket ``t`` and tier ``(a, b)``::

    E[prize | t lands in (a, b)] = Fund_ab * E_W[ (1 - exp(-lam_W)) / lam_W ]
    lam_W = entries * P_crowd(tier (a, b) | drawn numbers W)

where the expectation is over drawn sets ``W`` compatible with ``t`` matching exactly ``a`` mains
and ``b`` stars, and ``(1 - e^-lam) / lam`` is ``E[1 / (1 + K)]`` for ``K ~ Poisson(lam)``
co-winners. ``P(t lands in (a, b))`` is the fixed hypergeometric probability; nothing here moves it.

Everything is expected value under stated economics. It is not realised profit, and the base game
stays negative-sum: the point is to measure how much *less* negative a ticket is than a random one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .crowd_forecast import CrowdModel, hypergeometric_tier_probability
from .protocol import GameSpec, Ticket

__all__ = [
    "TierEconomics",
    "portfolio_report",
    "ticket_expected_return",
    "uniform_ticket_baseline",
]

#: EuroMillions share of the prize fund per tier, as published by the game rules (they sum to
#: 90%; the remaining 10% feeds the reserve/booster fund). The 5+2 share is 50% for the first five
#: draws of a rollover series and 42% afterwards; pass ``jackpot`` to price the advertised amount.
EUROMILLIONS_TIER_SHARES: dict[tuple[int, int], float] = {
    (5, 2): 0.50, (5, 1): 0.0261, (5, 0): 0.0061, (4, 2): 0.0019, (4, 1): 0.0035,
    (3, 2): 0.0037, (4, 0): 0.0026, (2, 2): 0.0130, (3, 1): 0.0145, (3, 0): 0.0270,
    (1, 2): 0.0327, (2, 1): 0.1030, (2, 0): 0.1659,
}


@dataclass(frozen=True)
class TierEconomics:
    """Per-draw money facts needed to price a ticket.

    ``fund_per_line`` is prize-fund money generated per entry. The EuroMillions default (1.10 EUR
    of a 2.50 EUR line) is an assumption checked against the published long-run average 2+0 prize;
    re-estimate it from prize-per-winner data with :func:`estimate_fund_per_line` before relying
    on absolute ROI levels. Differences between tickets (alpha) are far less sensitive to it.
    """

    tier_shares: dict[tuple[int, int], float]
    ticket_price: float
    fund_per_line: float
    entries: float
    jackpot: float | None = None
    jackpot_tier: tuple[int, int] | None = None
    currency: str = "EUR"

    @classmethod
    def euromillions(cls, *, entries: float, jackpot: float | None = None) -> "TierEconomics":
        return cls(
            tier_shares=dict(EUROMILLIONS_TIER_SHARES), ticket_price=2.50, fund_per_line=1.10,
            entries=float(entries), jackpot=jackpot, jackpot_tier=(5, 2),
        )

    def tier_fund(self, tier: tuple[int, int]) -> float:
        if self.jackpot is not None and tier == self.jackpot_tier:
            return float(self.jackpot)
        return float(self.tier_shares[tier] * self.fund_per_line * self.entries)


def estimate_fund_per_line(
    prize_per_winner: float, winners: float, tier_share: float, entries: float
) -> float:
    """Back out prize-fund money per entry from one tier's published prize and winner count."""
    if min(winners, tier_share, entries) <= 0:
        raise ValueError("winners, tier_share and entries must be positive")
    return float(prize_per_winner * winners / tier_share / entries)


def _expected_share(lam: np.ndarray) -> np.ndarray:
    """E[1 / (1 + K)] for K ~ Poisson(lam), stable as lam -> 0."""
    lam = np.asarray(lam, dtype=float)
    safe = np.where(lam < 1e-8, 1.0, lam)
    return np.where(lam < 1e-8, 1.0 - lam / 2.0, -np.expm1(-safe) / safe)


def _compatible_draws(
    pool_n: int, k: int, held: np.ndarray, matches: int, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample drawn sets that share exactly ``matches`` numbers with ``held`` (1-based).

    Positions are sampled by rank (which of the held numbers, which ranks of the complement), so
    two different tickets evaluated with the same seed see the same random structure. That is
    common random numbers: ticket-versus-ticket differences are far less noisy than either level.
    """
    held = np.asarray(held, dtype=int)
    others = np.setdiff1d(np.arange(1, pool_n + 1), held)
    keep = np.argsort(rng.random((n_samples, k)), axis=1)[:, :matches]
    rest = np.argsort(rng.random((n_samples, others.size)), axis=1)[:, : k - matches]
    return np.sort(np.concatenate([held[keep], others[rest]], axis=1), axis=1)


def ticket_expected_return(
    crowd: CrowdModel,
    economics: TierEconomics,
    ticket: Ticket,
    *,
    n_samples: int = 400,
    seed: int = 20261109,
) -> dict:
    """Expected payout and ROI of one ticket across every paying tier.

    Returns ``{"expected_payout", "roi", "by_tier": {(a, b): contribution}}`` where contributions
    are in currency units per ticket and sum to ``expected_payout``.
    """
    spec: GameSpec = crowd.spec
    spec.validate_ticket(ticket)
    main, star = (np.array(ticket[0], dtype=int), np.array(ticket[1], dtype=int))
    by_tier: dict[tuple[int, int], float] = {}
    for tier in economics.tier_shares:
        a, b = tier
        rng = np.random.default_rng([seed, a, b])
        exact = a == spec.main_k and b == spec.star_k
        samples = 1 if exact else n_samples
        main_w = _compatible_draws(spec.main_n, spec.main_k, main, a, samples, rng)
        if spec.star_k > 0:
            star_w = _compatible_draws(spec.star_n, spec.star_k, star, b, samples, rng)
        else:
            star_w = None
        prob = crowd.tier_probabilities(main_w, star_w)[:, a, b]
        share = _expected_share(economics.entries * prob).mean()
        hit = hypergeometric_tier_probability(spec, a, b)
        by_tier[tier] = float(hit * economics.tier_fund(tier) * share)
    payout = float(sum(by_tier.values()))
    return {
        "expected_payout": payout,
        "roi": payout / economics.ticket_price - 1.0,
        "by_tier": by_tier,
    }


def uniform_ticket_baseline(
    crowd: CrowdModel,
    economics: TierEconomics,
    *,
    n_tickets: int = 300,
    n_samples: int = 200,
    seed: int = 20261109,
) -> dict:
    """Mean expected return of uniformly random tickets under the same crowd and economics.

    This is the equal-budget null for the payout track: same crowd model, same money, tickets
    drawn with no preference at all.
    """
    spec = crowd.spec
    rng = np.random.default_rng([seed, 7])
    rois, payouts = [], []
    for _ in range(n_tickets):
        main = tuple(sorted(int(v) + 1 for v in rng.choice(spec.main_n, spec.main_k, replace=False)))
        star = (
            tuple(sorted(int(v) + 1 for v in rng.choice(spec.star_n, spec.star_k, replace=False)))
            if spec.star_k > 0 else ()
        )
        result = ticket_expected_return(
            crowd, economics, (main, star), n_samples=n_samples, seed=seed
        )
        rois.append(result["roi"])
        payouts.append(result["expected_payout"])
    rois_arr = np.array(rois)
    return {
        "mean_roi": float(rois_arr.mean()),
        "roi_p05": float(np.percentile(rois_arr, 5)),
        "roi_p95": float(np.percentile(rois_arr, 95)),
        "mean_expected_payout": float(np.mean(payouts)),
        "tickets": int(n_tickets),
    }


@dataclass
class _Report:
    rows: list[dict] = field(default_factory=list)


def portfolio_report(
    crowd: CrowdModel,
    economics: TierEconomics,
    tickets: list[Ticket],
    *,
    baseline: dict | None = None,
    n_samples: int = 400,
    seed: int = 20261109,
) -> dict:
    """Expected return of a ticket set and its alpha (percentage points) over random tickets.

    ``alpha_pp`` splits into ``alpha_jackpot_pp`` and ``alpha_lower_tiers_pp`` so the report shows
    how much of the edge comes from tiers that are actually hit at human time scales.
    """
    if not tickets:
        raise ValueError("portfolio_report needs at least one ticket")
    baseline = baseline or uniform_ticket_baseline(crowd, economics, seed=seed)
    results = [
        ticket_expected_return(crowd, economics, t, n_samples=n_samples, seed=seed)
        for t in tickets
    ]
    roi = float(np.mean([r["roi"] for r in results]))
    jackpot_tier = economics.jackpot_tier
    jackpot_part = float(
        np.mean([r["by_tier"].get(jackpot_tier, 0.0) for r in results])
    ) if jackpot_tier else 0.0
    return {
        "tickets": len(tickets),
        "mean_roi": roi,
        "null_mean_roi": baseline["mean_roi"],
        "alpha_pp": 100.0 * (roi - baseline["mean_roi"]),
        "jackpot_payout_per_ticket": jackpot_part,
        "lower_tier_payout_per_ticket": float(
            np.mean([r["expected_payout"] for r in results]) - jackpot_part
        ),
        "best_ticket_roi": float(max(r["roi"] for r in results)),
        "worst_ticket_roi": float(min(r["roi"] for r in results)),
        "currency": economics.currency,
        "claims_boundary": (
            "Modeled expected return under a crowd model and stated prize-fund economics. Not "
            "realised profit; every ticket keeps the same probability of winning each tier."
        ),
    }
