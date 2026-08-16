"""Diversity-aware, equal-budget aggregation -- the "coordinated distributed inference" step.

The framework's central research question is: *can coordinating many independent providers beat any
single provider spending the same ticket budget?* The mechanism that could make that true is here.

Given several providers' envelopes (proposals under some budget) and a **shared budget** ``B``, the
aggregator selects exactly ``B`` distinct tickets that jointly maximise a submodular objective
combining three levers we actually control:

* **provider consensus** -- tickets multiple providers rank highly (normalised, cross-provider),
* **coverage/diversity** -- marginal combinatorial reach a ticket adds to the portfolio,
* **unpopularity ROI** -- expected conditional payout from :mod:`lotteries_core.roi`.

Selection is greedy over a monotone submodular gain, which gives a well-known ``1 - 1/e`` guarantee
and, crucially, is **deterministic** given identical envelopes -- a precondition (see ``repurpose.md``)
before any networking is added. Coordination here is *pure recombination of already-proposed tickets*;
the aggregator never invents tickets, moves money, or assumes predictive power.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coverage import pair_coverage
from .envelope import InferenceEnvelope
from .popularity import PopularityModel
from .protocol import GameSpec, Ticket
from .roi import JackpotModel, expected_jackpot_payout


@dataclass(frozen=True)
class AggregationWeights:
    """Relative importance of the three levers in the greedy objective (need not sum to 1)."""

    consensus: float = 1.0
    coverage: float = 1.0
    unpopularity: float = 1.0


def _rank_scores(scores: np.ndarray) -> np.ndarray:
    """Convert arbitrary provider scores to a comparable 0..1 rank-normalised scale."""
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores
    order = scores.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(scores.size, dtype=float)
    if scores.size == 1:
        return np.ones(1)
    return ranks / (scores.size - 1)


def build_consensus(envelopes: list[InferenceEnvelope]) -> dict[Ticket, float]:
    """Cross-provider consensus score per ticket: sum of rank-normalised scores across envelopes.

    A ticket proposed strongly by several providers accrues more consensus than one favoured by a
    single provider. Scores are rank-normalised *within each provider* first, so no provider's
    arbitrary scale dominates.
    """
    consensus: dict[Ticket, float] = {}
    for env in envelopes:
        rn = _rank_scores(np.asarray(env.scores, dtype=float))
        for t, s in zip(env.tickets, rn):
            consensus[t] = consensus.get(t, 0.0) + float(s)
    return consensus


def aggregate(
    envelopes: list[InferenceEnvelope],
    spec: GameSpec,
    budget: int,
    *,
    weights: AggregationWeights | None = None,
    jackpot: JackpotModel | None = None,
    popularity: PopularityModel | None = None,
) -> list[Ticket]:
    """Select ``budget`` distinct tickets from the union of envelope proposals.

    Deterministic greedy maximisation of::

        gain(t | S) = w_consensus * consensus[t]
                    + w_coverage  * marginal_pair_coverage(t | S)
                    + w_unpopularity * normalised_unpopularity_payout(t)

    Ties are broken by a stable key so results are reproducible.
    """
    weights = weights or AggregationWeights()
    jackpot = jackpot or JackpotModel()
    popularity = popularity or PopularityModel()

    candidates: list[Ticket] = sorted(set(t for env in envelopes for t in env.tickets))
    if not candidates:
        return []
    budget = min(budget, len(candidates))

    consensus = build_consensus(envelopes)
    c_vals = np.array([consensus.get(t, 0.0) for t in candidates], dtype=float)
    c_norm = c_vals / c_vals.max() if c_vals.max() > 0 else c_vals

    # Static unpopularity payout term (higher = less crowded = better conditional ROI).
    shares = popularity.absolute_shares(spec, candidates)
    payouts = np.array(
        [expected_jackpot_payout(spec, jackpot, float(s)) for s in shares], dtype=float
    )
    u_norm = payouts / payouts.max() if payouts.max() > 0 else payouts

    total_pairs = spec.main_n * (spec.main_n - 1) // 2
    chosen: list[Ticket] = []
    chosen_idx: list[int] = []
    remaining = set(range(len(candidates)))

    def marginal_coverage(idx: int) -> float:
        if not chosen:
            base = 0.0
        else:
            base = pair_coverage(spec, chosen) * total_pairs
        after = pair_coverage(spec, chosen + [candidates[idx]]) * total_pairs
        return (after - base) / total_pairs

    while len(chosen) < budget and remaining:
        best_idx = -1
        best_gain = -np.inf
        for idx in remaining:
            gain = (
                weights.consensus * c_norm[idx]
                + weights.coverage * marginal_coverage(idx)
                + weights.unpopularity * u_norm[idx]
            )
            # stable tie-break: prefer higher consensus, then lexicographically smaller ticket
            if gain > best_gain + 1e-12 or (
                abs(gain - best_gain) <= 1e-12
                and best_idx >= 0
                and candidates[idx] < candidates[best_idx]
            ):
                best_gain = gain
                best_idx = idx
        chosen.append(candidates[best_idx])
        chosen_idx.append(best_idx)
        remaining.discard(best_idx)

    return chosen
