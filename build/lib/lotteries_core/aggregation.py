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
from itertools import combinations

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


def _best_single_portfolio(
    envelopes: list[InferenceEnvelope], spec: GameSpec, budget: int
) -> tuple[list[Ticket] | None, float]:
    """Highest-pair-coverage single-provider portfolio at the identical budget.

    Only envelopes that can actually field ``budget`` tickets are eligible, so the comparison the
    floor makes is equal-budget by construction. Ties keep the earliest envelope, for determinism.
    """
    best_tickets: list[Ticket] | None = None
    best_cov = -1.0
    for env in envelopes:
        tickets = list(env.tickets)[:budget]
        if len(tickets) < budget:
            continue
        cov = pair_coverage(spec, tickets)
        if cov > best_cov:
            best_cov, best_tickets = cov, tickets
    return best_tickets, best_cov


def aggregate(
    envelopes: list[InferenceEnvelope],
    spec: GameSpec,
    budget: int,
    *,
    weights: AggregationWeights | None = None,
    jackpot: JackpotModel | None = None,
    popularity: PopularityModel | None = None,
    coverage_floor: bool = True,
) -> list[Ticket]:
    """Select ``budget`` distinct tickets from the union of envelope proposals.

    Deterministic greedy maximisation of::

        gain(t | S) = w_consensus * consensus[t]
                    + w_coverage  * new_pair_fraction(t | S)
                    + w_unpopularity * normalised_unpopularity_payout(t)

    Ties are broken by a stable key so results are reproducible.

    ``coverage_floor`` enforces the framework's headline promise: coordination must not *lose*
    combinatorial reach against a single provider spending the same budget. If the greedy blend ends
    up below the best single-provider portfolio on pair coverage, that portfolio is returned instead.
    The floor is a backstop, not the mechanism -- the objective above is what should normally win --
    so a run where it fires often is a signal that the weights need attention.
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

    chosen: list[Ticket] = []
    chosen_idx: list[int] = []
    remaining = set(range(len(candidates)))

    # Marginal reach as the fraction of *this ticket's own* pairs that are new, so the term lives on
    # 0..1 like the consensus and unpopularity terms. Dividing the new-pair count by the game's total
    # pair universe instead (as this did originally) caps the term near 0.008 for a 5-of-50 game,
    # which silently reduced the coverage lever to roughly a hundredth of the weight it was given.
    ticket_pairs: list[frozenset[tuple[int, int]]] = [
        frozenset(combinations(t[0], 2)) for t in candidates
    ]
    covered_pairs: set[tuple[int, int]] = set()

    def marginal_coverage(idx: int) -> float:
        pairs = ticket_pairs[idx]
        if not pairs:
            return 0.0
        return len(pairs - covered_pairs) / len(pairs)

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
        covered_pairs |= ticket_pairs[best_idx]
        remaining.discard(best_idx)

    if coverage_floor:
        floor_tickets, floor_cov = _best_single_portfolio(envelopes, spec, budget)
        if floor_tickets is not None and pair_coverage(spec, chosen) < floor_cov - 1e-12:
            return floor_tickets

    return chosen
