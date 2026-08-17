"""Combinatorial coverage and diversity metrics -- treated as first-class objectives.

A fixed ticket *budget* buys a fixed number of tickets. Two portfolios of the same size can be
very different: one may pile onto near-identical combinations, another may spread across the
number space. Because we cannot predict the draw, *how well a portfolio covers the space* is one
of the few things genuinely under our control, and it is the quantity a coordinated multi-provider
run is trying to improve versus any single provider spending the same budget.

Metrics here are deterministic and cheap:

* :func:`number_coverage` -- fraction of the main pool touched by at least one ticket.
* :func:`pair_coverage` -- fraction of all number *pairs* covered (combinatorial reach).
* :func:`mean_jaccard_diversity` -- 1 minus average pairwise Jaccard overlap between tickets.
* :func:`portfolio_entropy` -- Shannon entropy of the number-usage distribution (spread).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .protocol import GameSpec, Ticket


def _main_sets(tickets: list[Ticket]) -> list[set[int]]:
    return [set(t[0]) for t in tickets]


def number_coverage(spec: GameSpec, tickets: list[Ticket]) -> float:
    """Fraction of the main number pool covered by at least one ticket (0..1)."""
    if not tickets:
        return 0.0
    used: set[int] = set()
    for s in _main_sets(tickets):
        used |= s
    return len(used) / spec.main_n


def pair_coverage(spec: GameSpec, tickets: list[Ticket]) -> float:
    """Fraction of all C(main_n, 2) number pairs covered by at least one ticket (0..1)."""
    if not tickets:
        return 0.0
    total_pairs = spec.main_n * (spec.main_n - 1) // 2
    seen: set[tuple[int, int]] = set()
    for s in _main_sets(tickets):
        for a, b in combinations(sorted(s), 2):
            seen.add((a, b))
    return len(seen) / total_pairs


def mean_jaccard_diversity(tickets: list[Ticket]) -> float:
    """1 - mean pairwise Jaccard similarity of the main parts (1 = all disjoint, 0 = identical)."""
    sets = _main_sets(tickets)
    m = len(sets)
    if m < 2:
        return 1.0
    sims = []
    for i in range(m):
        for j in range(i + 1, m):
            inter = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j])
            sims.append(inter / union if union else 0.0)
    return 1.0 - float(np.mean(sims))


def portfolio_entropy(spec: GameSpec, tickets: list[Ticket]) -> float:
    """Normalised Shannon entropy (0..1) of how often each main number is used across the portfolio.

    1.0 means every number is used equally often (maximal spread); low values mean the portfolio
    leans on a few numbers.
    """
    if not tickets:
        return 0.0
    counts = np.zeros(spec.main_n, dtype=float)
    for s in _main_sets(tickets):
        for v in s:
            counts[v - 1] += 1.0
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    nz = p[p > 0]
    h = -np.sum(nz * np.log(nz))
    return float(h / np.log(spec.main_n))


def coverage_report(spec: GameSpec, tickets: list[Ticket]) -> dict:
    """A bundle of all coverage/diversity metrics for a ticket portfolio."""
    return {
        "n_tickets": len(tickets),
        "n_distinct_tickets": len(set(tickets)),
        "number_coverage": number_coverage(spec, tickets),
        "pair_coverage": pair_coverage(spec, tickets),
        "mean_jaccard_diversity": mean_jaccard_diversity(tickets),
        "portfolio_entropy": portfolio_entropy(spec, tickets),
    }
