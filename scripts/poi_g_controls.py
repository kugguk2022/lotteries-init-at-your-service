"""Controls that isolate *why* a POI-G shortlist scores the way it does.

A raw lift over uniform random tickets does not by itself show that pair co-occurrence carries
information. A shortlist that merely favours frequently-drawn numbers would beat uniform random too,
without any pair structure being involved. These controls hold one factor fixed at a time:

``uniform``          uniformly random legal tickets -- the null.
``frequency_matched`` tickets sampled from the per-number marginal frequencies of the POI-G
                     shortlist itself, destroying pair structure while preserving which numbers the
                     shortlist likes. A POI-G lift that survives against THIS control is evidence
                     for the pair mechanism; one that vanishes was a marginal-frequency effect.
``shuffled_history``  POI-G refitted on a history whose draw order is permuted, destroying the
                     temporal/causal signal the target is supposed to track while preserving the
                     marginal and pair statistics of the pool.
"""

from __future__ import annotations

import numpy as np


def uniform_shortlist(
    rng: np.random.Generator, size: int, main_n: int, main_k: int
) -> np.ndarray:
    """(size, main_k) uniformly random distinct-number tickets."""
    return np.argsort(rng.random((size, main_n)), axis=1)[:, :main_k] + 1


def frequency_matched_shortlist(
    rng: np.random.Generator, reference: np.ndarray, main_n: int, main_k: int
) -> np.ndarray:
    """Tickets drawn from ``reference``'s per-number marginals, with pair structure destroyed.

    ``reference`` is a (size, main_k) array of the POI-G shortlist's numbers. Sampling without
    replacement from its marginal distribution keeps "which numbers does POI-G favour" while
    discarding "which numbers does POI-G pair together".
    """
    counts = np.bincount(reference.ravel() - 1, minlength=main_n).astype(float)
    if counts.sum() <= 0:
        counts = np.ones(main_n)
    weights = counts / counts.sum()
    # Gumbel top-k: one weighted sample without replacement per row, vectorised over all rows at
    # once. A per-row Python loop here dominated the whole evaluation.
    keys = rng.gumbel(size=(len(reference), main_n)) + np.log(np.maximum(weights, 1e-12))
    return np.argpartition(-keys, main_k - 1, axis=1)[:, :main_k] + 1


def best_match(shortlist: np.ndarray, actual_main: set[int]) -> int:
    """Highest number of drawn main numbers matched by any ticket in the shortlist."""
    return int(np.isin(shortlist, list(actual_main)).sum(axis=1).max())


def match_rates(
    shortlist: np.ndarray, actual_main: set[int], thresholds: tuple[int, ...] = (3, 4, 5)
) -> dict[int, int]:
    """Per-threshold count of tickets matching at least that many drawn main numbers."""
    hits = np.isin(shortlist, list(actual_main)).sum(axis=1)
    return {threshold: int((hits >= threshold).sum()) for threshold in thresholds}
