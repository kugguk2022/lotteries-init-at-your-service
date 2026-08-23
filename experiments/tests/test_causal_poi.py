"""Tests for the causal (no-look-ahead) POI feature and legacy-vs-causal contrast."""

from __future__ import annotations

import numpy as np

from euromillions_agent.phase2_sobol import (
    build_causal_pair_features,
    build_pair_features,
)


def _draws(n: int = 60, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array(
        [sorted(rng.choice(np.arange(1, 51), size=5, replace=False)) for _ in range(n)], dtype=int
    )


def test_causal_poi_has_no_lookahead():
    """The first causal POI must be 0: there is no history before the first draw."""
    draws = _draws()
    causal = build_causal_pair_features(draws, main_n=50)
    assert causal.poi[0] == 0.0
    # Legacy with include_current=True cannot be zero on the first draw (counts itself).
    legacy = build_pair_features(draws, main_n=50, include_current=True)
    assert legacy.poi[0] > 0.0


def test_causal_poi_prefix_invariance():
    """A causal feature at time t must not change when future draws are appended (no leakage)."""
    draws = _draws(n=60)
    full = build_causal_pair_features(draws, main_n=50)
    prefix = build_causal_pair_features(draws[:40], main_n=50)
    # POI values for the first 40 draws are identical whether or not draws 41..60 exist.
    assert np.allclose(full.poi[:40], prefix.poi[:40])


def test_causal_baseline_is_not_euler_phi():
    """Causal g is the draw index, not Euler-phi; the two differ for non-trivial length."""
    draws = _draws()
    causal = build_causal_pair_features(draws, main_n=50)
    legacy = build_pair_features(draws, main_n=50)
    assert np.array_equal(causal.g, np.arange(1, len(draws) + 1))
    assert not np.array_equal(causal.g, legacy.g)
