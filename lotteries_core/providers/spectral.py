"""Markov / Perron-Frobenius ("PageRank") ranking over the draw co-occurrence graph.

Mechanism
---------
Treat the numbers of a game as nodes of a weighted graph whose edge ``W[i, j]`` counts how often
``i`` and ``j`` were drawn together. Column-normalise ``W`` into a transition matrix ``P`` (dangling
nodes -- numbers never drawn -- teleport uniformly), damp it exactly as Google's original PageRank
does::

    M = d * P + (1 - d) / n * 11^T

``M`` is strictly positive, hence primitive, so Perron-Frobenius guarantees a **unique** positive
stationary vector ``pi`` with ``M pi = pi``, and power iteration converges geometrically at rate
``|lambda_2| <= d``. ``pi`` is the long-run visit frequency of a random walker on the co-occurrence
graph: the "importance" score the user knows from search ranking.

Why this provider ships its own falsification metric
----------------------------------------------------
On a *fair* draw every pair is equally likely, so ``E[W]`` is a constant matrix, ``P`` is the uniform
chain, and ``pi`` collapses to ``1/n``. Any structure you see in ``pi`` is therefore finite-sample
noise plus whatever real bias the machine has. The provider reports ``tv_from_uniform`` -- the total
variation distance ``0.5 * sum |pi_i - 1/n|`` -- and :func:`null_tv_band` computes the distribution of
that same statistic over simulated *fair* histories of identical length. If the observed value sits
inside the null band, the ranking carries no information beyond sampling noise. That is the honest,
quantitative reason to promote or demote this method, and it is the reason it keeps losing: damping
pulls ``pi`` toward uniform, and a fair generator supplies nothing for it to pull away from.

Orientation
-----------
``affinity`` prefers high-``pi`` numbers (the classic reading: "well-connected" numbers). ``contrarian``
reverses the order and prefers low-``pi`` numbers. Contrarian is not a prediction claim -- it feeds the
only defensible lever in the repo (crowd-avoidance / jackpot-sharing), because the crowd's own picking
heuristics correlate with the same "popular, well-connected" statistics the walk rewards.

Ticket construction is a deterministic inverse-CDF sweep of a Kronecker (golden-ratio) low-discrepancy
sequence over ``pi``. It is shared between both orientations and degenerates to a well-spread uniform
portfolio when ``pi`` is uniform, so the benchmark compares the *ranking signal*, not two different
samplers. No RNG is consumed: results are byte-reproducible from the history alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..likely_set_generator import GameConfig, build_comatrices, detect_columns
from ..protocol import GameSpec, InferenceProvider, ProviderResult, Ticket

#: frac(1/phi) -- the additive recurrence u_{k+1} = frac(u_k + GOLDEN) is the 1-D Kronecker sequence
#: with the lowest possible discrepancy, so consecutive tickets sweep the CDF evenly.
GOLDEN = 0.6180339887498949

#: ``uniform`` is the ablation control, not a strategy: it runs the identical Kronecker sampler with
#: the stationary vector discarded. Any metric difference between it and the other two orientations is
#: attributable to the PageRank signal; any metric the three share is attributable to the sampler.
ORIENTATIONS = ("affinity", "contrarian", "uniform")


@dataclass(frozen=True)
class StationaryRank:
    """Result of the power iteration: the Perron vector plus its convergence/strength diagnostics."""

    pi: np.ndarray
    iterations: int
    residual: float
    tv_from_uniform: float
    total_edge_weight: float


def stationary_distribution(
    W: np.ndarray, *, damping: float = 0.85, tol: float = 1e-12, max_iter: int = 1000
) -> StationaryRank:
    """Damped PageRank stationary vector of a symmetric non-negative co-occurrence matrix.

    ``W`` is used as an undirected weighted adjacency (its diagonal is ignored). Columns with no mass
    are treated as dangling and redistribute uniformly, which keeps ``M`` stochastic and primitive.
    """
    A = np.array(W, dtype=float, copy=True)
    n = A.shape[0]
    if n == 0 or A.shape[0] != A.shape[1]:
        raise ValueError(f"W must be a non-empty square matrix, got shape {A.shape}")
    if not (0.0 < damping < 1.0):
        raise ValueError(f"damping must lie strictly inside (0, 1), got {damping}")
    if np.any(A < 0):
        raise ValueError("co-occurrence weights must be non-negative")
    np.fill_diagonal(A, 0.0)

    col = A.sum(axis=0)
    live = col > 0
    P = np.empty_like(A)
    P[:, live] = A[:, live] / col[live]
    P[:, ~live] = 1.0 / n  # dangling numbers (never drawn) teleport uniformly

    pi = np.full(n, 1.0 / n)
    residual = float("inf")
    iterations = 0
    for iterations in range(1, max_iter + 1):
        nxt = damping * (P @ pi) + (1.0 - damping) / n
        nxt /= nxt.sum()
        residual = float(np.abs(nxt - pi).sum())
        pi = nxt
        if residual <= tol:
            break

    return StationaryRank(
        pi=pi,
        iterations=iterations,
        residual=residual,
        tv_from_uniform=0.5 * float(np.abs(pi - 1.0 / n).sum()),
        total_edge_weight=float(col.sum() / 2.0),
    )


def null_tv_band(
    spec: GameSpec,
    n_draws: int,
    *,
    replicates: int = 64,
    damping: float = 0.85,
    seed: int = 0,
    quantiles: tuple[float, ...] = (0.5, 0.95, 0.99),
) -> dict:
    """Distribution of ``tv_from_uniform`` under a *fair* generator, for calibration.

    Simulates ``replicates`` fair histories of ``n_draws`` draws, runs the same power iteration on
    each, and returns the quantiles of the resulting total-variation distances. An observed TV below
    the 95th percentile is indistinguishable from "the machine is fair and the walk learned nothing".
    """
    rng = np.random.default_rng(seed)
    tvs = np.empty(replicates, dtype=float)
    for r in range(replicates):
        W = np.zeros((spec.main_n, spec.main_n), dtype=float)
        for _ in range(n_draws):
            drawn = rng.choice(spec.main_n, size=spec.main_k, replace=False)
            W[np.ix_(drawn, drawn)] += 1.0
        np.fill_diagonal(W, 0.0)
        tvs[r] = stationary_distribution(W, damping=damping).tv_from_uniform
    return {
        "replicates": int(replicates),
        "n_draws": int(n_draws),
        "mean": float(tvs.mean()),
        **{f"q{int(q * 100)}": float(np.quantile(tvs, q)) for q in quantiles},
    }


def _orient(pi: np.ndarray, orientation: str) -> np.ndarray:
    """Turn a stationary vector into a sampling weight, preserving spread but possibly the order."""
    if orientation == "affinity":
        w = pi.astype(float)
    elif orientation == "contrarian":
        # max+min-pi is strictly positive (>= min(pi) > 0), order-reversing, and keeps the same range.
        w = float(pi.max() + pi.min()) - pi.astype(float)
    elif orientation == "uniform":
        w = np.full(pi.shape[0], 1.0 / pi.shape[0])  # ablation: same sampler, no spectral signal
    else:
        raise ValueError(f"orientation must be one of {ORIENTATIONS}, got {orientation!r}")
    total = w.sum()
    return w / total if total > 0 else np.full(pi.shape[0], 1.0 / pi.shape[0])


def _pick(cdf: np.ndarray, u: float) -> int:
    """Inverse-CDF lookup returning a 0-based index, clipped against float round-off at the top."""
    return int(min(np.searchsorted(cdf, u * cdf[-1], side="right"), cdf.shape[0] - 1))


def _fill(cdf: np.ndarray, k: int, step: int, phase: float) -> tuple[list[int], int]:
    """Draw ``k`` distinct 1-based numbers by sweeping the Kronecker sequence; returns (nums, step)."""
    picked: list[int] = []
    guard = 0
    while len(picked) < k and guard < 200 * k:
        value = _pick(cdf, (phase + step * GOLDEN) % 1.0) + 1
        step += 1
        guard += 1
        if value not in picked:
            picked.append(value)
    return picked, step


def build_portfolio_from_weights(
    spec: GameSpec,
    budget: int,
    main_w: np.ndarray,
    star_w: np.ndarray | None,
    *,
    max_shared_main: int | None = None,
    phase: float = 0.5,
) -> list[Ticket]:
    """Deterministic low-discrepancy portfolio of distinct legal tickets drawn against ``main_w``.

    The overlap cap is applied first; if it starves the portfolio the remaining slots are filled
    without it, so the returned budget is always met (equal-budget comparison is non-negotiable).
    """
    budget = min(int(budget), spec.n_tickets())
    if max_shared_main is None:
        max_shared_main = max(1, spec.main_k - 2)
    main_cdf = np.cumsum(np.asarray(main_w, dtype=float))
    star_cdf = np.cumsum(np.asarray(star_w, dtype=float)) if spec.star_k > 0 else None

    out: list[Ticket] = []
    seen: set[Ticket] = set()
    kept: list[set[int]] = []
    step = 0
    for enforce_overlap in (True, False):
        attempts = 0
        max_attempts = max(budget * 200, 2000)
        while len(out) < budget and attempts < max_attempts:
            attempts += 1
            mains, step = _fill(main_cdf, spec.main_k, step, phase)
            if len(mains) < spec.main_k:
                break
            if spec.star_k > 0:
                stars, step = _fill(star_cdf, spec.star_k, step, phase)
                if len(stars) < spec.star_k:
                    break
            else:
                stars = []
            ticket: Ticket = (tuple(sorted(mains)), tuple(sorted(stars)))
            if ticket in seen:
                continue
            ms = set(ticket[0])
            if enforce_overlap and any(len(ms & prev) > max_shared_main for prev in kept):
                continue
            seen.add(ticket)
            kept.append(ms)
            out.append(ticket)
        if len(out) >= budget:
            break
    return out


class PerronFrobeniusProvider(InferenceProvider):
    """PageRank-style stationary ranking of the co-occurrence graph, as a competing provider.

    It states its own strength: ``tv_from_uniform`` in the diagnostics is the distance between the
    learned ranking and "no information at all". Compare it against :func:`null_tv_band` before
    believing any ordering this provider produces.
    """

    name = "perron_frobenius"
    description = (
        "Damped Markov random-walk (PageRank) stationary ranking of the pair co-occurrence graph, "
        "sampled by a deterministic low-discrepancy inverse-CDF sweep. Reports its own distance "
        "from the uninformative uniform ranking; claims no change to draw odds."
    )

    def __init__(
        self,
        *,
        damping: float = 0.85,
        orientation: str = "affinity",
        max_shared_main: int | None = None,
    ) -> None:
        if orientation not in ORIENTATIONS:
            raise ValueError(f"orientation must be one of {ORIENTATIONS}, got {orientation!r}")
        if not (0.0 < damping < 1.0):
            raise ValueError(f"damping must lie strictly inside (0, 1), got {damping}")
        self.damping = float(damping)
        self.orientation = orientation
        # Both orientations can run in the same benchmark, so the provenance name must distinguish
        # them (the evaluator keys per-provider results by name).
        self.name = f"{type(self).name}_{orientation}"
        self.max_shared_main = max_shared_main
        self._main: StationaryRank | None = None
        self._star: StationaryRank | None = None
        self._star_fallback = False
        self._history_rows = 0

    def fit(self, history: pd.DataFrame, spec: GameSpec | None = None) -> "PerronFrobeniusProvider":
        if spec is None:
            raise ValueError("PerronFrobeniusProvider.fit requires a GameSpec")
        cfg = GameConfig(spec.name, spec.main_n, spec.main_k, spec.star_n, spec.star_k)
        main_cols, star_cols = detect_columns(history, cfg)
        W = build_comatrices(history, cfg, main_cols, star_cols)
        self._history_rows = len(history)
        self._main = stationary_distribution(W.Wmain, damping=self.damping)
        if spec.star_k > 0:
            if W.Wstar.sum() > 0:
                # star_k >= 2: real star-star edges exist, so the walk is well defined.
                self._star = stationary_distribution(W.Wstar, damping=self.damping)
                self._star_fallback = False
            else:
                # star_k == 1: a draw never produces a star-star pair, so the star graph has no
                # edges and its walk would be exactly uniform. Fall back to the cross-graph
                # marginal (how much main-mass each bonus ball attracts), which is the rank-1
                # stationary vector of the only chain the data actually supports.
                mass = W.Wcross.sum(axis=0).astype(float)
                pi = mass / mass.sum() if mass.sum() > 0 else np.full(spec.star_n, 1.0 / spec.star_n)
                self._star = StationaryRank(
                    pi=pi,
                    iterations=1,
                    residual=0.0,
                    tv_from_uniform=0.5 * float(np.abs(pi - 1.0 / spec.star_n).sum()),
                    total_edge_weight=float(mass.sum()),
                )
                self._star_fallback = True
        return self

    def propose(self, spec: GameSpec, budget: int, rng: np.random.Generator) -> ProviderResult:
        del rng  # the Kronecker sweep is deterministic for a fixed history/config
        if self._main is None:
            raise RuntimeError("PerronFrobeniusProvider.propose called before fit()")
        main_w = _orient(self._main.pi, self.orientation)
        star_w = (
            _orient(self._star.pi, self.orientation)
            if (spec.star_k > 0 and self._star is not None)
            else None
        )
        tickets = build_portfolio_from_weights(
            spec, budget, main_w, star_w, max_shared_main=self.max_shared_main
        )
        scores = np.array(
            [
                float(np.log(main_w[np.asarray(m) - 1]).sum())
                + (float(np.log(star_w[np.asarray(s) - 1]).sum()) if star_w is not None and s else 0.0)
                for m, s in tickets
            ]
        )
        return ProviderResult(
            tickets=tickets,
            scores=scores,
            diagnostics={
                "damping": self.damping,
                "orientation": self.orientation,
                "history_rows": self._history_rows,
                "power_iterations": self._main.iterations,
                "residual": self._main.residual,
                "tv_from_uniform": self._main.tv_from_uniform,
                "star_tv_from_uniform": self._star.tv_from_uniform if self._star else None,
                "star_used_cross_marginal": self._star_fallback,
            },
        )
