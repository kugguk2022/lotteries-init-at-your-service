"""Crowd-forecast track: predict how many *people* win each prize tier, not which balls drop.

Why this track exists
---------------------
A fair draw cannot be forecast; every benchmark in this repository that scores hit rate against
the uniform null converges on lift 1.0. But the number of *winners* in each prize tier is not
random in the same way: it is ``entries x P(a crowd ticket lands in that tier | drawn numbers)``,
and the second factor is a property of how humans choose numbers. Operators publish the realised
winner counts for every draw, so this is a forecasting target with ground truth, a proper null
(the crowd picks uniformly), and a large learnable signal: on EuroMillions the realised counts
deviate from the uniform-crowd expectation by roughly +/-10-20% per tier, draw after draw.

It is also the quantity that matters economically. Every EuroMillions tier is pari-mutuel, so
prize-per-winner is ``tier fund / winners``. A model that forecasts winners forecasts payouts.
See :mod:`lotteries_core.tier_payout` for the expected-return layer built on top of this model.

The model
---------
The crowd picks a ``k``-subset of the main pool with probability proportional to the product of
per-number weights ``w_i = exp(theta_i)`` (a conditional-Bernoulli / exponential-family model on
subsets), and independently a star pair the same way. A fraction ``u`` of entries are machine
"lucky dips" that pick uniformly. For drawn numbers ``W`` the probability that a crowd ticket
matches exactly ``a`` of them is closed-form in elementary symmetric polynomials ``e_j``::

    P(a | W) = e_a(w_W) * e_{k-a}(w_notW) / e_k(w_all)

so expected winners per tier are exact and cheap, with no simulation.

Scope and claims boundary
-------------------------
This forecasts human ticket choice. It never changes the probability that any ticket wins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .popularity import PopularityModel
from .protocol import GameSpec

__all__ = [
    "CrowdModel",
    "TierTable",
    "WinnerHistory",
    "calibration_slope",
    "esp",
    "fit_crowd_model",
    "forward_benchmark",
    "load_winner_history",
    "static_prior_model",
    "tier_keys",
    "uniform_model",
]

#: Model-error scale (log units) used in the quasi-likelihood weights. With ``mu`` expected
#: winners, ``Var[log count] ~= 1/mu + SIGMA_MODEL**2``: tiny tiers are Poisson-noise limited,
#: million-winner tiers are model-error limited and must not drown everything else.
SIGMA_MODEL = 0.05


# ------------------------------------------------------------------------------------------
# Game combinatorics
# ------------------------------------------------------------------------------------------
def tier_keys(spec: GameSpec) -> list[tuple[int, int]]:
    """Every (main matches, star matches) cell, including non-paying ones."""
    return [(a, b) for a in range(spec.main_k + 1) for b in range(spec.star_k + 1)]


def hypergeometric_tier_probability(spec: GameSpec, a: int, b: int) -> float:
    """P(a uniformly random ticket matches exactly ``a`` mains and ``b`` stars)."""
    main = comb(spec.main_k, a) * comb(spec.main_n - spec.main_k, spec.main_k - a)
    main /= comb(spec.main_n, spec.main_k)
    if spec.star_k == 0:
        return float(main) if b == 0 else 0.0
    star = comb(spec.star_k, b) * comb(spec.star_n - spec.star_k, spec.star_k - b)
    star /= comb(spec.star_n, spec.star_k)
    return float(main * star)


def esp(x: np.ndarray, k: int) -> np.ndarray:
    """Elementary symmetric polynomials ``e_0..e_k`` of each row of ``x`` (shape ``[rows, m]``).

    Returns shape ``[rows, k + 1]``. Standard O(m k) dynamic programme, vectorised over rows.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    out = np.zeros((x.shape[0], k + 1), dtype=float)
    out[:, 0] = 1.0
    for col in range(x.shape[1]):
        out[:, 1:] = out[:, 1:] + out[:, :-1] * x[:, col : col + 1]
    return out


def _match_distribution(weights: np.ndarray, drawn: np.ndarray, k: int) -> np.ndarray:
    """P(crowd ticket matches exactly a of the drawn numbers), shape ``[draws, k + 1]``.

    ``weights`` has length ``n``; ``drawn`` is ``[draws, k]`` of 1-based numbers.
    """
    n = weights.shape[0]
    idx = np.asarray(drawn, dtype=int) - 1
    in_w = weights[idx]  # [draws, k]
    mask = np.ones((idx.shape[0], n), dtype=bool)
    np.put_along_axis(mask, idx, False, axis=1)
    out_w = np.broadcast_to(weights, mask.shape)[mask].reshape(idx.shape[0], n - k)
    e_in = esp(in_w, k)  # e_a(w_W)
    e_out = esp(out_w, k)  # e_j(w_notW)
    total = esp(weights[None, :], k)[0, k]
    return e_in * e_out[:, ::-1] / total


# ------------------------------------------------------------------------------------------
# The crowd model
# ------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CrowdModel:
    """Product-form crowd choice model with a uniform lucky-dip mixture.

    ``theta_main``/``theta_star`` are log-weights (mean zero by convention; only differences
    matter). ``lucky_dip`` is the fraction of entries assumed to be uniform machine picks.
    """

    spec: GameSpec
    theta_main: np.ndarray
    theta_star: np.ndarray
    lucky_dip: float = 0.0
    name: str = "crowd"
    diagnostics: dict = field(default_factory=dict, compare=False)

    def tier_probabilities(
        self, main_drawn: np.ndarray, star_drawn: np.ndarray | None = None
    ) -> np.ndarray:
        """P(crowd ticket lands in tier (a, b)) per draw, shape ``[draws, k+1, star_k+1]``."""
        spec = self.spec
        main_drawn = np.atleast_2d(main_drawn)
        pm = _match_distribution(np.exp(self.theta_main), main_drawn, spec.main_k)
        hm = np.array(
            [hypergeometric_tier_probability(spec, a, 0) for a in range(spec.main_k + 1)]
        )
        if spec.star_k > 0:
            if star_drawn is None:
                raise ValueError("star_drawn is required for a game with a star pool")
            ps = _match_distribution(
                np.exp(self.theta_star), np.atleast_2d(star_drawn), spec.star_k
            )
            hs = np.array(
                [
                    comb(spec.star_k, b)
                    * comb(spec.star_n - spec.star_k, spec.star_k - b)
                    / comb(spec.star_n, spec.star_k)
                    for b in range(spec.star_k + 1)
                ]
            )
            # hm above already folds in P(b=0) for stars; undo that to get the pure main marginal.
            hm = hm / hs[0]
        else:
            ps = np.ones((main_drawn.shape[0], 1))
            hs = np.ones(1)
        chosen = pm[:, :, None] * ps[:, None, :]
        uniform = (hm[:, None] * hs[None, :])[None, :, :]
        u = float(self.lucky_dip)
        return (1.0 - u) * chosen + u * uniform

    def expected_winners(
        self, main_drawn: np.ndarray, star_drawn: np.ndarray | None, entries: np.ndarray
    ) -> np.ndarray:
        """Expected winners per tier per draw, shape ``[draws, k+1, star_k+1]``."""
        probs = self.tier_probabilities(main_drawn, star_drawn)
        return probs * np.asarray(entries, dtype=float).reshape(-1, 1, 1)

    def ticket_log_crowding(self, main: Sequence[int], star: Sequence[int] = ()) -> float:
        """Sum of log-weights of a ticket's numbers: the first-order crowding score.

        Lower means fewer people share these numbers. Used to rank candidate pools cheaply; the
        exact expected-return calculation lives in :mod:`lotteries_core.tier_payout`.
        """
        total = float(np.sum(self.theta_main[np.asarray(main, dtype=int) - 1]))
        if len(star):
            total += float(np.sum(self.theta_star[np.asarray(star, dtype=int) - 1]))
        return total

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "game": self.spec.name,
            "lucky_dip": float(self.lucky_dip),
            "theta_main": [float(v) for v in self.theta_main],
            "theta_star": [float(v) for v in self.theta_star],
        }


def uniform_model(spec: GameSpec) -> CrowdModel:
    """The null: everybody picks uniformly at random."""
    return CrowdModel(
        spec, np.zeros(spec.main_n), np.zeros(max(spec.star_n, 0)), 0.0, name="uniform_crowd"
    )


def static_prior_model(spec: GameSpec, prior: PopularityModel | None = None) -> CrowdModel:
    """The repository's hand-set popularity prior, expressed in this model's terms.

    Only the per-number weights transfer (pattern multipliers are not product-form). Stars are
    uniform because the static prior has no star component. This entrant exists so the prior that
    currently drives the Crowd Escape agent is graded against real winner counts.
    """
    prior = prior or PopularityModel()
    theta = np.log(prior.number_weights(spec))
    return CrowdModel(
        spec, theta - theta.mean(), np.zeros(max(spec.star_n, 0)), 0.0, name="static_prior"
    )


# ------------------------------------------------------------------------------------------
# Data contract
# ------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TierTable:
    """Which tiers are observed (paying tiers with published winner counts)."""

    tiers: tuple[tuple[int, int], ...]

    @classmethod
    def euromillions(cls) -> "TierTable":
        return cls(
            (
                (5, 2), (5, 1), (5, 0), (4, 2), (4, 1), (3, 2), (4, 0),
                (2, 2), (3, 1), (3, 0), (1, 2), (2, 1), (2, 0),
            )
        )

    def columns(self) -> list[str]:
        return [f"w_{a}_{b}" for a, b in self.tiers]


@dataclass(frozen=True)
class WinnerHistory:
    """Per-draw drawn numbers, total entries, and realised winners per observed tier."""

    spec: GameSpec
    table: TierTable
    dates: np.ndarray  # [draws] ISO strings, ascending
    main: np.ndarray  # [draws, main_k]
    star: np.ndarray  # [draws, star_k]
    entries: np.ndarray  # [draws]
    winners: np.ndarray  # [draws, tiers]

    def __len__(self) -> int:
        return int(self.dates.shape[0])

    def slice(self, index: np.ndarray | slice) -> "WinnerHistory":
        return WinnerHistory(
            self.spec, self.table, self.dates[index], self.main[index], self.star[index],
            self.entries[index], self.winners[index],
        )


def load_winner_history(
    path: str | Path, spec: GameSpec | None = None, table: TierTable | None = None
) -> WinnerHistory:
    """Load and validate a winner-count CSV (see ``data/crowd/README.md`` for the schema).

    Fails closed on anything that looks like a transcription error: a row whose total winners
    are not within a plausible band of ``entries x P(any prize)`` is rejected, not repaired.
    """
    spec = spec or GameSpec.euromillions()
    table = table or TierTable.euromillions()
    frame = pd.read_csv(path).sort_values("draw_date").reset_index(drop=True)
    main_cols = [f"ball_{i}" for i in range(1, spec.main_k + 1)]
    star_cols = [f"star_{i}" for i in range(1, spec.star_k + 1)]
    missing = [c for c in ["draw_date", "entries", *main_cols, *star_cols, *table.columns()]
               if c not in frame.columns]
    if missing:
        raise ValueError(f"winner history is missing columns: {missing}")
    if frame["draw_date"].duplicated().any():
        raise ValueError("winner history has duplicate draw dates")
    main = np.sort(frame[main_cols].to_numpy(dtype=int), axis=1)
    star = np.sort(frame[star_cols].to_numpy(dtype=int), axis=1)
    for row_m, row_s in zip(main, star):
        spec.validate_ticket((tuple(int(v) for v in row_m), tuple(int(v) for v in row_s)))
    entries = frame["entries"].to_numpy(dtype=float)
    winners = frame[table.columns()].to_numpy(dtype=float)
    if np.any(entries <= 0) or np.any(winners < 0) or np.any(~np.isfinite(winners)):
        raise ValueError("entries must be positive and winner counts non-negative")
    p_any = sum(hypergeometric_tier_probability(spec, a, b) for a, b in table.tiers)
    ratio = winners.sum(axis=1) / (entries * p_any)
    bad = np.where((ratio < 0.5) | (ratio > 2.0))[0]
    if bad.size:
        raise ValueError(
            "implausible winners/entries ratio (check transcription) on: "
            + ", ".join(str(frame["draw_date"].iloc[i]) for i in bad)
        )
    return WinnerHistory(
        spec, table, frame["draw_date"].astype(str).to_numpy(), main, star, entries, winners
    )


# ------------------------------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------------------------------
def _tier_index(table: TierTable) -> tuple[np.ndarray, np.ndarray]:
    return (np.array([a for a, _ in table.tiers]), np.array([b for _, b in table.tiers]))


def predict(model: CrowdModel, history: WinnerHistory) -> np.ndarray:
    """Expected winners per observed tier, shape ``[draws, tiers]``."""
    full = model.expected_winners(history.main, history.star, history.entries)
    a_idx, b_idx = _tier_index(history.table)
    return full[:, a_idx, b_idx]


def _null_expectation(history: WinnerHistory) -> np.ndarray:
    probs = np.array(
        [hypergeometric_tier_probability(history.spec, a, b) for a, b in history.table.tiers]
    )
    return history.entries[:, None] * probs[None, :]


def weighted_log_error(pred: np.ndarray, history: WinnerHistory) -> np.ndarray:
    """Per-draw quasi-likelihood loss: precision-weighted squared log-ratio, summed over tiers.

    Weights depend only on the uniform-null expectation, so every entrant is scored with the
    same weights and no entrant can move them.
    """
    mu0 = _null_expectation(history)
    weight = 1.0 / (1.0 / mu0 + SIGMA_MODEL**2)
    resid = np.log(history.winners + 0.5) - np.log(pred + 0.5)
    return (weight * resid**2).sum(axis=1)


def big_tier_rmse(pred: np.ndarray, history: WinnerHistory, min_expected: float = 1000.0) -> float:
    """RMSE of log(actual / predicted) over tiers with >= ``min_expected`` null winners.

    Readable as "typical percentage miss on the winner count" for the tiers that hold most of
    the lower-tier prize money.
    """
    mask = _null_expectation(history) >= min_expected
    resid = np.log(history.winners + 0.5) - np.log(pred + 0.5)
    return float(np.sqrt(np.mean(resid[mask] ** 2)))


def calibration_slope(
    pred: np.ndarray, history: WinnerHistory, min_expected: float = 1000.0
) -> float:
    """Slope of log(actual / null) on log(predicted / null) over large tiers (through origin).

    1.0 means the forecast deviations from the uniform crowd have the right size; below 1.0 they
    are overstated (and any payout edge priced from the model should be scaled down accordingly).
    """
    mu0 = _null_expectation(history)
    mask = mu0 >= min_expected
    x = (np.log(pred + 0.5) - np.log(mu0 + 0.5))[mask]
    y = (np.log(history.winners + 0.5) - np.log(mu0 + 0.5))[mask]
    denom = float(x @ x)
    return float(x @ y / denom) if denom > 1e-12 else float("nan")


# ------------------------------------------------------------------------------------------
# Fitting
# ------------------------------------------------------------------------------------------
def _features(n: int) -> np.ndarray:
    """Low-dimensional structure the literature on number choice predicts, as a design matrix.

    Columns: calendar day (<= 31), month (<= 12), multiple of 7, single digit, centred linear
    trend, and edge-of-playslip proxy (centred quadratic). The fit learns the size *and sign* of
    each from winner counts; nothing here is assumed to be positive.
    """
    num = np.arange(1, n + 1, dtype=float)
    lin = (num - num.mean()) / num.std()
    cols = [
        (num <= 31).astype(float),
        (num <= 12).astype(float),
        (num % 7 == 0).astype(float),
        (num <= 9).astype(float),
        lin,
        lin**2 - np.mean(lin**2),
    ]
    x = np.stack(cols, axis=1)
    return x - x.mean(axis=0, keepdims=True)


def fit_crowd_model(
    history: WinnerHistory,
    *,
    ridge_residual: float = 40.0,
    ridge_feature: float = 0.5,
    ridge_star: float = 10.0,
    fit_lucky_dip: bool = True,
    name: str = "fitted_crowd",
    start: CrowdModel | None = None,
) -> CrowdModel:
    """Fit log-weights to realised winner counts by penalised quasi-likelihood.

    ``theta_main = X beta + delta``: a handful of structural effects plus a per-number residual
    that is shrunk hard (``ridge_residual``) so that a number drawn only once or twice in the
    training window cannot be fitted to noise. Stars get a plain ridge. All penalties are fixed
    constants, never tuned on evaluation draws.
    """
    from scipy.optimize import minimize

    spec = history.spec
    x_main = _features(spec.main_n)
    n_beta = x_main.shape[1]
    n_main, n_star = spec.main_n, max(spec.star_n, 0)
    a_idx, b_idx = _tier_index(history.table)
    mu0 = _null_expectation(history)
    weight = 1.0 / (1.0 / mu0 + SIGMA_MODEL**2)
    log_obs = np.log(history.winners + 0.5)
    n_draws = max(len(history), 1)

    def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
        beta = params[:n_beta]
        delta = params[n_beta : n_beta + n_main]
        phi = params[n_beta + n_main : n_beta + n_main + n_star]
        raw_u = params[-1]
        u = float(1.0 / (1.0 + np.exp(-raw_u))) if fit_lucky_dip else 0.0
        theta = x_main @ beta + delta
        return theta - theta.mean(), phi - (phi.mean() if n_star else 0.0), u, beta, delta

    def objective(params: np.ndarray) -> float:
        theta, phi, u, beta, delta = unpack(params)
        model = CrowdModel(spec, theta, phi, u)
        pred = model.expected_winners(history.main, history.star, history.entries)[
            :, a_idx, b_idx
        ]
        loss = float((weight * (log_obs - np.log(pred + 0.5)) ** 2).sum()) / n_draws
        loss += ridge_feature * float(beta @ beta)
        loss += ridge_residual * float(delta @ delta)
        loss += ridge_star * float(phi @ phi)
        return loss

    x0 = np.zeros(n_beta + n_main + n_star + 1)
    x0[-1] = -1.0  # lucky-dip fraction starts near 27%
    if start is not None and start.diagnostics.get("params") is not None:
        x0 = np.asarray(start.diagnostics["params"], dtype=float)
    result = minimize(objective, x0, method="L-BFGS-B", options={"maxiter": 400})
    theta, phi, u, beta, _delta = unpack(result.x)
    return CrowdModel(
        spec, theta, phi, u, name=name,
        diagnostics={
            "params": result.x.copy(),
            "objective": float(result.fun),
            "converged": bool(result.success),
            "training_draws": len(history),
            "feature_effects": {
                key: float(value)
                for key, value in zip(
                    ("calendar_le31", "month_le12", "multiple_of_7", "single_digit",
                     "linear", "quadratic"),
                    beta,
                )
            },
        },
    )


# ------------------------------------------------------------------------------------------
# Forward-only benchmark
# ------------------------------------------------------------------------------------------
Forecaster = Callable[[WinnerHistory], CrowdModel]


def default_forecasters(spec: GameSpec) -> dict[str, Forecaster]:
    """Entrants shipped with the track. Each maps a *training* history to a committed model."""
    previous: list[CrowdModel] = []

    def fitted(train: WinnerHistory) -> CrowdModel:
        # Warm start from the previous (strictly earlier) fit: same optimum, far fewer steps.
        model = fit_crowd_model(train, start=previous[-1] if previous else None)
        previous.append(model)
        return model

    return {
        "uniform_crowd": lambda _train: uniform_model(spec),
        "static_prior": lambda _train: static_prior_model(spec),
        "fitted_crowd": fitted,
    }


def _paired_bootstrap_ci(
    loss_model: np.ndarray, loss_null: np.ndarray, *, n_boot: int = 4000, seed: int = 20261109
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = loss_model.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    skill = 1.0 - loss_model[idx].sum(axis=1) / loss_null[idx].sum(axis=1)
    lo, hi = np.percentile(skill, [2.5, 97.5])
    return float(lo), float(hi)


def forward_benchmark(
    history: WinnerHistory,
    forecasters: dict[str, Forecaster] | None = None,
    *,
    min_train: int = 20,
    null_name: str = "uniform_crowd",
) -> dict:
    """Walk forward: for each draw ``i >= min_train`` fit on draws ``< i`` and forecast draw ``i``.

    The forecast is the committed crowd model; it is evaluated at the realised numbers and
    entries of the held-out draw. ``skill`` is ``1 - loss / null loss`` on the precision-weighted
    log error, with a paired bootstrap interval over evaluation draws. Skill 0 is the null;
    negative skill means the entrant is worse than assuming a uniform crowd.
    """
    forecasters = forecasters or default_forecasters(history.spec)
    if null_name not in forecasters:
        raise ValueError(f"null forecaster {null_name!r} missing")
    if len(history) <= min_train:
        raise ValueError(f"need more than {min_train} draws, have {len(history)}")
    eval_index = np.arange(min_train, len(history))
    losses = {name: [] for name in forecasters}
    preds = {name: [] for name in forecasters}
    for i in eval_index:
        train = history.slice(slice(0, int(i)))
        test = history.slice(slice(int(i), int(i) + 1))
        for name, make in forecasters.items():
            pred = predict(make(train), test)
            losses[name].append(float(weighted_log_error(pred, test)[0]))
            preds[name].append(pred[0])
    test_all = history.slice(eval_index)
    null_loss = np.array(losses[null_name])
    rows = []
    for name in forecasters:
        loss = np.array(losses[name])
        pred = np.stack(preds[name])
        lo, hi = (0.0, 0.0) if name == null_name else _paired_bootstrap_ci(loss, null_loss)
        rows.append(
            {
                "forecaster": name,
                "eval_draws": int(eval_index.size),
                "skill_vs_uniform": float(1.0 - loss.sum() / null_loss.sum()),
                "skill_ci95_low": lo,
                "skill_ci95_high": hi,
                "draws_better_than_null": int((loss < null_loss).sum()),
                "big_tier_log_rmse": big_tier_rmse(pred, test_all),
                "calibration_slope": calibration_slope(pred, test_all),
                "mean_weighted_log_error": float(loss.mean()),
            }
        )
    rows.sort(key=lambda r: -r["skill_vs_uniform"])
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return {
        "track": "crowd_forecast",
        "version": "crowd-forecast-v1",
        "game": history.spec.name,
        "protocol": "forward_only_refit_each_draw",
        "min_train": int(min_train),
        "first_evaluated_draw": str(history.dates[eval_index[0]]),
        "last_evaluated_draw": str(history.dates[-1]),
        "history_draws": len(history),
        "metric": "1 - weighted_log_error / uniform_crowd weighted_log_error",
        "claims_boundary": (
            "Forecasts how many tickets share a prize tier given the drawn numbers. It does not "
            "forecast the draw, and no entrant changes any ticket's probability of winning."
        ),
        "leaderboard": rows,
        "per_draw": {
            "draw_date": [str(d) for d in history.dates[eval_index]],
            **{f"loss_{name}": [float(v) for v in losses[name]] for name in forecasters},
        },
        # Held-out forecasts per tier, so a reader can recompute every number above.
        "tiers": [f"{a}+{b}" for a, b in history.table.tiers],
        "actual_winners": [[float(v) for v in row] for row in test_all.winners],
        "predicted_winners": {
            name: [[float(v) for v in row] for row in preds[name]] for name in forecasters
        },
    }
