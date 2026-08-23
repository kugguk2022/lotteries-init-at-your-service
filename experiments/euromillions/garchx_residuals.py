from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import expit, logsumexp
from scipy.stats import t as t_dist

RESIDUAL_MODEL_CHOICES = ("student_t", "student_t_mixture")
MIN_RESID_SCALE = 1e-6
MIN_T_DF = 2.1


@dataclass
class ResidualMixtureComponent:
    weight: float
    loc: float
    scale: float
    df: float


@dataclass
class ResidualDistribution:
    family: str
    loc: float
    scale: float
    df: float | None
    aic: float
    components: list[ResidualMixtureComponent] | None = None


def mixture_component_variance(component: ResidualMixtureComponent) -> float:
    return (component.df / max(component.df - 2.0, 1e-6)) * (component.scale**2)


def mixture_distribution(
    components: list[ResidualMixtureComponent],
    *,
    log_likelihood: float,
) -> ResidualDistribution:
    mean = float(sum(component.weight * component.loc for component in components))
    variance = float(
        sum(
            component.weight
            * (mixture_component_variance(component) + ((component.loc - mean) ** 2))
            for component in components
        )
    )
    return ResidualDistribution(
        family="student_t_mixture",
        loc=mean,
        scale=max(float(np.sqrt(max(variance, MIN_RESID_SCALE**2))), MIN_RESID_SCALE),
        df=None,
        aic=(2.0 * 7.0) - (2.0 * float(log_likelihood)),
        components=components,
    )


def _logit(prob: float) -> float:
    clipped = float(np.clip(prob, 1e-4, 1.0 - 1e-4))
    return float(np.log(clipped / (1.0 - clipped)))


def _positive_from_log_param(param: float, *, floor: float) -> float:
    return max(float(np.exp(np.clip(float(param), -12.0, 8.0))), floor)


def _student_t_mixture_components_from_params(params: np.ndarray) -> list[ResidualMixtureComponent]:
    weight_left = float(expit(float(params[0])))
    loc_left = float(params[1])
    scale_left = _positive_from_log_param(float(params[2]), floor=MIN_RESID_SCALE)
    df_left = float(MIN_T_DF + _positive_from_log_param(float(params[3]), floor=1e-3))
    loc_right = float(params[4])
    scale_right = _positive_from_log_param(float(params[5]), floor=MIN_RESID_SCALE)
    df_right = float(MIN_T_DF + _positive_from_log_param(float(params[6]), floor=1e-3))
    return [
        ResidualMixtureComponent(weight=weight_left, loc=loc_left, scale=scale_left, df=df_left),
        ResidualMixtureComponent(weight=1.0 - weight_left, loc=loc_right, scale=scale_right, df=df_right),
    ]


def _student_t_mixture_negloglik(params: np.ndarray, arr: np.ndarray) -> float:
    components = _student_t_mixture_components_from_params(params)
    log_terms = np.vstack(
        [
            np.log(max(component.weight, 1e-12))
            + t_dist.logpdf(arr, df=component.df, loc=component.loc, scale=component.scale)
            for component in components
        ]
    )
    value = -float(np.sum(logsumexp(log_terms, axis=0)))
    if not np.isfinite(value):
        return 1e18
    return value


def _student_t_mixture_initial_params(arr: np.ndarray) -> list[np.ndarray]:
    centered = arr - float(arr.mean())
    spread = max(float(arr.std(ddof=0)), 1.0)
    quantiles = np.quantile(arr, [0.35, 0.5, 0.65])

    masks = [
        arr <= quantiles[1],
        arr < 0.0,
        arr <= quantiles[0],
        arr <= quantiles[2],
        centered <= 0.0,
    ]
    inits: list[np.ndarray] = []
    for mask in masks:
        left = arr[mask]
        right = arr[~mask]
        if len(left) < 10 or len(right) < 10:
            continue
        w0 = float(len(left) / len(arr))
        left_scale = max(float(left.std(ddof=0)), 0.25 * spread, MIN_RESID_SCALE)
        right_scale = max(float(right.std(ddof=0)), 0.25 * spread, MIN_RESID_SCALE)
        inits.append(
            np.array(
                [
                    _logit(w0),
                    float(left.mean()),
                    float(np.log(left_scale)),
                    float(np.log(6.0 - MIN_T_DF)),
                    float(right.mean()),
                    float(np.log(right_scale)),
                    float(np.log(6.0 - MIN_T_DF)),
                ],
                dtype=float,
            )
        )

    inits.append(
        np.array(
            [
                0.0,
                float(arr.mean() - 0.75 * spread),
                float(np.log(max(0.6 * spread, MIN_RESID_SCALE))),
                float(np.log(8.0 - MIN_T_DF)),
                float(arr.mean() + 0.75 * spread),
                float(np.log(max(0.6 * spread, MIN_RESID_SCALE))),
                float(np.log(8.0 - MIN_T_DF)),
            ],
            dtype=float,
        )
    )
    return inits


def fit_student_t_distribution(arr: np.ndarray) -> ResidualDistribution:
    df_t, loc_t, scale_t = t_dist.fit(arr)
    scale_t = max(float(scale_t), MIN_RESID_SCALE)
    ll_t = float(np.sum(t_dist.logpdf(arr, df=float(df_t), loc=float(loc_t), scale=scale_t)))
    aic_t = 2.0 * 3.0 - 2.0 * ll_t
    return ResidualDistribution(
        family="student_t",
        loc=float(loc_t),
        scale=scale_t,
        df=float(df_t),
        aic=aic_t,
    )


def fit_student_t_mixture_distribution(arr: np.ndarray) -> ResidualDistribution:
    best_result = None
    best_value = float("inf")
    for init in _student_t_mixture_initial_params(arr)[:4]:
        result = minimize(
            _student_t_mixture_negloglik,
            init,
            args=(arr,),
            method="L-BFGS-B",
            options={"maxiter": 450},
        )
        objective = float(result.fun)
        if np.isfinite(objective) and objective < best_value:
            best_value = objective
            best_result = result

    if best_result is None:
        raise RuntimeError("Student-t mixture fit did not converge.")

    components = _student_t_mixture_components_from_params(np.asarray(best_result.x, dtype=float))
    return mixture_distribution(components, log_likelihood=-best_value)


def fit_residual_distribution(resid: np.ndarray, *, residual_model: str) -> ResidualDistribution:
    arr = np.asarray(resid, dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot fit a residual distribution to an empty array.")
    if residual_model == "student_t":
        return fit_student_t_distribution(arr)
    if residual_model == "student_t_mixture":
        return fit_student_t_mixture_distribution(arr)
    raise ValueError(f"Unsupported residual_model={residual_model}")


def residual_cdf(dist: ResidualDistribution, x: float) -> float:
    point = float(x)
    if dist.family == "student_t_mixture" and dist.components:
        return float(
            sum(
                component.weight * t_dist.cdf(point, df=component.df, loc=component.loc, scale=component.scale)
                for component in dist.components
            )
        )
    if dist.family == "student_t" and dist.df is not None:
        return float(t_dist.cdf(point, df=dist.df, loc=dist.loc, scale=dist.scale))
    raise ValueError(f"Unsupported distribution family={dist.family}")


def residual_ppf(dist: ResidualDistribution, quantile: float) -> float:
    q = float(quantile)
    if dist.family == "student_t_mixture" and dist.components:
        lo = min(component.loc - (18.0 * component.scale) for component in dist.components)
        hi = max(component.loc + (18.0 * component.scale) for component in dist.components)
        span = max(hi - lo, 8.0 * dist.scale, 10.0)
        for _ in range(16):
            cdf_lo = residual_cdf(dist, lo)
            cdf_hi = residual_cdf(dist, hi)
            if cdf_lo <= q <= cdf_hi:
                break
            lo -= span
            hi += span
            span *= 1.75
        return float(brentq(lambda value: residual_cdf(dist, value) - q, lo, hi))
    if dist.family == "student_t" and dist.df is not None:
        return float(t_dist.ppf(q, df=dist.df, loc=dist.loc, scale=dist.scale))
    raise ValueError(f"Unsupported distribution family={dist.family}")
