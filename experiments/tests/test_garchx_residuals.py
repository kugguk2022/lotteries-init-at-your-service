from __future__ import annotations

import numpy as np

from euromillions.garchx_residuals import (
    ResidualMixtureComponent,
    fit_residual_distribution,
    mixture_distribution,
    residual_cdf,
    residual_ppf,
)


def test_student_t_mixture_fit_finds_two_components() -> None:
    rng = np.random.default_rng(4321)
    left = rng.standard_t(df=5.0, size=320) * 0.55 - 1.25
    right = rng.standard_t(df=8.0, size=320) * 0.75 + 1.1
    resid = np.concatenate([left, right]).astype(float)

    mixture_fit = fit_residual_distribution(resid, residual_model="student_t_mixture")
    student_fit = fit_residual_distribution(resid, residual_model="student_t")

    assert mixture_fit.family == "student_t_mixture"
    assert mixture_fit.components is not None
    assert len(mixture_fit.components) == 2
    assert mixture_fit.aic < student_fit.aic


def test_student_t_mixture_quantiles_round_trip_through_cdf() -> None:
    dist = mixture_distribution(
        [
            ResidualMixtureComponent(weight=0.45, loc=-0.9, scale=0.35, df=6.0),
            ResidualMixtureComponent(weight=0.55, loc=0.85, scale=0.5, df=9.0),
        ],
        log_likelihood=-80.0,
    )

    for quantile in [0.025, 0.10, 0.50, 0.90, 0.975]:
        point = residual_ppf(dist, quantile)
        recovered = residual_cdf(dist, point)
        assert np.isclose(recovered, quantile, atol=1e-5)
