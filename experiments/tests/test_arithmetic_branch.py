from __future__ import annotations

import numpy as np

from euromillions.arithmetic_branch import (
    ResidualMixtureComponent,
    fit_residual_distribution,
    mixture_distribution,
    residual_cdf,
    residual_ppf,
)


def test_student_t_mixture_beats_single_t_on_double_humped_residuals() -> None:
    rng = np.random.default_rng(1234)
    left = rng.standard_t(df=5.0, size=350) * 4.0 - 10.0
    right = rng.standard_t(df=7.0, size=350) * 5.0 + 11.0
    resid = np.concatenate([left, right]).astype(float)

    mixture_fit = fit_residual_distribution(resid, residual_model="student_t_mixture")
    student_fit = fit_residual_distribution(resid, residual_model="student_t")
    auto_fit = fit_residual_distribution(resid, residual_model="auto")

    assert mixture_fit.family == "student_t_mixture"
    assert mixture_fit.components is not None
    assert len(mixture_fit.components) == 2
    assert mixture_fit.aic < student_fit.aic
    assert auto_fit.family in {"normal", "student_t"}


def test_student_t_mixture_quantiles_round_trip_through_cdf() -> None:
    dist = mixture_distribution(
        [
            ResidualMixtureComponent(weight=0.45, loc=-8.0, scale=3.0, df=6.0),
            ResidualMixtureComponent(weight=0.55, loc=9.5, scale=4.5, df=9.0),
        ],
        log_likelihood=-100.0,
    )

    for quantile in [0.025, 0.10, 0.50, 0.90, 0.975]:
        point = residual_ppf(dist, quantile)
        recovered = residual_cdf(dist, point)
        assert np.isclose(recovered, quantile, atol=1e-5)
