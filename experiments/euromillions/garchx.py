from __future__ import annotations

import argparse
import calendar
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import jarque_bera, probplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

from euromillions.garchx_residuals import (
    RESIDUAL_MODEL_CHOICES,
    ResidualDistribution,
    fit_residual_distribution,
    residual_ppf,
)

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY = REPO_ROOT / "data" / "euromillions.csv"
DEFAULT_POI = REPO_ROOT / "outputs" / "euromillions" / "features" / "poi.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "euromillions" / "garchx"
FLOOR = 1e-2
MODEL_VERSION = "v5.0"


@dataclass
class FitSummary:
    model_version: str
    rows: int
    cutoff_start_date: str
    start_date: str
    end_date: str
    weekday_flag: str
    fourier_order: int
    holdout: int
    best_garch_order: int
    residual_model: str
    nll: float
    mean_const: float
    mean_trend: float
    mean_weekday: float
    alpha: float
    betas: list[float]
    nu: float
    innovation_distribution: ResidualDistribution
    variance_fourier: dict[str, float]
    peak_week: int
    trough_week: int
    variance_ratio: float
    rmse: float
    mae: float
    coverage_80: float
    coverage_95: float
    ljung_box_std_pvalues: dict[str, float]
    ljung_box_sq_pvalues: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a seasonal GARCH-X model to EuroMillions POI features."
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=DEFAULT_HISTORY,
        help="Normalized EuroMillions history CSV with draw_date.",
    )
    parser.add_argument(
        "--poi",
        type=Path,
        default=DEFAULT_POI,
        help="Path to phase2_sobol poi.csv aligned row-for-row with --history.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for charts, fitted CSV, and summary JSON.",
    )
    parser.add_argument(
        "--holdout",
        type=int,
        default=52,
        help="Trailing observations used for interval coverage reporting.",
    )
    parser.add_argument(
        "--fourier-order",
        type=int,
        default=2,
        help="Number of Fourier harmonics in the variance equation.",
    )
    parser.add_argument(
        "--max-garch-order",
        type=int,
        choices=(1, 2),
        default=2,
        help="Fit GARCH(1,1) and optionally GARCH(1,2), then select by AIC.",
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=FLOOR,
        help="Small positive floor added to the seasonal variance baseline.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional inclusive draw-date cutoff in YYYY-MM-DD for a regime-only rerun.",
    )
    parser.add_argument(
        "--residual-model",
        choices=RESIDUAL_MODEL_CHOICES,
        default="student_t_mixture",
        help="v5.0 innovation calibration applied to standardized GARCH residuals.",
    )
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def choose_weekday_flag(dow: pd.Series) -> tuple[str, np.ndarray]:
    unique = sorted(int(value) for value in pd.unique(dow))
    if len(unique) <= 1:
        return "is_weekday_flag", np.zeros(len(dow), dtype=float)
    flagged = unique[-1]
    label = f"is_{calendar.day_name[flagged].lower()}"
    return label, (dow.to_numpy() == flagged).astype(float)


def load_model_frame(
    history_path: Path, poi_path: Path, *, start_date: str | None
) -> tuple[pd.DataFrame, str]:
    history = pd.read_csv(history_path, parse_dates=["draw_date"])
    history = history.sort_values("draw_date").drop_duplicates(subset=["draw_date"]).reset_index(drop=True)

    poi = pd.read_csv(poi_path, header=None).squeeze("columns")
    poi = pd.to_numeric(poi, errors="coerce")
    poi = poi.dropna().reset_index(drop=True)

    if len(history) != len(poi):
        raise ValueError(
            f"history rows ({len(history)}) do not match poi rows ({len(poi)}). "
            "Rebuild features from the same history file before fitting garchx."
        )

    frame = pd.DataFrame({"date": history["draw_date"], "poi": poi})
    if start_date is not None:
        frame = frame[frame["date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
    frame["t"] = np.arange(len(frame), dtype=int)
    frame["dow"] = frame["date"].dt.dayofweek.astype(int)
    frame["woy"] = frame["date"].dt.isocalendar().week.astype(int)
    weekday_flag, values = choose_weekday_flag(frame["dow"])
    frame[weekday_flag] = values
    return frame, weekday_flag


def build_fourier_terms(woy: pd.Series, order: int) -> tuple[np.ndarray, list[str]]:
    cols: dict[str, np.ndarray] = {}
    woy_arr = woy.to_numpy(dtype=float)
    for harmonic in range(1, order + 1):
        cols[f"sin_{harmonic}"] = np.sin(2 * np.pi * harmonic * woy_arr / 52.0)
        cols[f"cos_{harmonic}"] = np.cos(2 * np.pi * harmonic * woy_arr / 52.0)
    names = list(cols.keys())
    return pd.DataFrame(cols).to_numpy(dtype=float), names


def garchx_filter(
    params: np.ndarray,
    eps: np.ndarray,
    fourier_terms: np.ndarray,
    *,
    garch_order: int,
    floor: float,
) -> tuple[float, np.ndarray]:
    omega = params[0]
    alpha = params[1]
    beta_end = 2 + garch_order
    betas = params[2:beta_end]
    gamma_end = beta_end + fourier_terms.shape[1]
    gamma = params[beta_end:gamma_end]
    log_nu = params[gamma_end]
    nu = np.exp(log_nu) + 2.01

    sigma2 = np.zeros(len(eps), dtype=float)
    sigma2[0] = max(np.var(eps), floor)
    log_c = (
        gammaln((nu + 1.0) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * np.log(np.pi * (nu - 2.0))
    )

    ll = 0.0
    for idx in range(1, len(eps)):
        seasonal_var = np.exp(omega + float(fourier_terms[idx] @ gamma)) + floor
        arch_term = alpha * eps[idx - 1] ** 2
        garch_term = 0.0
        for lag, beta in enumerate(betas, start=1):
            if idx - lag >= 0:
                garch_term += beta * sigma2[idx - lag]
        sigma2[idx] = max(seasonal_var + arch_term + garch_term, 1e-6)

        z = eps[idx] / np.sqrt(sigma2[idx])
        ll += log_c - 0.5 * np.log(sigma2[idx]) - ((nu + 1.0) / 2.0) * np.log(
            1.0 + (z**2) / (nu - 2.0)
        )

    return -ll, sigma2


def make_objective(
    eps: np.ndarray, fourier_terms: np.ndarray, *, garch_order: int, floor: float
) -> callable:
    def objective(params: np.ndarray) -> float:
        alpha = params[1]
        betas = params[2 : 2 + garch_order]
        if alpha <= 0 or np.any(betas <= 0):
            return 1e10
        if alpha + float(np.sum(betas)) >= 0.999:
            return 1e10
        nll, _ = garchx_filter(
            params, eps, fourier_terms, garch_order=garch_order, floor=floor
        )
        return nll

    return objective


def fit_garch_orders(
    eps: np.ndarray,
    fourier_terms: np.ndarray,
    *,
    max_garch_order: int,
    floor: float,
) -> tuple[int, object]:
    candidates: dict[int, object] = {}
    n_fourier = fourier_terms.shape[1]

    for order in range(1, max_garch_order + 1):
        p0 = np.array(
            [np.log(max(np.var(eps) * 0.5, floor)), 0.05]
            + [0.80 / order] * order
            + [0.0] * n_fourier
            + [np.log(5.0)],
            dtype=float,
        )
        result = minimize(
            make_objective(eps, fourier_terms, garch_order=order, floor=floor),
            p0,
            method="Nelder-Mead",
            options={"maxiter": 100000, "xatol": 1e-7, "fatol": 1e-7},
        )
        candidates[order] = result

    def aic(order: int) -> float:
        result = candidates[order]
        k_params = 2 + order + n_fourier + 1
        return 2.0 * k_params + 2.0 * float(result.fun)

    best_order = min(candidates, key=aic)
    return best_order, candidates[best_order]


def choose_holdout(n_rows: int, requested: int) -> int:
    if n_rows < 30:
        raise ValueError("Need at least 30 observations for this report.")
    if requested < n_rows:
        return requested
    return max(8, n_rows // 5)


def save_diagnostics_plot(
    df: pd.DataFrame,
    *,
    point_forecast: np.ndarray,
    sigma2: np.ndarray,
    seasonal_profile: np.ndarray,
    seasonal_component: np.ndarray,
    pi_80_lo_full: np.ndarray,
    pi_80_hi_full: np.ndarray,
    pi_95_lo_full: np.ndarray,
    pi_95_hi_full: np.ndarray,
    pred_center: np.ndarray,
    actual_oos: np.ndarray,
    coverage_80: float,
    coverage_95: float,
    std_resid: np.ndarray,
    peak_woy: int,
    trough_woy: int,
    crossings: np.ndarray,
    out_path: Path,
) -> None:
    woy_range = np.arange(1, 53)
    holdout = len(actual_oos)

    fig = plt.figure(figsize=(18, 22))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("Seasonal GARCH-X: Fourier Variance (Beam Model)", fontsize=14, y=1.01)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df["date"], df["poi"], lw=0.6, alpha=0.5, color="steelblue", label="poi")
    ax1.plot(df["date"], point_forecast, lw=1.2, color="navy", label="forecast center")
    ax1.fill_between(
        df["date"],
        pi_95_lo_full,
        pi_95_hi_full,
        alpha=0.25,
        color="tomato",
        label="95% conditional PI",
    )
    ax1.fill_between(
        df["date"],
        pi_80_lo_full,
        pi_80_hi_full,
        alpha=0.35,
        color="tomato",
        label="80% conditional PI",
    )
    ax1.set_title("poi with GARCH-X conditional prediction intervals")
    ax1.legend(fontsize=8)
    ax1.set_ylabel("poi")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(woy_range, np.sqrt(seasonal_profile), color="purple", lw=2)
    ax2.axhline(np.sqrt(seasonal_profile.mean()), color="gray", ls="--", lw=1)
    for cx in crossings:
        ax2.axvline(cx, color="tomato", ls=":", lw=1, alpha=0.7)
    ax2.set_title("Seasonal sigma by week-of-year")
    ax2.set_xlabel("week of year")
    ax2.set_ylabel("sigma")
    ax2.annotate(
        f"peak wk {peak_woy}",
        xy=(peak_woy, np.sqrt(seasonal_profile.max())),
        fontsize=8,
        color="purple",
    )
    ax2.annotate(
        f"trough wk {trough_woy}",
        xy=(trough_woy, np.sqrt(seasonal_profile.min())),
        fontsize=8,
        color="purple",
    )

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(df["date"], np.sqrt(sigma2), lw=0.8, color="tomato")
    ax3.set_title("Conditional sigma_t over time")
    ax3.set_ylabel("sigma_t")

    ax4 = fig.add_subplot(gs[1, 2])
    idx = np.arange(holdout)
    ax4.plot(idx, actual_oos, lw=1, color="steelblue", label="actual")
    ax4.plot(idx, pred_center, lw=1.5, color="navy", ls="--", label="forecast center")
    ax4.fill_between(idx, pi_95_lo_full[-holdout:], pi_95_hi_full[-holdout:], alpha=0.2, color="tomato", label="95% PI")
    ax4.fill_between(idx, pi_80_lo_full[-holdout:], pi_80_hi_full[-holdout:], alpha=0.35, color="tomato", label="80% PI")
    ax4.set_title(f"OOS: 80% cov={coverage_80:.2f}, 95% cov={coverage_95:.2f}")
    ax4.legend(fontsize=7)

    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(df["date"], std_resid, lw=0.5, alpha=0.7, color="steelblue")
    ax5.axhline(0, color="red", lw=0.8)
    ax5.axhline(2, color="gray", ls="--", lw=0.8)
    ax5.axhline(-2, color="gray", ls="--", lw=0.8)
    ax5.set_title("Standardized residuals eps_t/sigma_t")

    ax6 = fig.add_subplot(gs[2, 1])
    probplot(std_resid[1:], dist="norm", plot=ax6)
    ax6.set_title("QQ: standardized residuals")

    ax7 = fig.add_subplot(gs[2, 2])
    max_lag = max(1, min(40, len(std_resid) - 2))
    plot_acf(std_resid[1:] ** 2, lags=max_lag, ax=ax7, zero=False)
    ax7.set_title("ACF: std_resid^2")

    dynamic_component = np.clip(sigma2 - seasonal_component, 0, None)
    ax8 = fig.add_subplot(gs[3, :2])
    ax8.stackplot(
        df["date"],
        np.sqrt(np.clip(seasonal_component, 0, None)),
        np.sqrt(dynamic_component),
        labels=["seasonal sigma", "GARCH dynamic sigma"],
        colors=["purple", "tomato"],
        alpha=0.7,
    )
    ax8.set_title("Variance decomposition: seasonal vs GARCH dynamics")
    ax8.legend(fontsize=8)
    ax8.set_ylabel("sigma contribution")

    ax9 = fig.add_subplot(gs[3, 2])
    woy_width = (
        pd.DataFrame({"woy": df["woy"], "pi_width": pi_95_hi_full - pi_95_lo_full})
        .groupby("woy", as_index=True)["pi_width"]
        .mean()
    )
    ax9.plot(woy_width.index, woy_width.values, color="purple", lw=2)
    ax9.axhline(woy_width.mean(), color="gray", ls="--", lw=1)
    for cx in crossings:
        ax9.axvline(cx, color="tomato", ls=":", lw=1, alpha=0.7)
    ax9.set_title("95% PI width by week-of-year")
    ax9.set_xlabel("week of year")
    ax9.set_ylabel("interval width")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    history_path = resolve_repo_path(args.history)
    poi_path = resolve_repo_path(args.poi)
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, weekday_flag = load_model_frame(history_path, poi_path, start_date=args.start_date)
    cutoff_start_date = args.start_date or df["date"].min().date().isoformat()

    X_mean = sm.add_constant(df[["t", weekday_flag]])
    ols = sm.OLS(df["poi"], X_mean).fit()
    df["mean_hat"] = ols.fittedvalues
    df["resid"] = ols.resid

    fourier_terms, fourier_names = build_fourier_terms(df["woy"], args.fourier_order)
    eps = df["resid"].to_numpy(dtype=float)
    best_order, best_result = fit_garch_orders(
        eps,
        fourier_terms,
        max_garch_order=args.max_garch_order,
        floor=args.floor,
    )

    params_hat = best_result.x
    beta_end = 2 + best_order
    gamma_end = beta_end + fourier_terms.shape[1]
    omega_hat = float(params_hat[0])
    alpha_hat = float(params_hat[1])
    betas_hat = params_hat[2:beta_end].astype(float)
    gamma_hat = params_hat[beta_end:gamma_end].astype(float)
    nu_hat = float(np.exp(params_hat[gamma_end]) + 2.01)

    nll, sigma2 = garchx_filter(
        params_hat,
        eps,
        fourier_terms,
        garch_order=best_order,
        floor=args.floor,
    )
    sigma_hat = np.sqrt(sigma2)
    std_resid = eps / sigma_hat
    innovation_dist = fit_residual_distribution(
        std_resid[1:],
        residual_model=args.residual_model,
    )
    innovation_mean = float(innovation_dist.loc)
    df["mu_hat"] = df["mean_hat"] + (sigma_hat * innovation_mean)

    holdout = choose_holdout(len(df), args.holdout)
    pred_mean = df["mu_hat"].to_numpy(dtype=float)[-holdout:]
    actual_oos = df["poi"].to_numpy(dtype=float)[-holdout:]

    q10 = residual_ppf(innovation_dist, 0.10)
    q90 = residual_ppf(innovation_dist, 0.90)
    q025 = residual_ppf(innovation_dist, 0.025)
    q975 = residual_ppf(innovation_dist, 0.975)
    pi_80_lo_full = df["mean_hat"].to_numpy(dtype=float) + (sigma_hat * q10)
    pi_80_hi_full = df["mean_hat"].to_numpy(dtype=float) + (sigma_hat * q90)
    pi_95_lo_full = df["mean_hat"].to_numpy(dtype=float) + (sigma_hat * q025)
    pi_95_hi_full = df["mean_hat"].to_numpy(dtype=float) + (sigma_hat * q975)
    ci_80_lo = pi_80_lo_full[-holdout:]
    ci_80_hi = pi_80_hi_full[-holdout:]
    ci_95_lo = pi_95_lo_full[-holdout:]
    ci_95_hi = pi_95_hi_full[-holdout:]

    coverage_80 = float(np.mean((actual_oos >= ci_80_lo) & (actual_oos <= ci_80_hi)))
    coverage_95 = float(np.mean((actual_oos >= ci_95_lo) & (actual_oos <= ci_95_hi)))
    rmse_mid = float(np.sqrt(np.mean((actual_oos - pred_mean) ** 2)))
    mae_mid = float(np.mean(np.abs(actual_oos - pred_mean)))

    woy_range = np.arange(1, 53)
    seasonal_profile = np.exp(
        omega_hat
        + np.array(
            [
                sum(
                    gamma_hat[i]
                    * ([np.sin, np.cos][i % 2])(2 * np.pi * ((i // 2) + 1) * week / 52.0)
                    for i in range(len(gamma_hat))
                )
                for week in woy_range
            ],
            dtype=float,
        )
    ) + args.floor
    centered = seasonal_profile - seasonal_profile.mean()
    crossings = woy_range[np.where(np.diff(np.sign(centered)) != 0)[0]]
    peak_woy = int(woy_range[np.argmax(seasonal_profile)])
    trough_woy = int(woy_range[np.argmin(seasonal_profile)])
    seasonal_component = np.exp(omega_hat + fourier_terms @ gamma_hat) + args.floor

    lags = [lag for lag in (4, 8, 16) if lag < len(std_resid)]
    lb_std = acorr_ljungbox(std_resid[1:], lags=lags, return_df=True)
    lb_sq = acorr_ljungbox(std_resid[1:] ** 2, lags=lags, return_df=True)
    _, jb_p = jarque_bera(std_resid[1:])

    fitted_df = df[["date", "poi", "mean_hat", "mu_hat"]].copy()
    fitted_df["sigma_hat"] = sigma_hat
    fitted_df["std_resid"] = std_resid
    fitted_df["pi95_lo"] = pi_95_lo_full
    fitted_df["pi95_hi"] = pi_95_hi_full
    fitted_df["pi80_lo"] = pi_80_lo_full
    fitted_df["pi80_hi"] = pi_80_hi_full
    fitted_df.to_csv(out_dir / "garchx_fitted.csv", index=False)

    save_diagnostics_plot(
        df,
        point_forecast=df["mu_hat"].to_numpy(dtype=float),
        sigma2=sigma2,
        seasonal_profile=seasonal_profile,
        seasonal_component=seasonal_component,
        pi_80_lo_full=pi_80_lo_full,
        pi_80_hi_full=pi_80_hi_full,
        pi_95_lo_full=pi_95_lo_full,
        pi_95_hi_full=pi_95_hi_full,
        pred_center=pred_mean,
        actual_oos=actual_oos,
        coverage_80=coverage_80,
        coverage_95=coverage_95,
        std_resid=std_resid,
        peak_woy=peak_woy,
        trough_woy=trough_woy,
        crossings=crossings,
        out_path=out_dir / "seasonal_garchx.png",
    )

    summary = FitSummary(
        model_version=MODEL_VERSION,
        rows=len(df),
        cutoff_start_date=cutoff_start_date,
        start_date=df["date"].min().date().isoformat(),
        end_date=df["date"].max().date().isoformat(),
        weekday_flag=weekday_flag,
        fourier_order=args.fourier_order,
        holdout=holdout,
        best_garch_order=best_order,
        residual_model=args.residual_model,
        nll=float(nll),
        mean_const=float(ols.params["const"]),
        mean_trend=float(ols.params["t"]),
        mean_weekday=float(ols.params[weekday_flag]),
        alpha=alpha_hat,
        betas=[float(value) for value in betas_hat],
        nu=nu_hat,
        innovation_distribution=innovation_dist,
        variance_fourier={name: float(value) for name, value in zip(fourier_names, gamma_hat)},
        peak_week=peak_woy,
        trough_week=trough_woy,
        variance_ratio=float(seasonal_profile.max() / seasonal_profile.min()),
        rmse=rmse_mid,
        mae=mae_mid,
        coverage_80=coverage_80,
        coverage_95=coverage_95,
        ljung_box_std_pvalues={
            str(index): float(value) for index, value in lb_std["lb_pvalue"].items()
        },
        ljung_box_sq_pvalues={
            str(index): float(value) for index, value in lb_sq["lb_pvalue"].items()
        },
    )
    (out_dir / "garchx_summary.json").write_text(
        json.dumps(asdict(summary), indent=2),
        encoding="utf-8",
    )

    print("=" * 60)
    print("SEASONAL GARCH-X — Beam Variance Model")
    print("=" * 60)
    print(
        f"Rows: {len(df)}  Cutoff: {summary.cutoff_start_date}  "
        f"Date range: {summary.start_date} -> {summary.end_date}"
    )
    print(
        f"Mean equation: poi ~ t + {weekday_flag}  "
        f"(R^2={ols.rsquared:.4f}, JB p={jb_p:.4f})"
    )
    print(
        f"Selected GARCH(1,{best_order})-t  "
        f"alpha={alpha_hat:.4f}  betas={[round(float(v), 4) for v in betas_hat]}  nu={nu_hat:.2f}"
    )
    print(
        f"Innovation calibration: {args.residual_model}  "
        f"loc={innovation_dist.loc:.4f}  scale={innovation_dist.scale:.4f}"
    )
    print(
        f"Seasonality: peak week={peak_woy}, trough week={trough_woy}, "
        f"variance ratio={summary.variance_ratio:.2f}x"
    )
    print(
        f"OOS coverage: 80%={coverage_80:.3f}, 95%={coverage_95:.3f}, "
        f"RMSE={rmse_mid:.3f}, MAE={mae_mid:.3f}"
    )
    print(f"Wrote: {out_dir / 'seasonal_garchx.png'}")
    print(f"Wrote: {out_dir / 'garchx_fitted.csv'}")
    print(f"Wrote: {out_dir / 'garchx_summary.json'}")


if __name__ == "__main__":
    main()
