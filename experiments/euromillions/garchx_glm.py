from __future__ import annotations

import argparse
import calendar
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import t as t_dist

from euromillions.garch_glm_diagnostics import save_diagnostics_plot

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY = REPO_ROOT / "data" / "euromillions.csv"
DEFAULT_POI = REPO_ROOT / "outputs" / "euromillions" / "features" / "poi.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "euromillions" / "garchx"
FLOOR = 1e-2


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


@dataclass
class FitSummary:
    rows: int
    start_date: str
    end_date: str
    weekday_flag: str
    fourier_order: int
    holdout: int
    best_garch_order: int
    nll: float
    mean_const: float
    mean_trend: float
    mean_weekday: float
    alpha: float
    betas: list[float]
    nu: float
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
    parser = argparse.ArgumentParser(description="Poisson GLM + Floored GARCH-X for EuroMillions POI")
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--poi", type=Path, default=DEFAULT_POI)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--holdout", type=int, default=52)
    parser.add_argument("--fourier-order", type=int, default=2)
    parser.add_argument("--max-garch-order", type=int, choices=(1, 2), default=2)
    parser.add_argument("--floor", type=float, default=FLOOR)
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional inclusive draw-date cutoff in YYYY-MM-DD for a regime-only rerun.",
    )
    return parser.parse_args()


def choose_weekday_flag(dow: pd.Series) -> tuple[str, np.ndarray]:
    unique = sorted(int(v) for v in pd.unique(dow))
    if len(unique) <= 1:
        return "is_weekday_flag", np.zeros(len(dow), dtype=float)
    flagged = unique[-1]
    return f"is_{calendar.day_name[flagged].lower()}", (dow.to_numpy() == flagged).astype(float)


def load_model_frame(history_path: Path, poi_path: Path, *, start_date: str | None):
    history = pd.read_csv(history_path, parse_dates=["draw_date"])
    history = history.sort_values("draw_date").drop_duplicates("draw_date").reset_index(drop=True)

    poi = pd.to_numeric(pd.read_csv(poi_path, header=None).squeeze(), errors="coerce").dropna().reset_index(drop=True)
    if len(history) != len(poi):
        raise ValueError(f"Row mismatch: history={len(history)}, poi={len(poi)}")

    df = pd.DataFrame({"date": history["draw_date"], "poi": poi})
    if start_date is not None:
        df = df[df["date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
    df["t"] = np.arange(len(df))
    df["dow"] = df["date"].dt.dayofweek.astype(int)
    df["woy"] = df["date"].dt.isocalendar().week.astype(int)
    weekday_flag, values = choose_weekday_flag(df["dow"])
    df[weekday_flag] = values
    return df, weekday_flag


def build_fourier_terms(woy: pd.Series, order: int):
    woy_arr = woy.to_numpy(dtype=float)
    cols = {}
    for h in range(1, order + 1):
        cols[f"sin_{h}"] = np.sin(2 * np.pi * h * woy_arr / 52)
        cols[f"cos_{h}"] = np.cos(2 * np.pi * h * woy_arr / 52)
    names = list(cols.keys())
    return pd.DataFrame(cols).to_numpy(dtype=float), names


# ====================== FLOORED GARCH-t (NO FOURIER) ======================
def garchx_filter(params, eps, *, garch_order, floor):
    omega = params[0]
    alpha = params[1]
    betas = params[2:2 + garch_order]
    nu = np.exp(params[-1]) + 2.01

    sigma2 = np.zeros(len(eps))
    sigma2[0] = max(np.var(eps), floor)
    log_c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))

    ll = 0.0
    for i in range(1, len(eps)):
        arch = alpha * eps[i-1]**2
        garch = sum(b * sigma2[i - lag] for lag, b in enumerate(betas, 1) if i - lag >= 0)
        sigma2[i] = max(omega + arch + garch, floor)

        z = eps[i] / np.sqrt(sigma2[i])
        ll += log_c - 0.5 * np.log(sigma2[i]) - ((nu + 1) / 2) * np.log(1 + (z**2) / (nu - 2))
    return -ll, sigma2


def make_objective(eps, garch_order, floor):
    def obj(params):
        if params[1] <= 0 or np.any(params[2:2+garch_order] <= 0):
            return 1e10
        if params[1] + sum(params[2:2+garch_order]) >= 0.999:
            return 1e10
        return garchx_filter(params, eps, garch_order=garch_order, floor=floor)[0]
    return obj


def fit_garch(eps, max_order, floor):
    best_aic = np.inf
    best_order, best_res = 1, None
    for order in range(1, max_order + 1):
        p0 = np.array([max(np.var(eps)*0.5, floor), 0.05] + [0.8/order]*order + [np.log(5.0)])
        res = minimize(make_objective(eps, order, floor), p0, method="Nelder-Mead",
                       options={"maxiter": 100000, "xatol": 1e-7, "fatol": 1e-7})
        k = 2 + order + 1
        aic = 2*k + 2*res.fun
        if aic < best_aic:
            best_aic, best_order, best_res = aic, order, res
    return best_order, best_res


def main():
    args = parse_args()
    history_path = resolve_repo_path(args.history)
    poi_path = resolve_repo_path(args.poi)
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, weekday_flag = load_model_frame(history_path, poi_path, start_date=args.start_date)
    cutoff_start_date = args.start_date or df["date"].min().date().isoformat()

    # === BUILD FOURIER FOR MEAN ===
    fourier_arr, fourier_names = build_fourier_terms(df["woy"], args.fourier_order)

    # === POISSON GLM MEAN EQUATION (with Fourier) ===
    X = sm.add_constant(
        pd.concat([
            df[["t", weekday_flag]],
            pd.DataFrame(fourier_arr, columns=fourier_names, index=df.index)
        ], axis=1)
    )

    glm = sm.GLM(df["poi"], X, family=sm.families.Poisson()).fit()
    df["mu_hat"] = glm.fittedvalues
    df["resid"] = glm.resid_pearson          # Pearson residuals for GARCH

    eps = df["resid"].to_numpy(dtype=float)

    # === GARCH on Pearson residuals (no Fourier) ===
    best_order, best_res = fit_garch(eps, args.max_garch_order, args.floor)
    params = best_res.x
    omega_hat = float(params[0])
    alpha_hat = float(params[1])
    betas_hat = params[2:2+best_order].astype(float)
    nu_hat = float(np.exp(params[-1]) + 2.01)

    nll, sigma2 = garchx_filter(params, eps, garch_order=best_order, floor=args.floor)
    sigma_hat = np.sqrt(sigma2)
    std_resid = eps / sigma_hat

    # OOS calculations (same as before)
    holdout = min(args.holdout, len(df) // 5) if args.holdout >= len(df) else args.holdout
    pred_mean = df["mu_hat"].values[-holdout:]
    pred_sigma = sigma_hat[-holdout:]
    actual = df["poi"].values[-holdout:]

    q80 = t_dist.ppf(0.90, nu_hat)
    q95 = t_dist.ppf(0.975, nu_hat)
    ci80_lo = pred_mean - q80 * pred_sigma
    ci80_hi = pred_mean + q80 * pred_sigma
    ci95_lo = pred_mean - q95 * pred_sigma
    ci95_hi = pred_mean + q95 * pred_sigma

    cov80 = np.mean((actual >= ci80_lo) & (actual <= ci80_hi))
    cov95 = np.mean((actual >= ci95_lo) & (actual <= ci95_hi))
    rmse = np.sqrt(np.mean((actual - pred_mean)**2))
    mae = np.mean(np.abs(actual - pred_mean))

    # Seasonal profile (for plotting)
    woy_range = np.arange(1, 53)
    seasonal_profile = np.exp(
        np.array(
            [
                glm.params["const"]
                + sum(
                    glm.params.get(name, 0.0)
                    * ([np.sin, np.cos][i % 2])(2 * np.pi * ((i // 2) + 1) * w / 52)
                    for i, name in enumerate(fourier_names)
                )
                for w in woy_range
            ],
            dtype=float,
        )
    )
    centered = seasonal_profile - seasonal_profile.mean()
    crossings = woy_range[np.where(np.diff(np.sign(centered)) != 0)[0]]
    peak_woy = int(woy_range[np.argmax(seasonal_profile)])
    trough_woy = int(woy_range[np.argmin(seasonal_profile)])
    seasonal_component = (
        np.exp(glm.params["const"] + fourier_arr @ glm.params[fourier_names].values) + args.floor
    )

    save_diagnostics_plot(
        df,
        sigma_hat=sigma_hat,
        sigma2=sigma2,
        seasonal_profile=seasonal_profile,
        seasonal_component=seasonal_component,
        pred_mean=pred_mean,
        actual_oos=actual,
        ci_80_lo=ci80_lo,
        ci_80_hi=ci80_hi,
        ci_95_lo=ci95_lo,
        ci_95_hi=ci95_hi,
        coverage_80=float(cov80),
        coverage_95=float(cov95),
        std_resid=std_resid,
        peak_woy=peak_woy,
        trough_woy=trough_woy,
        crossings=crossings,
        out_path=out_dir / "poisson_mean_floored_garchx.png",
    )

    print("=" * 70)
    print("POISSON GLM SEASONAL MEAN + FLOORED GARCH-X")
    print("=" * 70)
    print(
        f"Rows: {len(df)}  | cutoff {cutoff_start_date} |  "
        f"{df['date'].min().date()} → {df['date'].max().date()}"
    )
    print(f"Mean: Poisson GLM with Fourier({args.fourier_order}) + trend + {weekday_flag}")
    print(f"Selected: GARCH(1,{best_order})-t   alpha={alpha_hat:.4f}  betas={[round(b,4) for b in betas_hat]}  nu={nu_hat:.2f}")
    print(f"OOS: RMSE={rmse:.3f}  MAE={mae:.3f}  80% cov={cov80:.3f}  95% cov={cov95:.3f}")
    print(f"Output: {out_dir / 'poisson_mean_floored_garchx.png'}")

    # Save fitted data
    fitted = df[["date", "poi", "mu_hat"]].copy()
    fitted["sigma_hat"] = sigma_hat
    fitted["std_resid"] = std_resid
    fitted.to_csv(out_dir / "poisson_garchx_fitted.csv", index=False)

    # You can call your existing save_diagnostics_plot here with updated title
    # save_diagnostics_plot(..., out_path=out_dir / "poisson_mean_floored_garchx.png")


if __name__ == "__main__":
    main()
