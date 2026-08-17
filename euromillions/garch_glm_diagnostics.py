from __future__ import annotations

import argparse
import calendar
import warnings
from dataclasses import dataclass
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
from scipy.stats import probplot
from scipy.stats import t as t_dist
from statsmodels.graphics.tsaplots import plot_acf

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY = REPO_ROOT / "data" / "euromillions.csv"
DEFAULT_POI = REPO_ROOT / "outputs" / "euromillions" / "features" / "poi.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "euromillions" / "garch_glm_diagnostics"
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
        "--glm-family",
        choices=("gaussian", "poisson", "gamma", "negbin"),
        default="gaussian",
        help="Mean model family. Gaussian(identity) is the default because poi is strongly overdispersed.",
    )
    parser.add_argument(
        "--glm-link",
        choices=("identity", "log", "sqrt"),
        default=None,
        help="Optional GLM link override. Defaults to identity for gaussian and log otherwise.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional inclusive draw-date cutoff in YYYY-MM-DD for a regime-only rerun.",
    )
    return parser.parse_args()


def default_link_name(family_name: str) -> str:
    return "identity" if family_name == "gaussian" else "log"


def make_link(link_name: str):
    if link_name == "identity":
        return sm.families.links.Identity()
    if link_name == "log":
        return sm.families.links.Log()
    if link_name == "sqrt":
        return sm.families.links.Sqrt()
    raise ValueError(f"Unsupported link: {link_name}")


def estimate_nb_alpha(y: pd.Series | np.ndarray) -> float:
    arr = np.asarray(y, dtype=float)
    mean = float(arr.mean())
    var = float(arr.var())
    if mean <= 0:
        return 1.0
    return max((var - mean) / (mean**2), 1e-6)


def make_glm_family(family_name: str, link_name: str | None, y: pd.Series | np.ndarray):
    chosen_link = link_name or default_link_name(family_name)
    link = make_link(chosen_link)
    if family_name == "gaussian":
        family = sm.families.Gaussian(link=link)
    elif family_name == "poisson":
        family = sm.families.Poisson(link=link)
    elif family_name == "gamma":
        family = sm.families.Gamma(link=link)
    elif family_name == "negbin":
        family = sm.families.NegativeBinomial(alpha=estimate_nb_alpha(y), link=link)
    else:
        raise ValueError(f"Unsupported family: {family_name}")
    return family, f"{family_name}({chosen_link})"


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


# ====================== FLOORED GARCH-t (NO FOURIER IN VARIANCE) ======================
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
        if params[1] <= 0 or np.any(params[2:2+garch_order] <= 0) or params[1] + sum(params[2:2+garch_order]) >= 0.999:
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


# ====================== FULL DIAGNOSTICS PLOT ======================
def save_diagnostics_plot(
    df, *, sigma_hat, sigma2, seasonal_profile, seasonal_component,
    pred_mean, actual_oos, ci_80_lo, ci_80_hi, ci_95_lo, ci_95_hi,
    coverage_80, coverage_95, std_resid, peak_woy, trough_woy,
    crossings, out_path: Path
):
    woy_range = np.arange(1, 53)
    holdout = len(actual_oos)

    fig = plt.figure(figsize=(18, 22))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("Poisson GLM Seasonal Mean + Floored GARCH-X", fontsize=14, y=1.01)

    # Row 1: Main time series
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df["date"], df["poi"], lw=0.6, alpha=0.5, color="steelblue", label="poi")
    ax1.plot(df["date"], df["mu_hat"], lw=1.2, color="navy", label="mu_hat (Poisson GLM)")
    ax1.fill_between(df["date"], df["mu_hat"] - 1.96 * sigma_hat, df["mu_hat"] + 1.96 * sigma_hat,
                     alpha=0.25, color="tomato", label="95% conditional PI")
    ax1.fill_between(df["date"], df["mu_hat"] - 1.282 * sigma_hat, df["mu_hat"] + 1.282 * sigma_hat,
                     alpha=0.35, color="tomato", label="80% conditional PI")
    ax1.set_title("poi with Poisson GLM seasonal mean + floored GARCH-X band")
    ax1.legend(fontsize=8)
    ax1.set_ylabel("poi")

    # Row 2
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(woy_range, np.sqrt(seasonal_profile), color="purple", lw=2)
    ax2.axhline(np.sqrt(seasonal_profile.mean()), color="gray", ls="--", lw=1)
    for cx in crossings:
        ax2.axvline(cx, color="tomato", ls=":", lw=1, alpha=0.7)
    ax2.set_title("Seasonal sigma by week-of-year")
    ax2.set_xlabel("week of year")
    ax2.set_ylabel("sigma")
    ax2.annotate(f"peak wk {peak_woy}", xy=(peak_woy, np.sqrt(seasonal_profile.max())), fontsize=8, color="purple")
    ax2.annotate(f"trough wk {trough_woy}", xy=(trough_woy, np.sqrt(seasonal_profile.min())), fontsize=8, color="purple")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(df["date"], sigma_hat, lw=0.8, color="tomato")
    ax3.set_title("Conditional sigma_t over time")
    ax3.set_ylabel("sigma_t")

    ax4 = fig.add_subplot(gs[1, 2])
    idx = np.arange(holdout)
    ax4.plot(idx, actual_oos, lw=1, color="steelblue", label="actual")
    ax4.plot(idx, pred_mean, lw=1.5, color="navy", ls="--", label="mu_hat")
    ax4.fill_between(idx, ci_95_lo, ci_95_hi, alpha=0.2, color="tomato", label="95% PI")
    ax4.fill_between(idx, ci_80_lo, ci_80_hi, alpha=0.35, color="tomato", label="80% PI")
    ax4.set_title(f"OOS: 80% cov={coverage_80:.2f}, 95% cov={coverage_95:.2f}")
    ax4.legend(fontsize=7)

    # Row 3
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(df["date"], std_resid, lw=0.5, alpha=0.7, color="steelblue")
    ax5.axhline(0, color="red", lw=0.8)
    ax5.axhline(2, color="gray", ls="--", lw=0.8)
    ax5.axhline(-2, color="gray", ls="--", lw=0.8)
    ax5.set_title("Standardized residuals eps_t / sigma_t")

    ax6 = fig.add_subplot(gs[2, 1])
    probplot(std_resid[1:], dist="norm", plot=ax6)
    ax6.set_title("QQ: standardized residuals")

    ax7 = fig.add_subplot(gs[2, 2])
    max_lag = max(1, min(40, len(std_resid) - 2))
    plot_acf(std_resid[1:] ** 2, lags=max_lag, ax=ax7, zero=False)
    ax7.set_title("ACF: std_resid²")

    # Row 4
    dynamic_component = np.clip(sigma2 - seasonal_component, 0, None)
    ax8 = fig.add_subplot(gs[3, :2])
    ax8.stackplot(df["date"],
                  np.sqrt(np.clip(seasonal_component, 0, None)),
                  np.sqrt(dynamic_component),
                  labels=["seasonal sigma", "GARCH dynamic sigma"],
                  colors=["purple", "tomato"], alpha=0.7)
    ax8.set_title("Variance decomposition: seasonal vs GARCH dynamics")
    ax8.legend(fontsize=8)
    ax8.set_ylabel("sigma contribution")

    ax9 = fig.add_subplot(gs[3, 2])
    woy_width = (pd.DataFrame({"woy": df["woy"], "pi_width": 3.92 * sigma_hat})
                 .groupby("woy", as_index=True)["pi_width"].mean())
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

    fourier_arr, fourier_names = build_fourier_terms(df["woy"], args.fourier_order)

    # === POISSON GLM MEAN (with Fourier) ===
    X = sm.add_constant(pd.concat([
        df[["t", weekday_flag]],
        pd.DataFrame(fourier_arr, columns=fourier_names, index=df.index)
    ], axis=1))

    glm = sm.GLM(df["poi"], X, family=sm.families.Poisson()).fit()
    df["mu_hat"] = glm.fittedvalues
    df["resid"] = glm.resid_pearson

    eps = df["resid"].to_numpy(dtype=float)

    best_order, best_res = fit_garch(eps, args.max_garch_order, args.floor)
    params = best_res.x
    omega_hat = float(params[0])
    alpha_hat = float(params[1])
    betas_hat = params[2:2 + best_order].astype(float)
    nu_hat = float(np.exp(params[-1]) + 2.01)

    nll, sigma2 = garchx_filter(params, eps, garch_order=best_order, floor=args.floor)
    sigma_hat = np.sqrt(sigma2)
    std_resid = eps / sigma_hat

    holdout = min(args.holdout, len(df) // 5) if args.holdout >= len(df) else args.holdout
    pred_mean = df["mu_hat"].values[-holdout:]
    pred_sigma = sigma_hat[-holdout:]
    actual_oos = df["poi"].values[-holdout:]

    q80 = t_dist.ppf(0.90, nu_hat)
    q95 = t_dist.ppf(0.975, nu_hat)
    ci_80_lo = pred_mean - q80 * pred_sigma
    ci_80_hi = pred_mean + q80 * pred_sigma
    ci_95_lo = pred_mean - q95 * pred_sigma
    ci_95_hi = pred_mean + q95 * pred_sigma

    coverage_80 = float(np.mean((actual_oos >= ci_80_lo) & (actual_oos <= ci_80_hi)))
    coverage_95 = float(np.mean((actual_oos >= ci_95_lo) & (actual_oos <= ci_95_hi)))
    rmse = float(np.sqrt(np.mean((actual_oos - pred_mean)**2)))
    mae = float(np.mean(np.abs(actual_oos - pred_mean)))

    # Seasonal profile for plotting
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
    ) + args.floor

    centered = seasonal_profile - seasonal_profile.mean()
    crossings = woy_range[np.where(np.diff(np.sign(centered)) != 0)[0]]
    peak_woy = int(woy_range[np.argmax(seasonal_profile)])
    trough_woy = int(woy_range[np.argmin(seasonal_profile)])
    seasonal_component = np.exp(glm.params["const"] + fourier_arr @ glm.params[fourier_names].values) + args.floor

    # Save plot
    save_diagnostics_plot(
        df,
        sigma_hat=sigma_hat,
        sigma2=sigma2,
        seasonal_profile=seasonal_profile,
        seasonal_component=seasonal_component,
        pred_mean=pred_mean,
        actual_oos=actual_oos,
        ci_80_lo=ci_80_lo,
        ci_80_hi=ci_80_hi,
        ci_95_lo=ci_95_lo,
        ci_95_hi=ci_95_hi,
        coverage_80=coverage_80,
        coverage_95=coverage_95,
        std_resid=std_resid,
        peak_woy=peak_woy,
        trough_woy=trough_woy,
        crossings=crossings,
        out_path=out_dir / "poisson_mean_floored_garchx.png",
    )

    # Save fitted data
    fitted = df[["date", "poi", "mu_hat"]].copy()
    fitted["sigma_hat"] = sigma_hat
    fitted["std_resid"] = std_resid
    fitted["pi80_lo"] = df["mu_hat"] - 1.282 * sigma_hat
    fitted["pi80_hi"] = df["mu_hat"] + 1.282 * sigma_hat
    fitted["pi95_lo"] = df["mu_hat"] - 1.96 * sigma_hat
    fitted["pi95_hi"] = df["mu_hat"] + 1.96 * sigma_hat
    fitted.to_csv(out_dir / "poisson_garchx_fitted.csv", index=False)

    print("=" * 70)
    print("POISSON GLM SEASONAL MEAN + FLOORED GARCH-X")
    print("=" * 70)
    print(
        f"Rows: {len(df)}  | cutoff {cutoff_start_date} |  "
        f"{df['date'].min().date()} → {df['date'].max().date()}"
    )
    print(f"Mean: Poisson GLM + Fourier({args.fourier_order}) + trend + {weekday_flag}")
    print(f"GARCH(1,{best_order})-t   alpha={alpha_hat:.4f}  betas={[round(b,4) for b in betas_hat]}  nu={nu_hat:.2f}")
    print(f"OOS: RMSE={rmse:.3f}  MAE={mae:.3f}  80% cov={coverage_80:.3f}  95% cov={coverage_95:.3f}")
    print(f"Wrote: {out_dir / 'poisson_mean_floored_garchx.png'}")
    print(f"Wrote: {out_dir / 'poisson_garchx_fitted.csv'}")


if __name__ == "__main__":
    main()
