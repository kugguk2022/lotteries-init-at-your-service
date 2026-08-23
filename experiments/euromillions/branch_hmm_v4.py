#!/usr/bin/env python3
"""
branch_hmm_v4.py
Transparent versioned entrypoint for the current recommended BranchHMM workflow.

What changed vs v3:
- keeps v3 intact for comparison
- defaults to real EuroMillions-derived inputs
- makes the pair system and forecast method explicit in the summary
- writes versioned artifacts under outputs/euromillions/branch_hmm_v4

Forecast approach:
- latent state model: 2-state HMM over [poi, residual, phi_ratio, gcd_flag, ...]
- pair system: full21 (10 main-main + 10 main-star + 1 star-star)
- next-score model: empirical branch growth with bounded recent-support clipping
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from euromillions import branch_hmm_v3 as v3

REPO_ROOT = v3.REPO_ROOT
DEFAULT_HISTORY = v3.DEFAULT_HISTORY
DEFAULT_START_DATE = v3.DEFAULT_START_DATE
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "euromillions" / "branch_hmm_v4"


def _plot_results_v4(
    model: v3.BranchHMM,
    df: pd.DataFrame,
    forecast: dict[str, object],
    version: str,
    out_dir: Path,
    title_suffix: str = "",
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"BranchHMM v4 — {version.upper()} {title_suffix}", fontsize=14)

    ax = axes[0, 0]
    colors = ["#1f77b4", "#d62728"]
    for s in [0, 1]:
        mask = df.get("branch_label", pd.Series(0, index=df.index)) == s
        ax.scatter(
            df.index[mask],
            df["poi"][mask],
            c=colors[s],
            s=8,
            alpha=0.6,
            label=f"{'lower' if s == 0 else 'upper'} branch",
        )
    ax.scatter(
        [len(df)],
        [forecast["superlikely_poi"]],
        c="purple",
        s=120,
        marker="*",
        zorder=5,
        label=(
            f"next: {forecast['predicted_branch']}, "
            f"score={forecast['superlikely_poi']:.0f}"
        ),
    )
    ax.set_title("POI branch path + bounded next-score forecast")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(df.index, df["phi_ratio"], color="navy", lw=0.6)
    ax.axhline(0.5, color="gray", ls="--", label="default 0.5")
    ax.set_title("Normalized Euler-totient ratio (phi(x)/x)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    poi = df["poi"]
    ax.hist(
        poi[df.get("branch_label", 0) == 0],
        bins=40,
        alpha=0.5,
        density=True,
        color="blue",
        label="lower branch poi",
    )
    ax.hist(
        poi[df.get("branch_label", 0) == 1],
        bins=40,
        alpha=0.5,
        density=True,
        color="red",
        label="upper branch poi",
    )
    for s, col in enumerate(["blue", "red"]):
        branch_dist = model._branch_growth_distribution(df, s)
        support = np.asarray(branch_dist["poi_support"], dtype=float)
        probs = np.asarray(branch_dist["poi_probs"], dtype=float)
        ax.plot(support, probs, color=col, lw=2, label=f"{'lower' if s == 0 else 'upper'} empirical pmf")
    ax.axvline(
        forecast["superlikely_poi"],
        color="purple",
        ls="--",
        lw=1.4,
        label="forecast poi mode",
    )
    ax.set_title("Branch POI empirical distributions")
    ax.set_xlabel("poi")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    observed = np.array([
        float(((df["branch_label"].shift(1) == 0) & (df["branch_label"] == 1)).sum()),
        float(((df["branch_label"].shift(1) == 1) & (df["branch_label"] == 0)).sum()),
    ])
    denom = np.array([
        max(float((df["branch_label"].shift(1) == 0).sum()), 1.0),
        max(float((df["branch_label"].shift(1) == 1).sum()), 1.0),
    ])
    observed = observed / denom
    fitted = np.array([model.trans[0, 1], model.trans[1, 0]], dtype=float)
    x = np.arange(2)
    width = 0.36
    ax.bar(x - width / 2.0, observed, color=["gray", "purple"], width=width, label="observed flip share")
    ax.bar(
        x + width / 2.0,
        fitted,
        color=["#7f7f7f", "#b24bb2"],
        width=width,
        alpha=0.75,
        label="fitted HMM transition",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["lower→upper", "upper→lower"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Observed branch flips vs fitted HMM transitions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / f"branch_hmm_v4_{version}_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


class BranchHMMV4NoPruning(v3.BranchHMM_NoPruning):
    def plot_results(
        self,
        df: pd.DataFrame,
        forecast: dict[str, object],
        version: str,
        out_dir: Path,
        title_suffix: str = "",
    ) -> Path:
        return _plot_results_v4(self, df, forecast, version, out_dir, title_suffix)


class BranchHMMV4MinimalPruning(v3.BranchHMM_MinimalPruning):
    def plot_results(
        self,
        df: pd.DataFrame,
        forecast: dict[str, object],
        version: str,
        out_dir: Path,
        title_suffix: str = "",
    ) -> Path:
        return _plot_results_v4(self, df, forecast, version, out_dir, title_suffix)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BranchHMM v4 with explicit method metadata and versioned outputs."
    )
    parser.add_argument("--source", choices=("real", "synthetic"), default="real")
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--start-date", type=str, default=DEFAULT_START_DATE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--rows", type=int, default=2200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-iter", type=int, default=40)
    parser.add_argument("--history-window", type=int, default=50)
    parser.add_argument("--df", type=float, default=4.5)
    parser.add_argument("--prime-threshold", type=float, default=0.92)
    return parser.parse_args()


def _build_method_metadata(data_meta: dict[str, object]) -> dict[str, object]:
    return {
        "model_version": "branch_hmm_v4",
        "recommended_status": "current",
        "latent_state_model": "2-state HMM with multivariate-t emissions",
        "pair_system": data_meta.get("pair_system", "unknown"),
        "pair_events_per_draw": data_meta.get("pair_events_per_draw"),
        "pair_state_space": data_meta.get("pair_state_space"),
        "forecast_model": "empirical_branch_growth",
        "forecast_bounds": "branch support intersected with recent 12-draw support",
        "plot_density_model": "empirical branch POI pmf",
        "notes": [
            "v3 retained for historical comparison",
            "v4 is the transparent real-data-first entrypoint",
            "POI forecast is bounded to recent and branch support rather than using the latent-state mean",
        ],
    }


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "real":
        history_path = args.history if args.history.is_absolute() else REPO_ROOT / args.history
        data, data_meta = v3.load_real_feature_frame(history_path, args.start_date)
    else:
        data = v3.generate_synthetic_data(args.rows, seed=args.seed)
        data_meta = {
            "history_path": None,
            "effective_start_date": None,
            "rows": int(len(data)),
            "note": "synthetic debug data",
        }

    history = data.iloc[-args.history_window :]

    print("=" * 70)
    print(f"BRANCH HMM v4 — {'REAL DATA' if args.source == 'real' else 'SYNTHETIC DEBUG'}")
    print("=" * 70)

    print("\n[1/2] Fitting NO_EULER_PRUNING version...")
    model1 = BranchHMMV4NoPruning(df=args.df, random_state=args.seed)
    model1.fit(data, version="no_pruning", n_iter=args.n_iter)
    forecast1 = model1.predict_next(history, version="no_pruning")
    plot1 = model1.plot_results(
        data,
        forecast1,
        "no_pruning",
        out_dir,
        "(full features, no prime-line pruning)",
    )

    print("\n[2/2] Fitting MINIMAL_PRUNING version...")
    model2 = BranchHMMV4MinimalPruning(
        prime_threshold=args.prime_threshold,
        df=args.df,
        random_state=args.seed,
    )
    model2.fit(data, version="minimal_pruning", n_iter=args.n_iter)
    forecast2 = model2.predict_next(history, version="minimal_pruning")
    plot2 = model2.plot_results(
        data,
        forecast2,
        "minimal_pruning",
        out_dir,
        "(with near_upper_bound flag + down-weighting)",
    )

    summary = {
        "source": args.source,
        "rows": int(len(data)),
        "seed": int(args.seed),
        "n_iter": int(args.n_iter),
        "history_window": int(len(history)),
        "data_meta": data_meta,
        "method": _build_method_metadata(data_meta),
        "versions": {
            "no_pruning": {
                "forecast": forecast1,
                "plot": str(plot1),
            },
            "minimal_pruning": {
                "forecast": forecast2,
                "plot": str(plot2),
            },
        },
    }
    summary_path = out_dir / "branch_hmm_v4_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n" + "=" * 70)
    print(f"Done. Artifacts saved in {out_dir}")
    print(f"Summary: {summary_path}")
    print("=" * 70)
    return summary


def main() -> None:
    args = parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()