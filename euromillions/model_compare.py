from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY = REPO_ROOT / "data" / "euromillions.csv"
DEFAULT_OUTPUTS = REPO_ROOT / "outputs" / "euromillions"
DEFAULT_OUT_DIR = DEFAULT_OUTPUTS / "model_compare"
DEFAULT_POI_WINDOW = 52


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare EuroMillions model versions on forward top-5 and POI/regression metrics."
    )
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--poi-window",
        type=int,
        default=DEFAULT_POI_WINDOW,
        help="Trailing window used for the fair POI/regression comparison.",
    )
    return parser.parse_args()


def safe_float(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    if np.isnan(value) or np.isinf(value):
        return None
    return value


def metrics_for_picks(pred: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    hits = np.array([len(set(p) & set(a)) for p, a in zip(pred, actual)], dtype=int)
    return {
        "steps": len(actual),
        "mean_hits": float(hits.mean()),
        "recall_at_5": float((hits / 5.0).mean()),
        "any_hit_rate": float(np.mean(hits >= 1)),
        "exact_5_of_5_accuracy": float(np.mean(hits == 5)),
        "at_least_2_hits_rate": float(np.mean(hits >= 2)),
        "at_least_3_hits_rate": float(np.mean(hits >= 3)),
    }


def top5_freq_predict(train_draws: np.ndarray) -> np.ndarray:
    counts = np.zeros(50, dtype=float)
    for col in range(train_draws.shape[1]):
        counts += np.bincount(train_draws[:, col], minlength=51)[1:]
    order = np.lexsort((np.arange(1, 51), -counts))
    return np.sort(order[:5] + 1)


def classification_random_theory() -> dict[str, float]:
    support = np.arange(0, 6)
    pmf = hypergeom(M=50, n=5, N=5).pmf(support)
    mean_hits = float(np.sum(support * pmf))
    return {
        "mean_hits": mean_hits,
        "recall_at_5": mean_hits / 5.0,
        "any_hit_rate": float(1.0 - pmf[0]),
        "exact_5_of_5_accuracy": float(pmf[5]),
        "at_least_2_hits_rate": float(pmf[2:].sum()),
        "at_least_3_hits_rate": float(pmf[3:].sum()),
        "variance_hits": float(np.sum(((support - mean_hits) ** 2) * pmf)),
    }


def build_classification_table(history_path: Path, outputs_root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    history = pd.read_csv(history_path)
    mains = history[[f"ball_{i}" for i in range(1, 6)]].astype(int).to_numpy()

    agent_probs = pd.read_csv(
        outputs_root / "lottolab" / "agent_out" / "probs_main_test.csv", header=None
    ).to_numpy(float)
    truth_test = pd.read_csv(
        outputs_root / "lottolab" / "agent_out" / "truth_main_test.csv", header=None
    ).to_numpy(int)
    actual_test = mains[-len(truth_test) :]

    truth_from_history = np.zeros_like(truth_test)
    for idx, draw in enumerate(actual_test):
        truth_from_history[idx, draw - 1] = 1
    if not np.array_equal(truth_test, truth_from_history):
        raise ValueError("Saved agent truth_main_test.csv does not align with the current history tail.")

    agent_picks = np.argsort(agent_probs, axis=1)[:, ::-1][:, :5] + 1
    freq_test_picks = np.vstack(
        [top5_freq_predict(mains[:idx]) for idx in range(len(mains) - len(actual_test), len(mains))]
    )

    val_steps = int(0.6 * len(actual_test))
    mixer_truth = actual_test[val_steps:]
    agent_hold_picks = agent_picks[val_steps:]
    freq_hold_picks = freq_test_picks[val_steps:]
    mixer_picks = pd.read_csv(
        outputs_root / "lottolab" / "mixer_out" / "mixer_picks.csv", header=None
    ).to_numpy(int)
    if mixer_picks.shape != agent_hold_picks.shape:
        raise ValueError("Saved mixer picks do not align with the final agent holdout window.")

    rows: list[dict[str, object]] = []
    rows.append({"version": "agent_current_test349", "window": "agent_test", **metrics_for_picks(agent_picks, actual_test)})
    rows.append(
        {"version": "agent_current_hold140", "window": "mixer_holdout", **metrics_for_picks(agent_hold_picks, mixer_truth)}
    )
    rows.append(
        {"version": "mixer_current_hold140", "window": "mixer_holdout", **metrics_for_picks(mixer_picks, mixer_truth)}
    )
    rows.append(
        {"version": "freq_top5_test349", "window": "agent_test", **metrics_for_picks(freq_test_picks, actual_test)}
    )
    rows.append(
        {"version": "freq_top5_hold140", "window": "mixer_holdout", **metrics_for_picks(freq_hold_picks, mixer_truth)}
    )

    theory = classification_random_theory()
    rows.append(
        {
            "version": "random_theory",
            "window": "theory",
            "steps": None,
            "mean_hits": theory["mean_hits"],
            "recall_at_5": theory["recall_at_5"],
            "any_hit_rate": theory["any_hit_rate"],
            "exact_5_of_5_accuracy": theory["exact_5_of_5_accuracy"],
            "at_least_2_hits_rate": theory["at_least_2_hits_rate"],
            "at_least_3_hits_rate": theory["at_least_3_hits_rate"],
        }
    )

    run1_summary = json.loads((REPO_ROOT / "outputs" / "run1" / "agent_metrics.json").read_text())
    rows.append(
        {
            "version": "agent_run1_summary_only",
            "window": "unknown_saved_run",
            "steps": None,
            "mean_hits": float(run1_summary["mean_hits"]),
            "recall_at_5": float(run1_summary["mean_hits"]) / 5.0,
            "any_hit_rate": None,
            "exact_5_of_5_accuracy": None,
            "at_least_2_hits_rate": None,
            "at_least_3_hits_rate": None,
        }
    )

    table = pd.DataFrame(rows)
    table["random_mean_hits_theory"] = theory["mean_hits"]
    table["random_recall_theory"] = theory["recall_at_5"]
    table["lift_vs_random_mean_hits"] = table["mean_hits"] - theory["mean_hits"]
    table["lift_vs_random_recall"] = table["recall_at_5"] - theory["recall_at_5"]
    table["mean_hits_z_vs_random"] = np.nan
    for idx, row in table.iterrows():
        if pd.notna(row["steps"]):
            se = (theory["variance_hits"] / float(row["steps"])) ** 0.5
            table.loc[idx, "mean_hits_z_vs_random"] = (float(row["mean_hits"]) - theory["mean_hits"]) / se
    table["survives_vs_random_95"] = table["mean_hits_z_vs_random"] > 1.96

    summary = {
        "random_theory": {
            "mean_hits": theory["mean_hits"],
            "recall_at_5": theory["recall_at_5"],
            "any_hit_rate": theory["any_hit_rate"],
            "exact_5_of_5_accuracy": theory["exact_5_of_5_accuracy"],
        },
        "survivors": table.loc[table["survives_vs_random_95"], "version"].tolist(),
    }
    return table, summary


def interval_coverage(actual: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((actual >= lo) & (actual <= hi)))


def regression_metrics(actual: np.ndarray, pred: np.ndarray, prev: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    mae = float(np.mean(np.abs(actual - pred)))
    persistence_rmse = float(np.sqrt(np.mean((actual - prev) ** 2)))
    bias = float(np.mean(pred - actual))

    dir_actual = np.sign(actual - prev)
    dir_pred = np.sign(pred - prev)
    directional_accuracy = float(np.mean(dir_actual == dir_pred))
    n = len(actual)
    directional_z = float((directional_accuracy - 0.5) / (0.5 / np.sqrt(n)))

    design = np.c_[np.ones(len(pred)), pred]
    beta, *_ = np.linalg.lstsq(design, actual, rcond=None)
    return {
        "evaluation_rows": int(n),
        "rmse": rmse,
        "mae": mae,
        "persistence_rmse": persistence_rmse,
        "rmse_vs_persistence_ratio": float(rmse / persistence_rmse) if persistence_rmse > 0 else np.nan,
        "pred_minus_actual_bias": bias,
        "directional_accuracy": directional_accuracy,
        "directional_accuracy_z_vs_50": directional_z,
        "calibration_intercept": float(beta[0]),
        "calibration_slope": float(beta[1]),
    }


def build_regression_rows(outputs_root: Path, poi_window: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    interval_models = [
        (
            "garchx",
            outputs_root / "garchx" / "garchx_fitted.csv",
            outputs_root / "garchx" / "garchx_summary.json",
        ),
        (
            "garchx_alt_vol",
            outputs_root / "garchx_alternative_volatility" / "garchx_fitted.csv",
            outputs_root / "garchx_alternative_volatility" / "garchx_summary.json",
        ),
        (
            "garch_glm_diag",
            outputs_root / "garch_glm_diagnostics" / "poisson_garchx_fitted.csv",
            None,
        ),
    ]

    for version, fitted_path, summary_path in interval_models:
        df = pd.read_csv(fitted_path)
        eval_n = min(poi_window, len(df))
        if summary_path and summary_path.exists():
            summary = json.loads(summary_path.read_text())
            eval_n = min(eval_n, int(summary.get("holdout", eval_n)))
        eval_df = df.tail(eval_n).reset_index(drop=True)
        prev = df["poi"].shift(1).tail(eval_n).reset_index(drop=True).to_numpy(float)
        actual = eval_df["poi"].to_numpy(float)
        pred = eval_df["mu_hat"].to_numpy(float)

        row = {"version": version, "window": f"tail{eval_n}", **regression_metrics(actual, pred, prev)}
        row["coverage_80"] = interval_coverage(actual, eval_df["pi80_lo"].to_numpy(float), eval_df["pi80_hi"].to_numpy(float))
        row["coverage_95"] = interval_coverage(actual, eval_df["pi95_lo"].to_numpy(float), eval_df["pi95_hi"].to_numpy(float))
        row["coverage80_abs_error"] = abs(row["coverage_80"] - 0.80)
        row["coverage95_abs_error"] = abs(row["coverage_95"] - 0.95)
        rows.append(row)

    diag3 = pd.read_csv(outputs_root / "diagnostics3" / "diagnostics3_series.csv")
    actual_next = diag3["poi"].shift(-1).iloc[:-1].tail(poi_window).to_numpy(float)
    pred_next = diag3["glm_predicted_next_response"].iloc[:-1].tail(poi_window).to_numpy(float)
    prev_now = diag3["poi"].iloc[:-1].tail(poi_window).to_numpy(float)
    row = {"version": "diagnostics3_next", "window": f"tail{len(actual_next)}", **regression_metrics(actual_next, pred_next, prev_now)}
    row["coverage_80"] = np.nan
    row["coverage_95"] = np.nan
    row["coverage80_abs_error"] = np.nan
    row["coverage95_abs_error"] = np.nan
    rows.append(row)

    table = pd.DataFrame(rows)
    table["survives_min_bar"] = (
        (table["directional_accuracy_z_vs_50"] > 1.96)
        & (table["rmse_vs_persistence_ratio"] < 1.0)
    )
    has_intervals = table["coverage80_abs_error"].notna() & table["coverage95_abs_error"].notna()
    table["interval_calibration_ok"] = np.where(
        has_intervals,
        (table["coverage80_abs_error"] <= 0.10) & (table["coverage95_abs_error"] <= 0.10),
        np.nan,
    )
    return table


def build_report(classification: pd.DataFrame, regression: pd.DataFrame) -> str:
    cls_survivors = classification.loc[
        classification["survives_vs_random_95"], "version"
    ].tolist()
    reg_survivors = regression.loc[regression["survives_min_bar"], "version"].tolist()

    lines = [
        "# EuroMillions Model Comparison",
        "",
        "## Classification-style main-number predictors",
        f"- Survivors vs random at 95%: {', '.join(cls_survivors) if cls_survivors else 'none'}",
        "",
        "## POI / regression family",
        "- Survival rule: directional accuracy above 50% with z>1.96 and RMSE below persistence.",
        f"- Survivors on the common trailing window: {', '.join(reg_survivors) if reg_survivors else 'none'}",
        "",
        "## Notes",
        "- `garchx*` and `garch_glm_diagnostics` are compared on the same trailing window.",
        "- `diagnostics3_next` is cut to the same trailing window for fairness, even though it also has a longer full-history next-step series.",
        "- Exact 5/5 accuracy is zero for every directly comparable forward main-number model in the current outputs.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    history_path = resolve_repo_path(args.history)
    outputs_root = resolve_repo_path(args.outputs_root)
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classification_table, classification_summary = build_classification_table(history_path, outputs_root)
    regression_table = build_regression_rows(outputs_root, poi_window=int(args.poi_window))
    report_text = build_report(classification_table, regression_table)

    classification_table.to_csv(out_dir / "version_recall_accuracy.csv", index=False)
    regression_table.to_csv(out_dir / "version_poi_regression_compare.csv", index=False)
    (out_dir / "version_recall_accuracy.json").write_text(
        classification_table.to_json(orient="records", indent=2), encoding="utf-8"
    )
    (out_dir / "version_poi_regression_compare.json").write_text(
        regression_table.to_json(orient="records", indent=2), encoding="utf-8"
    )
    (out_dir / "classification_summary.json").write_text(
        json.dumps(classification_summary, indent=2), encoding="utf-8"
    )
    (out_dir / "model_compare_report.md").write_text(report_text, encoding="utf-8")

    print("Wrote:")
    print(out_dir / "version_recall_accuracy.csv")
    print(out_dir / "version_poi_regression_compare.csv")
    print(out_dir / "version_recall_accuracy.json")
    print(out_dir / "version_poi_regression_compare.json")
    print(out_dir / "classification_summary.json")
    print(out_dir / "model_compare_report.md")


if __name__ == "__main__":
    main()
