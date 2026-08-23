from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from euromillions.schema import validate_df
from euromillions_agent.lotto_lab import (
    BiasConfig,
    FeatureEngine,
    build_dataset,
    cls_metrics,
    evaluate_blocks,
    select_and_fit,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY = REPO_ROOT / "data" / "euromillions.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "euromillions" / "split_backtest"
RULE_CHANGE_DATE = pd.Timestamp("2016-09-27")
VALIDATION_START = pd.Timestamp("2023-01-01")
TEST_START = pd.Timestamp("2024-01-01")


@dataclass
class RuleChangeSummary:
    pre_2011_rows: int
    pre_2011_star_max: int
    from_2011_to_2016_rows: int
    from_2011_to_2016_star_max: int
    post_2016_rows: int
    post_2016_star_max: int
    first_star_12_date: str
    main_number_max_all_eras: int


@dataclass
class CleaningSummary:
    raw_rows: int
    cleaned_rows: int
    duplicate_dates_removed: int
    start_date: str
    end_date: str


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Date-split EuroMillions backtest with full-history vs post-2016 regime comparison."
    )
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--agent-splits", type=int, default=5)
    parser.add_argument("--train-start-post2016", type=str, default="2016-09-27")
    parser.add_argument("--validation-start", type=str, default="2023-01-01")
    parser.add_argument("--test-start", type=str, default="2024-01-01")
    return parser.parse_args()


def clean_history(history_path: Path) -> tuple[pd.DataFrame, CleaningSummary]:
    raw = pd.read_csv(history_path)
    validated = validate_df(raw)
    cleaned = (
        validated.sort_values("draw_date")
        .drop_duplicates(subset=["draw_date"])
        .reset_index(drop=True)
    )
    summary = CleaningSummary(
        raw_rows=int(len(raw)),
        cleaned_rows=int(len(cleaned)),
        duplicate_dates_removed=int(len(validated) - len(cleaned)),
        start_date=cleaned["draw_date"].min().date().isoformat(),
        end_date=cleaned["draw_date"].max().date().isoformat(),
    )
    return cleaned, summary


def verify_rule_change(history: pd.DataFrame) -> RuleChangeSummary:
    pre_2011 = history[history["draw_date"] <= pd.Timestamp("2011-05-10")]
    mid = history[
        (history["draw_date"] >= pd.Timestamp("2011-05-11"))
        & (history["draw_date"] <= pd.Timestamp("2016-09-26"))
    ]
    post = history[history["draw_date"] >= RULE_CHANGE_DATE]
    first_star_12 = history.loc[
        (history[["star_1", "star_2"]] == 12).any(axis=1), "draw_date"
    ].min()
    main_max = int(history[[f"ball_{idx}" for idx in range(1, 6)]].max().max())
    return RuleChangeSummary(
        pre_2011_rows=int(len(pre_2011)),
        pre_2011_star_max=int(pre_2011[["star_1", "star_2"]].max().max()),
        from_2011_to_2016_rows=int(len(mid)),
        from_2011_to_2016_star_max=int(mid[["star_1", "star_2"]].max().max()),
        post_2016_rows=int(len(post)),
        post_2016_star_max=int(post[["star_1", "star_2"]].max().max()),
        first_star_12_date=first_star_12.date().isoformat(),
        main_number_max_all_eras=main_max,
    )


def main_pick_metrics(pred: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    hits = np.array([len(set(p) & set(a)) for p, a in zip(pred, actual)], dtype=int)
    return {
        "steps": int(len(actual)),
        "mean_hits": float(hits.mean()),
        "recall_at_5": float((hits / 5.0).mean()),
        "any_hit_rate": float(np.mean(hits >= 1)),
        "exact_5_of_5_accuracy": float(np.mean(hits == 5)),
        "at_least_2_hits_rate": float(np.mean(hits >= 2)),
        "at_least_3_hits_rate": float(np.mean(hits >= 3)),
    }


def top5_frequency_picks(train_draws: np.ndarray) -> np.ndarray:
    counts = np.zeros(50, dtype=float)
    for col in range(train_draws.shape[1]):
        counts += np.bincount(train_draws[:, col], minlength=51)[1:]
    order = np.lexsort((np.arange(1, 51), -counts))
    return np.sort(order[:5] + 1)


def evaluate_frequency_baseline(
    history: pd.DataFrame,
    *,
    regime_start: pd.Timestamp | None,
    validation_start: pd.Timestamp,
    test_start: pd.Timestamp,
) -> pd.DataFrame:
    mains = history[[f"ball_{idx}" for idx in range(1, 6)]].to_numpy(dtype=int)
    dates = history["draw_date"].reset_index(drop=True)

    rows = []
    for window_name, start_date, end_mask in [
        ("validation", validation_start, (dates >= validation_start) & (dates < test_start)),
        ("test", test_start, dates >= test_start),
    ]:
        picks = []
        actual = []
        for idx, draw_date in enumerate(dates):
            if draw_date < start_date:
                continue
            if window_name == "validation" and draw_date >= test_start:
                continue
            train_mask = dates < draw_date
            if regime_start is not None:
                train_mask &= dates >= regime_start
            train_draws = mains[train_mask.to_numpy()]
            if len(train_draws) == 0:
                continue
            picks.append(top5_frequency_picks(train_draws))
            actual.append(np.sort(mains[idx]))
        actual_arr = np.vstack(actual)
        pred_arr = np.vstack(picks)
        metrics = main_pick_metrics(pred_arr, actual_arr)
        rows.append({"model": "frequency_top5", "window": window_name, **metrics})
    return pd.DataFrame(rows)


def evaluate_agent_split(
    history: pd.DataFrame,
    *,
    regime_start: pd.Timestamp,
    validation_start: pd.Timestamp,
    test_start: pd.Timestamp,
    warmup: int,
    splits: int,
) -> pd.DataFrame:
    regime_df = history[history["draw_date"] >= regime_start].reset_index(drop=True)
    main_df = regime_df[[f"ball_{idx}" for idx in range(1, 6)]]
    bias_cfg = BiasConfig()
    X, y, steps, n_numbers, _ = build_dataset(main_df, k=5, bias_cfg=bias_cfg, min_warmup=warmup)
    step_dates = regime_df["draw_date"].iloc[warmup:].reset_index(drop=True)
    if len(step_dates) != steps:
        raise ValueError("Step/date alignment failed in split backtest.")

    train_mask = step_dates < validation_start
    val_mask = (step_dates >= validation_start) & (step_dates < test_start)
    test_mask = step_dates >= test_start

    train_steps = int(train_mask.sum())
    if train_steps < splits:
        raise ValueError(
            f"Not enough training steps ({train_steps}) for {splits}-fold CV in the selected regime."
        )

    X_train = X[: train_steps * n_numbers]
    y_train = y[: train_steps * n_numbers]
    model, best_c, cv_loss = select_and_fit(
        X_train,
        y_train,
        train_steps,
        n_numbers,
        cs_grid=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0),
        n_splits=splits,
        debug=False,
    )

    rows = []
    for window_name, mask in [("validation", val_mask), ("test", test_mask)]:
        start = int(np.flatnonzero(mask.to_numpy())[0])
        stop = start + int(mask.sum())
        x_block = X[start * n_numbers : stop * n_numbers]
        y_block = y[start * n_numbers : stop * n_numbers]
        probs = model.predict_proba(x_block)[:, 1]
        metrics = cls_metrics(probs, y_block)
        hits, _ = evaluate_blocks(probs, y_block, n_numbers, top_k=5)
        actual_draws = regime_df[[f"ball_{idx}" for idx in range(1, 6)]].iloc[warmup + start : warmup + stop]
        pred_idx = np.argsort(probs.reshape(len(hits), n_numbers), axis=1)[:, ::-1][:, :5] + 1
        pick_metrics = main_pick_metrics(pred_idx, np.sort(actual_draws.to_numpy(dtype=int), axis=1))
        rows.append(
            {
                "model": "agent_logistic",
                "window": window_name,
                "regime_start": regime_start.date().isoformat(),
                "effective_train_steps": train_steps,
                "best_C": float(best_c),
                "cv_logloss": float(cv_loss),
                **metrics,
                **pick_metrics,
            }
        )
    return pd.DataFrame(rows)


def infer_frequency_baseline(history: pd.DataFrame, *, regime_start: pd.Timestamp) -> dict[str, object]:
    train = history[history["draw_date"] >= regime_start].reset_index(drop=True)
    mains = train[[f"ball_{idx}" for idx in range(1, 6)]].to_numpy(dtype=int)
    pick = top5_frequency_picks(mains)
    row: dict[str, object] = {
        "model": "frequency_top5",
        "history_end_date": train["draw_date"].max().date().isoformat(),
    }
    for idx, value in enumerate(pick, start=1):
        row[f"pred_main_{idx}"] = int(value)
    return row


def infer_agent_next_draw(
    history: pd.DataFrame,
    *,
    regime_start: pd.Timestamp,
    warmup: int,
    splits: int,
) -> dict[str, object]:
    regime_df = history[history["draw_date"] >= regime_start].reset_index(drop=True)
    main_df = regime_df[[f"ball_{idx}" for idx in range(1, 6)]]
    bias_cfg = BiasConfig()
    X, y, steps, n_numbers, _ = build_dataset(main_df, k=5, bias_cfg=bias_cfg, min_warmup=warmup)
    model, best_c, cv_loss = select_and_fit(
        X,
        y,
        steps,
        n_numbers,
        cs_grid=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0),
        n_splits=splits,
        debug=False,
    )

    eng = FeatureEngine(n_numbers, 5, bias_cfg)
    for idx in range(len(main_df)):
        eng.consume_draw(main_df.iloc[idx].to_numpy(dtype=int))
    next_feats = eng.snapshot_features()
    cols = [col for col in next_feats.columns if col != "num"]
    probs = model.predict_proba(next_feats[cols].to_numpy())[:, 1]
    pick = np.sort(np.argsort(probs)[::-1][:5] + 1)

    row: dict[str, object] = {
        "model": "agent_logistic",
        "history_end_date": regime_df["draw_date"].max().date().isoformat(),
        "effective_train_steps": int(steps),
        "best_C": float(best_c),
        "cv_logloss": float(cv_loss),
    }
    for idx, value in enumerate(pick, start=1):
        row[f"pred_main_{idx}"] = int(value)
    return row


def evaluate_regime(
    history: pd.DataFrame,
    *,
    label: str,
    regime_start: pd.Timestamp,
    validation_start: pd.Timestamp,
    test_start: pd.Timestamp,
    warmup: int,
    splits: int,
) -> pd.DataFrame:
    freq = evaluate_frequency_baseline(
        history,
        regime_start=regime_start,
        validation_start=validation_start,
        test_start=test_start,
    )
    agent = evaluate_agent_split(
        history,
        regime_start=regime_start,
        validation_start=validation_start,
        test_start=test_start,
        warmup=warmup,
        splits=splits,
    )
    combined = pd.concat([freq, agent], ignore_index=True)
    combined.insert(0, "regime", label)
    return combined


def build_report(
    cleaning: CleaningSummary, rule_change: RuleChangeSummary, results: pd.DataFrame, infer_df: pd.DataFrame
) -> str:
    test_rows = results[results["window"] == "test"].copy()
    lines = [
        "# EuroMillions Split Backtest",
        "",
        "## Cleaning",
        f"- Raw rows: {cleaning.raw_rows}.",
        f"- Cleaned rows: {cleaning.cleaned_rows}.",
        f"- Duplicate draw dates removed: {cleaning.duplicate_dates_removed}.",
        f"- Cleaned history range: {cleaning.start_date} to {cleaning.end_date}.",
        "",
        "## Rule Change Verification",
        f"- Before 11 May 2011: Lucky Stars maxed at {rule_change.pre_2011_star_max}.",
        f"- 11 May 2011 to 26 September 2016: Lucky Stars maxed at {rule_change.from_2011_to_2016_star_max}.",
        f"- From 27 September 2016: Lucky Stars maxed at {rule_change.post_2016_star_max}.",
        f"- First observed star 12 in the cleaned local history: {rule_change.first_star_12_date}.",
        f"- Main balls stayed at 1-{rule_change.main_number_max_all_eras} throughout.",
        "",
        "## Test Window (2024-2026)",
    ]
    for _, row in test_rows.sort_values(["model", "regime"]).iterrows():
        lines.append(
            f"- {row['model']} | {row['regime']}: recall@5={row['recall_at_5']:.4f}, "
            f"any-hit={row['any_hit_rate']:.4f}, exact5={row['exact_5_of_5_accuracy']:.4f}"
        )
    lines.append("")
    lines.append("## Inference Snapshot")
    for _, row in infer_df.sort_values(["model", "regime"]).iterrows():
        pick = [int(row[f"pred_main_{idx}"]) for idx in range(1, 6)]
        lines.append(
            f"- {row['model']} | {row['regime']}: next main pick {pick} using history through {row['history_end_date']}"
        )
    lines.append("")
    lines.append("## Takeaway")
    freq_test = test_rows[test_rows["model"] == "frequency_top5"].set_index("regime")
    agent_test = test_rows[test_rows["model"] == "agent_logistic"].set_index("regime")
    if {
        "full_history",
        "post_2016",
    }.issubset(freq_test.index):
        better_freq = (
            "post_2016"
            if freq_test.loc["post_2016", "recall_at_5"] > freq_test.loc["full_history", "recall_at_5"]
            else "full_history"
        )
        lines.append(f"- Frequency baseline is better on the test split with `{better_freq}` training.")
    if {"full_history", "post_2016"}.issubset(agent_test.index):
        better_agent = (
            "post_2016"
            if agent_test.loc["post_2016", "recall_at_5"] > agent_test.loc["full_history", "recall_at_5"]
            else "full_history"
        )
        lines.append(f"- Logistic agent is better on the test split with `{better_agent}` training.")
    lines.append(
        "- Because the 2016 change affected Lucky Stars, not the 1-50 main-ball pool, any improvement here is due to regime stability/recency rather than a changed main-number support."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    history_path = resolve_repo_path(args.history)
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history, cleaning = clean_history(history_path)
    rule_change = verify_rule_change(history)

    validation_start = pd.Timestamp(args.validation_start)
    test_start = pd.Timestamp(args.test_start)
    post_2016_start = pd.Timestamp(args.train_start_post2016)

    full_history_start = history["draw_date"].min()
    results = pd.concat(
        [
            evaluate_regime(
                history,
                label="full_history",
                regime_start=full_history_start,
                validation_start=validation_start,
                test_start=test_start,
                warmup=args.warmup,
                splits=args.agent_splits,
            ),
            evaluate_regime(
                history,
                label="post_2016",
                regime_start=post_2016_start,
                validation_start=validation_start,
                test_start=test_start,
                warmup=args.warmup,
                splits=args.agent_splits,
            ),
        ],
        ignore_index=True,
    )

    infer_df = pd.DataFrame(
        [
            {"regime": "full_history", **infer_frequency_baseline(history, regime_start=full_history_start)},
            {
                "regime": "full_history",
                **infer_agent_next_draw(
                    history,
                    regime_start=full_history_start,
                    warmup=args.warmup,
                    splits=args.agent_splits,
                ),
            },
            {
                "regime": "post_2016",
                **infer_frequency_baseline(history, regime_start=post_2016_start),
            },
            {
                "regime": "post_2016",
                **infer_agent_next_draw(
                    history,
                    regime_start=post_2016_start,
                    warmup=args.warmup,
                    splits=args.agent_splits,
                ),
            },
        ]
    )

    report = build_report(cleaning, rule_change, results, infer_df)

    results.to_csv(out_dir / "split_backtest_results.csv", index=False)
    infer_df.to_csv(out_dir / "split_backtest_inference.csv", index=False)
    (out_dir / "split_backtest_results.json").write_text(
        results.to_json(orient="records", indent=2), encoding="utf-8"
    )
    (out_dir / "cleaning_summary.json").write_text(
        json.dumps(asdict(cleaning), indent=2), encoding="utf-8"
    )
    (out_dir / "rule_change_summary.json").write_text(
        json.dumps(asdict(rule_change), indent=2), encoding="utf-8"
    )
    (out_dir / "split_backtest_report.md").write_text(report, encoding="utf-8")

    print("Wrote:")
    print(out_dir / "split_backtest_results.csv")
    print(out_dir / "split_backtest_inference.csv")
    print(out_dir / "cleaning_summary.json")
    print(out_dir / "split_backtest_results.json")
    print(out_dir / "rule_change_summary.json")
    print(out_dir / "split_backtest_report.md")


if __name__ == "__main__":
    main()
