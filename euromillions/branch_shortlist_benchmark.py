from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from euromillions.arithmetic_branch import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_HISTORY,
    DEFAULT_START_DATE,
    DEFAULT_THRESHOLD,
    RESIDUAL_MODEL_CHOICES,
    build_branch_frame,
    evaluate_branch_mode,
    search_branch_candidates,
)
from euromillions.diagnostics3 import (
    annotate_match_statistics,
    apply_start_date_cutoff,
    build_pair_z_diagnostics,
    find_matching_full7_guesses,
    find_nearest_full7_guesses,
    fit_pruned_phi_glm,
    load_history,
    shortlist_matches,
)
from euromillions_agent.phase2_sobol import euler_phi_upto

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "euromillions" / "branch_shortlist_benchmark"
DEFAULT_HOLDOUT = 8
DEFAULT_TOP_N = 25


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark arithmetic-branch super-likely bars against diagnostics3 super-likely shortlists."
    )
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--start-date", type=str, default=DEFAULT_START_DATE)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--branch-mode",
        choices=("classic", "prime-pruned"),
        default="classic",
        help="Arithmetic-branch mode used to generate branch shortlists.",
    )
    parser.add_argument(
        "--residual-model",
        choices=RESIDUAL_MODEL_CHOICES,
        default="auto",
        help="Residual density wrapped around the branch regression forecasts.",
    )
    parser.add_argument("--holdout", type=int, default=DEFAULT_HOLDOUT)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args()


def score_ticket_rows(shortlist: pd.DataFrame, actual_draw: pd.Series) -> dict[str, int]:
    if shortlist.empty:
        return {
            "shortlist_size": 0,
            "best_rank": 0,
            "best_ball_hits": 0,
            "best_star_hits": 0,
            "exact_main5": 0,
            "exact_5plus2": 0,
        }

    actual_balls = {int(actual_draw[f"ball_{idx}"]) for idx in range(1, 6)}
    actual_stars = {int(actual_draw[f"star_{idx}"]) for idx in range(1, 3)}
    best_rank = 0
    best_ball_hits = -1
    best_star_hits = -1

    for row in shortlist.itertuples(index=False):
        row_dict = row._asdict()
        ball_hits = len(actual_balls.intersection({int(row_dict[f"ball_{idx}"]) for idx in range(1, 6)}))
        star_hits = len(actual_stars.intersection({int(row_dict["star_1"]), int(row_dict["star_2"])}))
        rank = int(row_dict.get("rank", 0))
        if (ball_hits, star_hits, -rank) > (best_ball_hits, best_star_hits, -best_rank):
            best_ball_hits = int(ball_hits)
            best_star_hits = int(star_hits)
            best_rank = int(rank)

    return {
        "shortlist_size": int(len(shortlist)),
        "best_rank": int(best_rank),
        "best_ball_hits": int(best_ball_hits),
        "best_star_hits": int(best_star_hits),
        "exact_main5": int(best_ball_hits == 5),
        "exact_5plus2": int(best_ball_hits == 5 and best_star_hits == 2),
    }


def align_shortlists(
    branch_shortlist: pd.DataFrame,
    diagnostics_shortlist: pd.DataFrame,
    *,
    requested_top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    common_n = min(int(requested_top_n), len(branch_shortlist), len(diagnostics_shortlist))
    if common_n <= 0:
        return branch_shortlist.head(0).copy(), diagnostics_shortlist.head(0).copy(), 0
    return (
        branch_shortlist.head(common_n).reset_index(drop=True),
        diagnostics_shortlist.head(common_n).reset_index(drop=True),
        int(common_n),
    )


def summarize_method(step_frame: pd.DataFrame) -> dict[str, float | int]:
    best_ball_hits = step_frame["best_ball_hits"].to_numpy(dtype=float)
    best_star_hits = step_frame["best_star_hits"].to_numpy(dtype=float)
    return {
        "steps": int(len(step_frame)),
        "ticket_budget": int(step_frame["shortlist_size"].sum()),
        "mean_best_ball_hits": float(best_ball_hits.mean()),
        "recall_at_5": float((best_ball_hits / 5.0).mean()),
        "mean_best_star_hits": float(best_star_hits.mean()),
        "recall_at_2_stars": float((best_star_hits / 2.0).mean()),
        "any_ball_hit_rate": float(np.mean(best_ball_hits >= 1.0)),
        "at_least_2_ball_hits_rate": float(np.mean(best_ball_hits >= 2.0)),
        "at_least_3_ball_hits_rate": float(np.mean(best_ball_hits >= 3.0)),
        "any_star_hit_rate": float(np.mean(best_star_hits >= 1.0)),
        "exact_main5_accuracy": float(step_frame["exact_main5"].mean()),
        "exact_5plus2_accuracy": float(step_frame["exact_5plus2"].mean()),
        "mean_best_rank": float(step_frame["best_rank"].replace(0, np.nan).mean()),
    }


def build_diagnostics3_shortlist(
    *,
    poi: np.ndarray,
    pair_counts: np.ndarray,
    main_n: int,
    star_n: int,
    pair_diag,
    batch_size: int,
    top_n: int,
) -> tuple[pd.DataFrame, dict[str, int | float | str]]:
    phi_full = euler_phi_upto(len(poi) + 1).astype(float)
    phi_current = phi_full[:-1]
    phi_shifted = phi_full[1:]
    _, _, fitted, shifted_pred, _, _ = fit_pruned_phi_glm(
        poi.astype(float),
        phi_current,
        phi_shifted,
        outlier_z=3.5,
    )
    current_expected_poi = float(fitted[-1])
    predicted_next_poi = float(shifted_pred[-1])
    reverse_growth_target_score = int(round((2.0 * current_expected_poi) - predicted_next_poi))

    reverse_matches, _ = find_matching_full7_guesses(
        pair_counts,
        main_n=main_n,
        star_n=star_n,
        target_score=reverse_growth_target_score,
        batch_size=batch_size,
    )
    if reverse_matches.empty:
        reverse_matches, reverse_gap = find_nearest_full7_guesses(
            pair_counts,
            main_n=main_n,
            star_n=star_n,
            target_score=reverse_growth_target_score,
            batch_size=batch_size,
        )
    else:
        reverse_gap = 0

    reverse_matches = annotate_match_statistics(reverse_matches, pair_diag)
    shortlist = shortlist_matches(reverse_matches, top_n, likely=True)
    shortlist["selection_source"] = "reverse_growth_exact" if reverse_gap == 0 else "reverse_growth_nearest"
    shortlist["reverse_target_gap"] = int(reverse_gap)
    return shortlist, {
        "target_score": int(reverse_growth_target_score),
        "reverse_target_gap": int(reverse_gap),
    }


def main() -> None:
    args = parse_args()
    history_path = resolve_repo_path(args.history)
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_history = load_history(history_path)
    history, cutoff_start_date = apply_start_date_cutoff(
        raw_history,
        mode="full7",
        start_date_arg=args.start_date,
    )

    eff_holdout = min(int(args.holdout), max(len(history) - 25, 1))
    rows: list[dict[str, object]] = []

    for end in range(len(history) - eff_holdout, len(history)):
        train_history = history.iloc[:end].reset_index(drop=True)
        actual_draw = history.iloc[end]
        branch_frame, pair_counts, poi, main_n, star_n, _ = build_branch_frame(
            train_history,
            ratio_mode="raw",
            modulus=None,
            threshold=float(args.threshold),
        )
        pair_diag = build_pair_z_diagnostics(train_history)

        branch_eval = evaluate_branch_mode(
            branch_frame,
            branch_mode=str(args.branch_mode),
            residual_model=str(args.residual_model),
        )
        _, branch_shortlist, _, branch_gap = search_branch_candidates(
            pair_counts,
            main_n=main_n,
            star_n=star_n,
            target_score=int(branch_eval["predicted_score"]),
            batch_size=int(args.batch_size),
            pair_diag=pair_diag,
            top_n=int(args.top_n),
            max_save_matches=0,
        )
        diagnostics_shortlist, diagnostics_meta = build_diagnostics3_shortlist(
            poi=poi,
            pair_counts=pair_counts,
            main_n=main_n,
            star_n=star_n,
            pair_diag=pair_diag,
            batch_size=int(args.batch_size),
            top_n=int(args.top_n),
        )
        branch_shortlist, diagnostics_shortlist, common_n = align_shortlists(
            branch_shortlist,
            diagnostics_shortlist,
            requested_top_n=int(args.top_n),
        )
        branch_score = score_ticket_rows(branch_shortlist, actual_draw)
        rows.append(
            {
                "draw_date": actual_draw["draw_date"],
                "method": f"branch_{args.branch_mode}_{args.residual_model}",
                "target_score": int(branch_eval["predicted_score"]),
                "target_gap": int(branch_gap),
                "effective_top_n": int(common_n),
                **branch_score,
            }
        )
        diagnostics_score = score_ticket_rows(diagnostics_shortlist, actual_draw)
        rows.append(
            {
                "draw_date": actual_draw["draw_date"],
                "method": "diagnostics3_super_likely",
                "target_score": int(diagnostics_meta["target_score"]),
                "target_gap": int(diagnostics_meta["reverse_target_gap"]),
                "effective_top_n": int(common_n),
                **diagnostics_score,
            }
        )

    step_frame = pd.DataFrame(rows).sort_values(["draw_date", "method"]).reset_index(drop=True)
    benchmark = {
        "history_rows": int(len(history)),
        "cutoff_start_date": cutoff_start_date,
        "holdout_steps": int(eff_holdout),
        "requested_top_n": int(args.top_n),
        "effective_top_n_min": int(step_frame["effective_top_n"].min()) if not step_frame.empty else 0,
        "effective_top_n_max": int(step_frame["effective_top_n"].max()) if not step_frame.empty else 0,
        "branch_mode": str(args.branch_mode),
        "residual_model": str(args.residual_model),
    }
    method_summaries: dict[str, dict[str, float | int]] = {}
    for method in step_frame["method"].unique().tolist():
        method_summaries[str(method)] = summarize_method(step_frame[step_frame["method"] == method].reset_index(drop=True))

    branch_key = f"branch_{args.branch_mode}_{args.residual_model}"
    benchmark["methods"] = method_summaries
    benchmark["recommended_method_by_recall_at_5"] = max(
        method_summaries,
        key=lambda key: (
            float(method_summaries[key]["recall_at_5"]),
            float(method_summaries[key]["exact_5plus2_accuracy"]),
        ),
    )
    benchmark["recommended_method_by_exact_accuracy"] = max(
        method_summaries,
        key=lambda key: float(method_summaries[key]["exact_5plus2_accuracy"]),
    )
    benchmark["comparison"] = {
        "branch_minus_diagnostics_recall_at_5": float(
            method_summaries[branch_key]["recall_at_5"]
            - method_summaries["diagnostics3_super_likely"]["recall_at_5"]
        ),
        "branch_minus_diagnostics_exact_5plus2_accuracy": float(
            method_summaries[branch_key]["exact_5plus2_accuracy"]
            - method_summaries["diagnostics3_super_likely"]["exact_5plus2_accuracy"]
        ),
        "branch_minus_diagnostics_at_least_2_ball_hits_rate": float(
            method_summaries[branch_key]["at_least_2_ball_hits_rate"]
            - method_summaries["diagnostics3_super_likely"]["at_least_2_ball_hits_rate"]
        ),
    }

    step_frame.to_csv(out_dir / "branch_shortlist_benchmark_steps.csv", index=False)
    (out_dir / "branch_shortlist_benchmark.json").write_text(
        json.dumps(benchmark, indent=2),
        encoding="utf-8",
    )

    print("SHORTLIST BENCHMARK")
    print(
        f"Rows: {len(history)}  Holdout: {eff_holdout}  requested_top_n={args.top_n}  "
        f"effective_top_n={benchmark['effective_top_n_min']}..{benchmark['effective_top_n_max']}  "
        f"branch_mode={args.branch_mode}  residual_model={args.residual_model}"
    )
    for method, summary in method_summaries.items():
        print(
            f"{method}: recall_at_5={summary['recall_at_5']:.4f}  "
            f"exact_5plus2_accuracy={summary['exact_5plus2_accuracy']:.4f}  "
            f"at_least_2_ball_hits_rate={summary['at_least_2_ball_hits_rate']:.4f}"
        )
    print(
        f"Recommended by recall@5: {benchmark['recommended_method_by_recall_at_5']}  "
        f"by exact accuracy: {benchmark['recommended_method_by_exact_accuracy']}"
    )
    print(out_dir / "branch_shortlist_benchmark_steps.csv")
    print(out_dir / "branch_shortlist_benchmark.json")


if __name__ == "__main__":
    main()
