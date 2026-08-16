from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

LotteryConfig = Dict[str, object]


LOTTERIES: List[LotteryConfig] = [
    {
        "name": "euromillions",
        "history": Path("data/euromillions.csv"),
        "fetch_cmd": [
            sys.executable,
            "-m",
            "euromillions.get_draws",
            "--out",
            "data/euromillions.csv",
            "--append",
            "--allow-stale",
        ],
        "main_cols": ["ball_1", "ball_2", "ball_3", "ball_4", "ball_5"],
        "bonus_cols": ["star_1", "star_2"],
    },
    {
        "name": "totoloto",
        "history": Path("data/totoloto.csv"),
        "fetch_cmd": [sys.executable, "totoloto/totoloto_get_draws.py", "--out", "data/totoloto.csv"],
        "main_cols": ["ball_1", "ball_2", "ball_3", "ball_4", "ball_5"],
        "bonus_cols": ["star_1"],  # dataset uses star_1 as bonus column
    },
    {
        "name": "eurodreams",
        "history": Path("data/eurodreams.csv"),
        "fetch_cmd": [
            sys.executable,
            "eurodreams/eurodreams_get_draws.py",
            "--out",
            "data/eurodreams.csv",
            "--allow-stale",
            "--quiet",
        ],
        "main_cols": ["n1", "n2", "n3", "n4", "n5", "n6"],
        "bonus_cols": ["dream"],
    },
]


def run_fetch(cmd: Sequence[str], *, dry_run: bool, quiet: bool) -> bool:
    if dry_run:
        if not quiet:
            print(f"[skip fetch] {' '.join(cmd)}")
        return True
    if not quiet:
        print(f"[fetch] {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        if not quiet:
            print(f"[warn] fetch failed ({exc}); will try to reuse existing data if present.")
        return False


def _counts(values: pd.Series, pop_size: int) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").dropna().astype(int).to_numpy()
    counts = np.bincount(arr, minlength=pop_size + 1)
    return counts[1 : pop_size + 1].astype(float)


def frequency_tables(
    df: pd.DataFrame,
    main_cols: Sequence[str],
    bonus_cols: Sequence[str],
    smoothing: float,
) -> Tuple[np.ndarray, np.ndarray]:
    subset = df[list(main_cols) + list(bonus_cols)].copy()
    for col in list(main_cols) + list(bonus_cols):
        subset[col] = pd.to_numeric(subset[col], errors="coerce")
    subset = subset.dropna()
    if subset.empty:
        raise ValueError("No rows with complete main+bonus numbers to score.")

    main_max = int(np.nanmax(subset[main_cols].to_numpy()))
    bonus_max = int(np.nanmax(subset[bonus_cols].to_numpy()))

    main_counts = np.zeros(main_max, dtype=float)
    for col in main_cols:
        main_counts += _counts(subset[col], main_max)
    bonus_counts = np.zeros(bonus_max, dtype=float)
    for col in bonus_cols:
        bonus_counts += _counts(subset[col], bonus_max)

    main_probs = (main_counts + smoothing) / (main_counts + smoothing).sum()
    bonus_probs = (bonus_counts + smoothing) / (bonus_counts + smoothing).sum()
    return main_probs, bonus_probs


def sample_candidates(
    main_pop: np.ndarray,
    bonus_pop: np.ndarray,
    main_k: int,
    bonus_k: int,
    rng: np.random.Generator,
    main_probs: np.ndarray | None = None,
    bonus_probs: np.ndarray | None = None,
    n: int = 100,
) -> pd.DataFrame:
    rows = []
    for _ in range(n):
        mains = np.sort(
            rng.choice(main_pop, size=main_k, replace=False, p=_normalize(main_probs, main_pop.size))
        )
        bonuses = (
            np.sort(
                rng.choice(
                    bonus_pop, size=bonus_k, replace=False, p=_normalize(bonus_probs, bonus_pop.size)
                )
            )
            if bonus_k > 0
            else np.array([])
        )
        rows.append({"mains": mains, "bonus": bonuses})
    return pd.DataFrame(rows)


def _normalize(probs: np.ndarray | None, size: int) -> np.ndarray | None:
    if probs is None:
        return None
    probs = np.asarray(probs, dtype=float)
    if probs.shape[0] != size:
        raise ValueError(f"probability length {probs.shape[0]} != population size {size}")
    probs = probs / probs.sum()
    return probs


def score_candidates(
    candidates: pd.DataFrame,
    top_main: set[int],
    top_bonus: set[int],
    main_k: int,
    bonus_k: int,
) -> np.ndarray:
    denom = main_k + bonus_k if bonus_k else main_k
    scores = []
    for _, row in candidates.iterrows():
        main_hits = sum(int(v in top_main) for v in row["mains"])
        bonus_hits = sum(int(v in top_bonus) for v in row["bonus"])
        scores.append((main_hits + bonus_hits) / denom)
    return np.asarray(scores, dtype=float)


def score_candidates_against_draw(
    candidates: pd.DataFrame,
    actual_main: set[int],
    actual_bonus: set[int],
    main_k: int,
    bonus_k: int,
) -> np.ndarray:
    denom = main_k + bonus_k if bonus_k else main_k
    scores = []
    for _, row in candidates.iterrows():
        main_hits = sum(int(v in actual_main) for v in row["mains"])
        bonus_hits = sum(int(v in actual_bonus) for v in row["bonus"])
        scores.append((main_hits + bonus_hits) / denom)
    return np.asarray(scores, dtype=float)


def permutation_pvalue(a: np.ndarray, b: np.ndarray, iters: int, rng: np.random.Generator) -> float:
    combined = np.concatenate([a, b])
    n = len(a)
    observed = a.mean() - b.mean()
    count = 0
    for _ in range(iters):
        rng.shuffle(combined)
        diff = combined[:n].mean() - combined[n:].mean()
        if diff >= observed:
            count += 1
    return (count + 1) / (iters + 1)


def evaluate_walk_forward(
    df: pd.DataFrame,
    main_cols: Sequence[str],
    bonus_cols: Sequence[str],
    *,
    smoothing: float,
    n_candidates: int,
    perm_iters: int,
    seed: int | None,
    test_frac: float,
    min_train_rows: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)

    subset = df[list(main_cols) + list(bonus_cols)].copy()
    for col in list(main_cols) + list(bonus_cols):
        subset[col] = pd.to_numeric(subset[col], errors="coerce")
    subset = subset.dropna().reset_index(drop=True)
    if subset.empty:
        raise ValueError("No rows with complete main+bonus numbers to score.")

    main_k = len(main_cols)
    bonus_k = len(bonus_cols)
    total_rows = len(subset)
    split_idx = max(min_train_rows, int(round((1.0 - test_frac) * total_rows)))
    split_idx = min(split_idx, total_rows - 1)
    split_idx = max(split_idx, 2)

    if total_rows - split_idx < 1:
        raise ValueError("Not enough rows for a forward holdout.")

    freq_step_scores = []
    rand_step_scores = []
    for t in range(split_idx, total_rows):
        train_df = subset.iloc[:t]
        actual = subset.iloc[t]
        main_probs, bonus_probs = frequency_tables(train_df, main_cols, bonus_cols, smoothing)
        main_pop = np.arange(1, len(main_probs) + 1)
        bonus_pop = np.arange(1, len(bonus_probs) + 1)

        freq_cands = sample_candidates(
            main_pop, bonus_pop, main_k, bonus_k, rng, main_probs, bonus_probs, n=n_candidates
        )
        rand_cands = sample_candidates(
            main_pop, bonus_pop, main_k, bonus_k, rng, None, None, n=n_candidates
        )

        actual_main = {int(actual[col]) for col in main_cols}
        actual_bonus = {int(actual[col]) for col in bonus_cols}
        freq_scores = score_candidates_against_draw(
            freq_cands, actual_main, actual_bonus, main_k, bonus_k
        )
        rand_scores = score_candidates_against_draw(
            rand_cands, actual_main, actual_bonus, main_k, bonus_k
        )
        freq_step_scores.append(float(freq_scores.mean()))
        rand_step_scores.append(float(rand_scores.mean()))

    freq_arr = np.asarray(freq_step_scores, dtype=float)
    rand_arr = np.asarray(rand_step_scores, dtype=float)
    pval = permutation_pvalue(freq_arr.copy(), rand_arr.copy(), perm_iters, rng)
    return {
        "freq_mean": float(freq_arr.mean()),
        "rand_mean": float(rand_arr.mean()),
        "mean_lift": float(freq_arr.mean() - rand_arr.mean()),
        "p_value": float(pval),
        "train_rows": float(split_idx),
        "test_rows": float(total_rows - split_idx),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch lottery histories, run a walk-forward baseline inference evaluation, and report significance."
    )
    parser.add_argument("--n-candidates", type=int, default=200, help="Samples per lottery per strategy")
    parser.add_argument("--permutation-iters", type=int, default=500, help="Permutation iterations for p-value")
    parser.add_argument("--smoothing", type=float, default=1.0, help="Additive smoothing for frequencies")
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.2,
        help="Fraction of complete history reserved for forward walk-forward evaluation.",
    )
    parser.add_argument(
        "--min-train-rows",
        type=int,
        default=100,
        help="Minimum initial training rows before the forward holdout begins.",
    )
    parser.add_argument("--skip-fetch", action="store_true", help="Use existing CSVs, skip network fetch")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for reproducibility")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output")
    args = parser.parse_args()

    for cfg in LOTTERIES:
        name = cfg["name"]  # type: ignore[index]
        history: Path = cfg["history"]  # type: ignore[assignment]
        fetch_cmd: Sequence[str] = cfg["fetch_cmd"]  # type: ignore[assignment]

        if not args.quiet:
            print(f"\n== {name.upper()} ==")

        ok = run_fetch(fetch_cmd, dry_run=args.skip_fetch, quiet=args.quiet)
        if not ok and not history.exists():
            raise SystemExit(f"Missing history file after fetch failed: {history}")
        if not ok and history.exists() and not args.quiet:
            print(f"[info] using existing history at {history}")

        df = pd.read_csv(history)
        result = evaluate_walk_forward(
            df,
            cfg["main_cols"],  # type: ignore[arg-type]
            cfg["bonus_cols"],  # type: ignore[arg-type]
            smoothing=args.smoothing,
            n_candidates=args.n_candidates,
            perm_iters=args.permutation_iters,
            seed=args.seed,
            test_frac=args.test_frac,
            min_train_rows=args.min_train_rows,
        )

        if not args.quiet:
            print(
                f"freq_mean={result['freq_mean']:.3f} | rand_mean={result['rand_mean']:.3f} | "
                f"lift={result['mean_lift']:.3f} | p≈{result['p_value']:.3f} | "
                f"train={int(result['train_rows'])} test={int(result['test_rows'])}"
            )

        # Save candidates for inspection
        out_dir = Path("runs")
        out_dir.mkdir(exist_ok=True)
        main_probs, bonus_probs = frequency_tables(
            df, cfg["main_cols"], cfg["bonus_cols"], smoothing=args.smoothing  # type: ignore[arg-type]
        )
        main_pop = np.arange(1, len(main_probs) + 1)
        bonus_pop = np.arange(1, len(bonus_probs) + 1)
        freq_cands = sample_candidates(
            main_pop,
            bonus_pop,
            len(cfg["main_cols"]),  # type: ignore[arg-type]
            len(cfg["bonus_cols"]),  # type: ignore[arg-type]
            np.random.default_rng(args.seed),
            main_probs,
            bonus_probs,
            n=args.n_candidates,
        )
        freq_cands.to_csv(out_dir / f"{name}_candidates.csv", index=False)


if __name__ == "__main__":
    main()
