from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


# We use simple validation here instead of the rigid schema import to allow flexibility
def validate_df_flexible(df: pd.DataFrame) -> pd.DataFrame:
    if "draw_date" in df.columns:
        df["draw_date"] = pd.to_datetime(df["draw_date"])
    return df

def _counts(values: pd.Series, max_value: int) -> np.ndarray:
    # Ensure no values exceed max_value (ignores them if they do)
    valid = values[values <= max_value]
    counts = np.bincount(valid.to_numpy(), minlength=max_value + 1)[1:]
    return counts.astype(float)


#: EuroMillions is the default game for this module; other shapes are passed explicitly.
DEFAULT_MAX_BALL = 50
DEFAULT_MAX_STAR = 12
DEFAULT_BALL_K = 5
DEFAULT_STAR_K = 2


def default_columns(ball_k: int = DEFAULT_BALL_K, star_k: int = DEFAULT_STAR_K):
    """Canonical normalized column names: ``ball_1..ball_k`` and ``star_1..star_k``."""
    return (
        [f"ball_{i}" for i in range(1, ball_k + 1)],
        [f"star_{i}" for i in range(1, star_k + 1)],
    )


def probability_tables(
    history: pd.DataFrame,
    max_ball: int = DEFAULT_MAX_BALL,
    max_star: int = DEFAULT_MAX_STAR,
    ball_cols: list[str] | None = None,
    star_cols: list[str] | None = None,
    smoothing: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return smoothed probability tables for balls and stars.

    Game shape defaults to EuroMillions; pass the explicit arguments for any other game.
    """
    if ball_cols is None or star_cols is None:
        default_ball, default_star = default_columns()
        ball_cols = default_ball if ball_cols is None else ball_cols
        star_cols = default_star if star_cols is None else star_cols
    df = validate_df_flexible(history)

    ball_counts = np.zeros(max_ball, dtype=float)
    for column in ball_cols:
        if column in df.columns:
            ball_counts += _counts(df[column], max_ball)
    
    star_counts = np.zeros(max_star, dtype=float)
    for column in star_cols:
        if column in df.columns:
            star_counts += _counts(df[column], max_star)

    ball_probs = (ball_counts + smoothing) / (ball_counts + smoothing).sum()
    star_probs = (star_counts + smoothing) / (star_counts + smoothing).sum() if max_star > 0 else np.array([])
    return ball_probs, star_probs


def _sample_numbers(
    population: Sequence[int], probabilities: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    if k == 0:
        return np.array([], dtype=int)
    probabilities = probabilities / probabilities.sum()
    return np.sort(rng.choice(population, size=k, replace=False, p=probabilities))


def generate_candidates(
    history: pd.DataFrame,
    n: int,
    max_ball: int = DEFAULT_MAX_BALL,
    max_star: int = DEFAULT_MAX_STAR,
    ball_cols: list[str] | None = None,
    star_cols: list[str] | None = None,
    ball_k: int = DEFAULT_BALL_K,
    star_k: int = DEFAULT_STAR_K,
    smoothing: float = 1.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate `n` candidate tickets using frequency-weighted sampling.

    Game shape defaults to EuroMillions. On a fair draw this has no predictive edge over
    :func:`random_candidates`; it is the reference baseline, not a strategy.
    """
    if ball_cols is None or star_cols is None:
        default_ball, default_star = default_columns(ball_k, star_k)
        ball_cols = default_ball if ball_cols is None else ball_cols
        star_cols = default_star if star_cols is None else star_cols
    ball_probs, star_probs = probability_tables(
        history, max_ball, max_star, ball_cols, star_cols, smoothing=smoothing
    )
    
    rng = np.random.default_rng(seed)
    rows = []
    balls_pop = range(1, max_ball + 1)
    stars_pop = range(1, max_star + 1) if max_star > 0 else []

    for _ in range(n):
        balls = _sample_numbers(balls_pop, ball_probs, k=ball_k, rng=rng)
        stars = _sample_numbers(stars_pop, star_probs, k=star_k, rng=rng)
        
        row = {}
        for i, col in enumerate(ball_cols):
             if i < len(balls):
                 row[col] = int(balls[i])
        for i, col in enumerate(star_cols):
             if i < len(stars):
                 row[col] = int(stars[i])
        rows.append(row)
        
    return pd.DataFrame(rows)


def random_candidates(
    n: int,
    max_ball: int = DEFAULT_MAX_BALL,
    max_star: int = DEFAULT_MAX_STAR,
    ball_cols: list[str] | None = None,
    star_cols: list[str] | None = None,
    ball_k: int = DEFAULT_BALL_K,
    star_k: int = DEFAULT_STAR_K,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate `n` uniformly random tickets -- the matched control for any weighted generator.

    Same output shape as :func:`generate_candidates` so the two can be scored side by side. On a fair
    draw this is not a worse strategy than any other; it is the yardstick that says so.
    """
    if ball_cols is None or star_cols is None:
        default_ball, default_star = default_columns(ball_k, star_k)
        ball_cols = default_ball if ball_cols is None else ball_cols
        star_cols = default_star if star_cols is None else star_cols

    uniform_balls = np.ones(max_ball, dtype=float)
    uniform_stars = np.ones(max_star, dtype=float) if max_star > 0 else np.array([])
    rng = np.random.default_rng(seed)
    rows = []
    balls_pop = range(1, max_ball + 1)
    stars_pop = range(1, max_star + 1) if max_star > 0 else []

    for _ in range(n):
        balls = _sample_numbers(balls_pop, uniform_balls, k=ball_k, rng=rng)
        stars = _sample_numbers(stars_pop, uniform_stars, k=star_k, rng=rng)
        row = {}
        for i, col in enumerate(ball_cols):
            if i < len(balls):
                row[col] = int(balls[i])
        for i, col in enumerate(star_cols):
            if i < len(stars):
                row[col] = int(stars[i])
        rows.append(row)

    return pd.DataFrame(rows)


def load_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return validate_df_flexible(df)


def save_candidates(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate candidate tickets from historical draws."
    )
    parser.add_argument("--history", type=Path, required=True, help="Path to history CSV")
    parser.add_argument("--n", type=int, default=10, help="Number of candidate tickets")
    parser.add_argument("--out", type=Path, default=None, help="Optional output CSV path")
    parser.add_argument("--smoothing", type=float, default=1.0, help="Additive smoothing weight")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    
    # Configuration
    parser.add_argument("--max-ball", type=int, default=50, help="Highest number allowed for main balls")
    parser.add_argument("--max-star", type=int, default=12, help="Highest number allowed for star balls")
    parser.add_argument("--num-balls", type=int, default=5, help="Number of main balls to pick")
    parser.add_argument("--num-stars", type=int, default=2, help="Number of star balls to pick")
    
    # Optional col names override (defaults follow standard pattern)
    parser.add_argument("--ball-prefix", type=str, default="ball_", help="Prefix for ball columns (e.g. 'ball_1')")
    parser.add_argument("--star-prefix", type=str, default="star_", help="Prefix for star columns (e.g. 'star_1')")

    args = parser.parse_args()

    # Determine column names
    # infer columns if mostly standard
    ball_cols = [f"{args.ball_prefix}{i}" for i in range(1, args.num_balls + 1)]
    star_cols = [f"{args.star_prefix}{i}" for i in range(1, args.num_stars + 1)]
    
    # Some datasets might use 'n1', 'n2'...
    # We will try to detect if standard ones fail? No, better explicit or rely on user to pass correct prefix or rename in generic pipeline.
    # For now, let's also allow a "n" prefix heuristic if history has n1 but not ball_1
    
    history = load_history(args.history)
    
    # Heuristic for column names if defaults missing
    if ball_cols[0] not in history.columns:
        if "n1" in history.columns:
            ball_cols = [f"n{i}" for i in range(1, args.num_balls + 1)]
    
    if args.num_stars > 0:
        if star_cols[0] not in history.columns:
             # Try other common names. Without this, a history using an unrecognised star column
             # silently produced uniform star probabilities instead of failing.
             for prefix in ("lucky_star_", "star", "n_star_"):
                 candidate = [f"{prefix}{i}" for i in range(1, args.num_stars + 1)]
                 if candidate[0] in history.columns:
                     star_cols = candidate
                     break
             else:
                 if "dream" in history.columns and args.num_stars == 1:
                     star_cols = ["dream"]
                 else:
                     raise SystemExit(
                         f"Could not find star columns in history; tried {star_cols} and common "
                         f"alternatives. Available columns: {list(history.columns)}"
                     )

    candidates = generate_candidates(
        history, 
        n=args.n, 
        max_ball=args.max_ball,
        max_star=args.max_star,
        ball_cols=ball_cols,
        star_cols=star_cols,
        ball_k=args.num_balls,
        star_k=args.num_stars,
        smoothing=args.smoothing, 
        seed=args.seed
    )

    if args.out:
        save_candidates(candidates, args.out)
        print(f"Wrote {len(candidates)} candidates -> {args.out}")
    else:
        print(candidates.to_csv(index=False))


if __name__ == "__main__":
    main()

