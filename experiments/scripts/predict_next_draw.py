from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from euromillions_agent.lotto_lab import BiasConfig, FeatureEngine, select_and_fit


def _fit_probabilities(
    values: pd.DataFrame, k: int, warmup: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """Train the logistic agent on ``values`` and return per-number probabilities."""

    if len(values) <= warmup:
        raise ValueError(f"Need more than {warmup} rows to build features (have {len(values)})")

    data = values.iloc[:, :k].astype(int)
    n_max = int(data.to_numpy().max())
    engine = FeatureEngine(n_max, k, BiasConfig())

    for t in range(warmup):
        engine.consume_draw(data.iloc[t].to_numpy())

    feature_rows, target_rows = [], []
    feature_names = None
    for t in range(warmup, len(data)):
        snapshot = engine.snapshot_features()
        cols = [c for c in snapshot.columns if c != "num"]
        if feature_names is None:
            feature_names = cols
        feature_rows.append(snapshot[cols].to_numpy())
        truth = data.iloc[t].to_numpy()
        target_rows.append(np.isin(snapshot["num"].to_numpy(), truth).astype(int))
        engine.consume_draw(truth)

    X = np.vstack(feature_rows)
    y = np.concatenate(target_rows)
    steps = len(data) - warmup
    model, _, _ = select_and_fit(
        X,
        y,
        steps,
        n_max,
        cs_grid=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0),
        n_splits=5,
    )

    next_snapshot = engine.snapshot_features()
    probs = model.predict_proba(next_snapshot[feature_names].to_numpy())[:, 1]
    nums = next_snapshot["num"].to_numpy()
    return nums, probs


def load_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"n1", "n2", "n3", "n4", "n5", "star1", "star2"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"History CSV is missing columns: {sorted(missing)}")
    return df


def main() -> None:
    csv_path = Path("data/euromillions_history.csv")
    history = load_history(csv_path)

    main_nums, main_probs = _fit_probabilities(history.iloc[:, :5], k=5)
    star_nums, star_probs = _fit_probabilities(history.iloc[:, 5:], k=2)

    top_main_idx = np.argsort(main_probs)[::-1][:5]
    top_star_idx = np.argsort(star_probs)[::-1][:2]

    print("Top 5 main numbers (probability):")
    for idx in top_main_idx:
        print(f"  {int(main_nums[idx]):2d} -> {main_probs[idx]:.4f}")

    print("\nTop 2 stars (probability):")
    for idx in top_star_idx:
        print(f"  {int(star_nums[idx]):2d} -> {star_probs[idx]:.4f}")


if __name__ == "__main__":
    main()
