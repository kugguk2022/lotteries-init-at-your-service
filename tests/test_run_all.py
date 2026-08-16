import numpy as np
import pandas as pd

from run_all import evaluate_walk_forward


def _persistent_history() -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2024-01-01")
    rng = np.random.default_rng(42)
    for i in range(20):
        balls = np.sort(rng.choice(np.arange(1, 51), size=5, replace=False))
        stars = np.sort(rng.choice(np.arange(1, 13), size=2, replace=False))
        rows.append(
            {
                "draw_date": (start + pd.Timedelta(days=i)).date().isoformat(),
                "ball_1": int(balls[0]),
                "ball_2": int(balls[1]),
                "ball_3": int(balls[2]),
                "ball_4": int(balls[3]),
                "ball_5": int(balls[4]),
                "star_1": int(stars[0]),
                "star_2": int(stars[1]),
            }
        )
    for i in range(80):
        rows.append(
            {
                "draw_date": (start + pd.Timedelta(days=20 + i)).date().isoformat(),
                "ball_1": 1,
                "ball_2": 2,
                "ball_3": 3,
                "ball_4": 4,
                "ball_5": 5,
                "star_1": 1,
                "star_2": 2,
            }
        )
    return pd.DataFrame(rows)


def test_walk_forward_frequency_beats_uniform_on_persistent_history():
    history = _persistent_history()
    result = evaluate_walk_forward(
        history,
        ["ball_1", "ball_2", "ball_3", "ball_4", "ball_5"],
        ["star_1", "star_2"],
        smoothing=0.1,
        n_candidates=100,
        perm_iters=100,
        seed=123,
        test_frac=0.2,
        min_train_rows=20,
    )

    assert result["freq_mean"] > result["rand_mean"]
    assert result["mean_lift"] > 0.0
    assert result["test_rows"] >= 1
