import numpy as np
import pandas as pd

from euromillions.infer import generate_candidates, probability_tables, random_candidates


def _biased_history() -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2024-01-01")

    # Strong bias toward low numbers to give the sampler a signal to exploit.
    for i in range(100):
        rows.append(
            {
                "draw_date": (start + pd.Timedelta(days=i)).date().isoformat(),
                "ball_1": 1,
                "ball_2": 2,
                "ball_3": 3,
                "ball_4": 4,
                "ball_5": 5,
                "star_1": 1,
                "star_2": 2,
            }
        )

    rng = np.random.default_rng(42)
    for i in range(20):
        balls = np.sort(rng.choice(np.arange(1, 51), size=5, replace=False))
        stars = np.sort(rng.choice(np.arange(1, 13), size=2, replace=False))
        rows.append(
            {
                "draw_date": (start + pd.Timedelta(days=100 + i)).date().isoformat(),
                "ball_1": int(balls[0]),
                "ball_2": int(balls[1]),
                "ball_3": int(balls[2]),
                "ball_4": int(balls[3]),
                "ball_5": int(balls[4]),
                "star_1": int(stars[0]),
                "star_2": int(stars[1]),
            }
        )

    return pd.DataFrame(rows)


def _top_frequency_score(df: pd.DataFrame) -> float:
    top_balls = set(range(1, 6))
    top_stars = {1, 2}

    ball_hits = df[["ball_1", "ball_2", "ball_3", "ball_4", "ball_5"]].isin(top_balls).to_numpy().mean()
    star_hits = df[["star_1", "star_2"]].isin(top_stars).to_numpy().mean()
    return float(ball_hits + star_hits)


def test_probability_tables_normalize():
    history = _biased_history()
    ball_probs, star_probs = probability_tables(history, smoothing=0.5)

    assert np.isclose(ball_probs.sum(), 1.0)
    assert np.isclose(star_probs.sum(), 1.0)
    assert (ball_probs > 0).all()
    assert (star_probs > 0).all()


def test_frequency_weighting_beats_uniform_random():
    history = _biased_history()
    weighted = generate_candidates(history, n=200, smoothing=0.1, seed=123)
    uniform = random_candidates(n=200, seed=123)

    weighted_score = _top_frequency_score(weighted)
    uniform_score = _top_frequency_score(uniform)

    # On a heavily biased history, frequency-weighted sampling should concentrate on the top bins.
    assert weighted_score > uniform_score
