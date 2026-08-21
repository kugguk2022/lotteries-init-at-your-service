import pandas as pd

from euromillions import (
    EuroMillionsGuess,
    evaluate_guess,
    generate_candidates,
    load_history,
    normalize,
    probability_tables,
    random_candidates,
)


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "draw_date": "2026-01-01",
                "ball_1": 1,
                "ball_2": 2,
                "ball_3": 3,
                "ball_4": 4,
                "ball_5": 5,
                "star_1": 1,
                "star_2": 2,
            }
        ]
    )


def test_public_api_exports_are_callable():
    assert all(
        callable(symbol)
        for symbol in (
            EuroMillionsGuess,
            evaluate_guess,
            generate_candidates,
            load_history,
            normalize,
            probability_tables,
            random_candidates,
        )
    )


def test_public_random_candidates_are_exact_ordered_and_deterministic():
    first = random_candidates(n=8, seed=42)
    second = random_candidates(n=8, seed=42)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 8
    assert list(first.columns) == [
        "ball_1",
        "ball_2",
        "ball_3",
        "ball_4",
        "ball_5",
        "star_1",
        "star_2",
    ]
    assert first.filter(like="ball_").apply(lambda row: row.is_monotonic_increasing, axis=1).all()
    assert first.filter(like="star_").apply(lambda row: row.is_monotonic_increasing, axis=1).all()
    assert first.filter(like="ball_").isin(range(1, 51)).all().all()
    assert first.filter(like="star_").isin(range(1, 13)).all().all()


def test_public_weighted_generator_uses_backward_compatible_defaults():
    candidates = generate_candidates(_history(), n=3, seed=7)

    assert len(candidates) == 3
    assert list(candidates.columns) == [
        "ball_1",
        "ball_2",
        "ball_3",
        "ball_4",
        "ball_5",
        "star_1",
        "star_2",
    ]
