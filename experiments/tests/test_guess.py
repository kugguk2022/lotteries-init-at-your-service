import pandas as pd
import pytest

from euromillions.guess import EuroMillionsGuess, evaluate_guess


def test_guess_parses_and_sorts_numbers():
    guess = EuroMillionsGuess.from_string("50 1 17 22 5 + 11 2")

    assert guess.balls == (1, 5, 17, 22, 50)
    assert guess.stars == (2, 11)


@pytest.mark.parametrize(
    "balls,stars,error",
    [
        ([1, 2, 3, 4], [1, 2], "5 unique"),
        ([1, 2, 3, 4, 5, 6], [1, 2], "5 unique"),
        ([1, 1, 3, 4, 5], [1, 2], "unique"),
        ([1, 2, 3, 4, 5], [1, 1], "unique"),
        ([0, 2, 3, 4, 5], [1, 2], "between 1 and 50"),
        ([1, 2, 3, 4, 5], [0, 2], "between 1 and 12"),
    ],
)
def test_guess_validation_errors(balls, stars, error):
    with pytest.raises(ValueError, match=error):
        EuroMillionsGuess(balls, stars)


def test_evaluate_guess_counts_hits():
    draw = pd.Series(
        {
            "draw_date": "2024-01-09",
            "ball_1": 7,
            "ball_2": 18,
            "ball_3": 22,
            "ball_4": 30,
            "ball_5": 45,
            "star_1": 3,
            "star_2": 9,
        }
    )
    guess = EuroMillionsGuess(balls=[5, 18, 22, 37, 45], stars=[4, 9])

    ball_hits, star_hits = evaluate_guess(draw, guess)

    assert ball_hits == 3
    assert star_hits == 1
