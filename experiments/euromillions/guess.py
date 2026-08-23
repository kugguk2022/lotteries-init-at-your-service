from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

import pandas as pd

_BALL_MIN, _BALL_MAX = 1, 50
_STAR_MIN, _STAR_MAX = 1, 12


def _ensure_sorted_unique(values: Sequence[int], expected_len: int, label: str) -> Tuple[int, ...]:
    numeric = tuple(int(v) for v in values)
    if len(numeric) != expected_len:
        raise ValueError(
            f"EuroMillions expects {expected_len} unique {label} values, got {len(numeric)}"
        )
    if len(set(numeric)) != expected_len:
        raise ValueError(f"{label.capitalize()} values must be unique")
    return tuple(sorted(numeric))


@dataclass(frozen=True)
class EuroMillionsGuess:
    """Immutable representation of a 5-number + 2-star ticket."""

    balls: Tuple[int, ...]
    stars: Tuple[int, ...]

    def __init__(self, balls: Sequence[int], stars: Sequence[int]):
        sorted_balls = _ensure_sorted_unique(balls, expected_len=5, label="ball")
        sorted_stars = _ensure_sorted_unique(stars, expected_len=2, label="star")
        EuroMillionsGuess._validate_ranges(sorted_balls, sorted_stars)
        object.__setattr__(self, "balls", sorted_balls)
        object.__setattr__(self, "stars", sorted_stars)

    @classmethod
    def from_string(cls, raw: str) -> "EuroMillionsGuess":
        """Parse a human-readable guess like ``\"1 2 3 4 5 + 6 7\"``."""

        sanitized = raw.replace("+", " ")
        tokens = [t for t in sanitized.replace(",", " ").split() if t]
        if len(tokens) != 7:
            raise ValueError(f"Expected 7 numbers (5 balls + 2 stars), received {len(tokens)}")

        numbers = [int(tok) for tok in tokens]
        return cls(numbers[:5], numbers[5:])

    @staticmethod
    def _validate_ranges(balls: Tuple[int, ...], stars: Tuple[int, ...]) -> None:
        if not all(_BALL_MIN <= b <= _BALL_MAX for b in balls):
            raise ValueError(f"Ball numbers must be between {_BALL_MIN} and {_BALL_MAX}")
        if not all(_STAR_MIN <= s <= _STAR_MAX for s in stars):
            raise ValueError(f"Star numbers must be between {_STAR_MIN} and {_STAR_MAX}")

    def as_dict(self) -> dict[str, Tuple[int, ...]]:
        return {"balls": self.balls, "stars": self.stars}


def evaluate_guess(
    draw: Mapping[str, int] | pd.Series, guess: EuroMillionsGuess
) -> tuple[int, int]:
    """Return (ball_hits, star_hits) for a guess against a normalized draw record."""

    draw_balls = {int(draw[f"ball_{i}"]) for i in range(1, 6)}
    draw_stars = {int(draw[f"star_{i}"]) for i in range(1, 3)}
    ball_hits = len(draw_balls.intersection(guess.balls))
    star_hits = len(draw_stars.intersection(guess.stars))
    return ball_hits, star_hits
