from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

EXPECTED_COLUMNS: List[str] = [
    "draw_date",
    "ball_1",
    "ball_2",
    "ball_3",
    "ball_4",
    "ball_5",
    "star_1",
    "star_2",
]


def _validate_numeric_bounds(df: pd.DataFrame, bounds: Dict[str, Tuple[int, int]]) -> None:
    """Raise ValueError when any column falls outside its configured [lo, hi] interval."""

    for column, (lo, hi) in bounds.items():
        try:
            series = pd.to_numeric(df[column], errors="raise").astype(int)
        except KeyError as exc:
            raise ValueError(f"Missing expected column: {column}") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Column {column} must contain numeric values") from exc

        if not ((series >= lo) & (series <= hi)).all():
            bad = series[(series < lo) | (series > hi)]
            raise ValueError(f"Column {column} contains out-of-range values: {list(bad)}")


def validate_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate draw data structure and ensure timezone-naive timestamps."""

    missing = [name for name in EXPECTED_COLUMNS if name not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    coerced = df.copy()
    coerced["draw_date"] = pd.to_datetime(coerced["draw_date"], utc=True).dt.tz_convert(None)

    bounds: Dict[str, Tuple[int, int]] = {
        "ball_1": (1, 50),
        "ball_2": (1, 50),
        "ball_3": (1, 50),
        "ball_4": (1, 50),
        "ball_5": (1, 50),
        "star_1": (1, 12),
        "star_2": (1, 12),
    }
    _validate_numeric_bounds(coerced, bounds)

    # Ensure integer dtype after the bound check so downstream callers receive consistent types.
    for column in bounds:
        coerced[column] = coerced[column].astype(int)

    return coerced
