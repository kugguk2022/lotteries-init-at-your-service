"""Validation of retrieved draw history against a game's declared shape.

Bounds come from the :class:`~lotteries_core.protocol.GameSpec` rather than from literals, so a
source that starts returning out-of-range numbers after a rule change fails here instead of
silently poisoning a benchmark.
"""

from __future__ import annotations

import pandas as pd

from ..protocol import GameSpec

CANONICAL_COLUMNS = [
    "draw_date",
    "ball_1",
    "ball_2",
    "ball_3",
    "ball_4",
    "ball_5",
    "star_1",
    "star_2",
]


def expected_columns(spec: GameSpec) -> list[str]:
    """Canonical column order for a game: the date, then main balls, then any bonus pool."""
    columns = ["draw_date"]
    columns += [f"ball_{index + 1}" for index in range(spec.main_k)]
    columns += [f"star_{index + 1}" for index in range(spec.star_k)]
    return columns


def column_bounds(spec: GameSpec) -> dict[str, tuple[int, int]]:
    bounds = {f"ball_{index + 1}": (1, spec.main_n) for index in range(spec.main_k)}
    bounds.update({f"star_{index + 1}": (1, spec.star_n) for index in range(spec.star_k)})
    return bounds


def _validate_numeric_bounds(frame: pd.DataFrame, bounds: dict[str, tuple[int, int]]) -> None:
    """Raise ``ValueError`` when any column falls outside its configured inclusive interval."""
    for column, (low, high) in bounds.items():
        try:
            series = pd.to_numeric(frame[column], errors="raise").astype(int)
        except KeyError as exc:
            raise ValueError(f"Missing expected column: {column}") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Column {column} must contain numeric values") from exc

        if not ((series >= low) & (series <= high)).all():
            bad = series[(series < low) | (series > high)]
            raise ValueError(f"Column {column} contains out-of-range values: {list(bad)}")


def validate_frame(frame: pd.DataFrame, spec: GameSpec | None = None) -> pd.DataFrame:
    """Validate structure and ranges, returning a copy with naive timestamps and int columns."""
    spec = spec or GameSpec.euromillions()
    required = expected_columns(spec)
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    coerced = frame.copy()
    coerced["draw_date"] = pd.to_datetime(coerced["draw_date"], utc=True).dt.tz_convert(None)

    bounds = column_bounds(spec)
    _validate_numeric_bounds(coerced, bounds)
    for column in bounds:
        numeric = pd.to_numeric(coerced[column], errors="raise")
        if not (numeric == numeric.round()).all():
            bad = numeric[numeric != numeric.round()].tolist()
            raise ValueError(f"Column {column} contains non-integer values: {bad}")
        coerced[column] = numeric.astype(int)

    main_columns = [f"ball_{index + 1}" for index in range(spec.main_k)]
    star_columns = [f"star_{index + 1}" for index in range(spec.star_k)]
    for row_index, row in coerced.iterrows():
        main = tuple(int(row[column]) for column in main_columns)
        stars = tuple(int(row[column]) for column in star_columns)
        if len(set(main)) != len(main):
            raise ValueError(f"Draw row {row_index} has duplicate main numbers: {main}")
        if stars and len(set(stars)) != len(stars):
            raise ValueError(f"Draw row {row_index} has duplicate auxiliary numbers: {stars}")
        # Sources may publish machine draw order. The benchmark treats pools as unordered, so the
        # canonical representation sorts them instead of rejecting otherwise valid history.
        for column, value in zip(main_columns, sorted(main)):
            coerced.at[row_index, column] = value
        for column, value in zip(star_columns, sorted(stars)):
            coerced.at[row_index, column] = value
    return coerced


def finalize(frame: pd.DataFrame, spec: GameSpec | None = None) -> pd.DataFrame:
    """Validate, sort by draw date, and drop duplicate draws."""
    validated = validate_frame(frame, spec)
    duplicates = validated[validated.duplicated(subset=["draw_date"], keep=False)]
    if not duplicates.empty:
        comparable = [column for column in expected_columns(spec or GameSpec.euromillions())]
        conflicts = duplicates.groupby("draw_date", dropna=False)[comparable].nunique(dropna=False)
        if (conflicts > 1).any(axis=None):
            dates = sorted({str(value.date()) for value in duplicates["draw_date"]})
            raise ValueError(f"Conflicting results for draw date(s): {dates}")
    return (
        validated.sort_values("draw_date")
        .drop_duplicates(subset=["draw_date"])
        .reset_index(drop=True)
    )
