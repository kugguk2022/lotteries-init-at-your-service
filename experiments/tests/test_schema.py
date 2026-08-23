import pandas as pd

from euromillions.schema import validate_df


def test_validate_df_roundtrip():
    df = pd.DataFrame(
        {
            "draw_date": ["2024-01-02", "2024-01-09"],
            "ball_1": [1, 2],
            "ball_2": [3, 4],
            "ball_3": [5, 6],
            "ball_4": [7, 8],
            "ball_5": [9, 10],
            "star_1": [1, 2],
            "star_2": [3, 4],
        }
    )
    validated = validate_df(df)

    assert list(validated.columns) == [
        "draw_date",
        "ball_1",
        "ball_2",
        "ball_3",
        "ball_4",
        "ball_5",
        "star_1",
        "star_2",
    ]
    assert validated["ball_1"].min() >= 1
    assert validated["ball_5"].max() <= 50
