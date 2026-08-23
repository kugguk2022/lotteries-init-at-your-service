from __future__ import annotations

import pandas as pd

from lotteries_core import dataset, storage


def _draw(date: str, first: int = 1) -> dict:
    return {
        "draw_date": date,
        "ball_1": first,
        "ball_2": 2,
        "ball_3": 3,
        "ball_4": 4,
        "ball_5": 5,
        "star_1": 1,
        "star_2": 2,
    }


def test_sqlite_roundtrip_and_upsert(tmp_path):
    database = tmp_path / "lotteries.db"
    storage.write_history(database, pd.DataFrame([_draw("2026-08-01")]), game="euromillions")
    storage.write_history(
        database,
        pd.DataFrame([_draw("2026-08-01", 9), _draw("2026-08-08", 10)]),
        game="euromillions",
    )

    result = storage.read_history(database, game="euromillions")
    assert list(result["draw_date"]) == ["2026-08-01", "2026-08-08"]
    assert list(result["ball_1"]) == [9, 10]


def test_one_database_keeps_games_separate(tmp_path):
    database = tmp_path / "lotteries.db"
    storage.write_history(database, pd.DataFrame([_draw("2026-08-01")]), game="euromillions")
    other = pd.DataFrame([{"draw_date": "2026-08-02", "ball_1": 7}])
    storage.write_history(database, other, game="uk-lotto")

    assert len(storage.read_history(database, game="euromillions")) == 1
    assert storage.read_history(database, game="uk-lotto").iloc[0]["ball_1"] == 7


def test_csv_import_export_compatibility(tmp_path):
    source = tmp_path / "history.csv"
    database = tmp_path / "lotteries.db"
    exported = tmp_path / "exported.csv"
    pd.DataFrame([_draw("2026-08-01")]).to_csv(source, index=False)

    assert storage.import_csv(source, database, game="euromillions") == 1
    assert storage.export_csv(database, exported, game="euromillions") == 1
    pd.testing.assert_frame_equal(pd.read_csv(source), pd.read_csv(exported))


def test_sqlite_provenance_is_stored_inside_database(tmp_path):
    database = tmp_path / "lotteries.db"
    storage.write_history(database, pd.DataFrame([_draw("2026-08-01")]), game="euromillions")
    written = dataset.write(database, game="euromillions", source="test")

    assert dataset.read(database, game="euromillions") == written
    assert dataset.verify(database, game="euromillions")[0]
