"""Local, cloud-free storage for lottery draw histories.

SQLite is the canonical runtime format.  CSV remains supported at the boundary so existing
research scripts do not have to migrate all at once.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

DRAW_TABLE = "draws"
METADATA_TABLE = "dataset_metadata"


def is_database(path: str | Path) -> bool:
    return Path(path).suffix.lower() in {".db", ".sqlite", ".sqlite3"}


def read_history(path: str | Path, *, game: str | None = None) -> pd.DataFrame:
    """Read one history from SQLite or a legacy CSV file."""
    path = Path(path)
    if not is_database(path):
        return pd.read_csv(path)
    if game is None:
        raise ValueError("game is required when reading a multi-game SQLite database")
    with sqlite3.connect(path) as connection:
        frame = pd.read_sql_query(
            f'SELECT * FROM "{DRAW_TABLE}" WHERE game = ? ORDER BY draw_date',
            connection,
            params=(game,),
        )
    return frame.drop(columns=["game"], errors="ignore")


def write_history(path: str | Path, frame: pd.DataFrame, *, game: str) -> None:
    """Upsert a normalized history atomically at the draw-date level."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not is_database(path):
        frame.to_csv(path, index=False)
        return

    stored = frame.copy()
    if "draw_date" not in stored.columns:
        raise ValueError("history has no draw_date column")
    stored.insert(0, "game", game)
    with sqlite3.connect(path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            f'CREATE TABLE IF NOT EXISTS "{DRAW_TABLE}" '
            "(game TEXT NOT NULL, draw_date TEXT NOT NULL, PRIMARY KEY (game, draw_date))"
        )
        # SQLite cannot add a DataFrame's changing game-specific columns via to_sql when the table
        # already exists, so add any missing columns explicitly before the upsert.
        known = {row[1] for row in connection.execute(f'PRAGMA table_info("{DRAW_TABLE}")')}
        for column in stored.columns:
            if column not in known:
                connection.execute(f'ALTER TABLE "{DRAW_TABLE}" ADD COLUMN "{column}" INTEGER')
        columns = list(stored.columns)
        placeholders = ", ".join("?" for _ in columns)
        quoted = ", ".join(f'"{column}"' for column in columns)
        updates = ", ".join(
            f'"{column}" = excluded."{column}"'
            for column in columns
            if column not in {"game", "draw_date"}
        )
        sql = (
            f'INSERT INTO "{DRAW_TABLE}" ({quoted}) VALUES ({placeholders}) '
            f"ON CONFLICT(game, draw_date) DO UPDATE SET {updates}"
        )
        connection.executemany(sql, stored.itertuples(index=False, name=None))
        connection.execute(
            "CREATE INDEX IF NOT EXISTS draws_by_game_date ON draws(game, draw_date)"
        )


def import_csv(csv_path: str | Path, db_path: str | Path, *, game: str) -> int:
    """Import a legacy CSV into SQLite and return the number of imported rows."""
    frame = pd.read_csv(csv_path)
    write_history(db_path, frame, game=game)
    return len(frame)


def export_csv(db_path: str | Path, csv_path: str | Path, *, game: str) -> int:
    """Export one SQLite history for legacy tools."""
    frame = read_history(db_path, game=game)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    return len(frame)


def write_metadata(path: str | Path, *, game: str, metadata: dict) -> None:
    """Store provenance beside the draws, inside SQLite."""
    with sqlite3.connect(path) as connection:
        connection.execute(
            f'CREATE TABLE IF NOT EXISTS "{METADATA_TABLE}" '
            "(game TEXT PRIMARY KEY, metadata_json TEXT NOT NULL)"
        )
        connection.execute(
            f'INSERT INTO "{METADATA_TABLE}" (game, metadata_json) VALUES (?, ?) '
            "ON CONFLICT(game) DO UPDATE SET metadata_json = excluded.metadata_json",
            (game, json.dumps(metadata, sort_keys=True)),
        )


def read_metadata(path: str | Path, *, game: str) -> dict | None:
    """Read stored provenance, returning ``None`` for an uninitialised database."""
    if not Path(path).exists():
        return None
    with sqlite3.connect(path) as connection:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (METADATA_TABLE,)
        ).fetchone()
        if not exists:
            return None
        row = connection.execute(
            f'SELECT metadata_json FROM "{METADATA_TABLE}" WHERE game = ?', (game,)
        ).fetchone()
    return json.loads(row[0]) if row else None
