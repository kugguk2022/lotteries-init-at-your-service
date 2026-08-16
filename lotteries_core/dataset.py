"""Canonical dataset metadata -- provenance and staleness for the draw histories.

A history CSV on disk answers "what draws do I have?" but not "where did they come from, when were
they fetched, and is this still current?". Those are the questions that actually bite: this project's
documented results were computed on a file that had been a year stale without anything saying so, and
a prospective ledger recorded against stale history is quietly invalid -- the method never sees the
draws immediately preceding the one it is predicting.

Each history therefore gets a sidecar ``<name>.meta.json`` recording row count, date span, column
schema, a SHA-256 of the normalized content, and the fetch timestamp. :func:`describe` reads a CSV,
:func:`verify` re-derives the digest to detect drift, and :func:`staleness_days` answers "should I
refresh before I rely on this?".

The digest is taken over the *canonicalized* frame, not the raw bytes, so re-saving a file with
different line endings or column order does not read as a data change.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

#: Sidecar suffix. ``data/euromillions.csv`` -> ``data/euromillions.meta.json``.
META_SUFFIX = ".meta.json"

#: A EuroMillions history older than this is very likely missing completed draws (two draws a week).
DEFAULT_STALE_AFTER_DAYS = 7


@dataclass(frozen=True)
class DatasetMetadata:
    """Provenance for one history file."""

    path: str
    game: str
    rows: int
    first_draw: str
    last_draw: str
    columns: list[str]
    content_sha256: str
    fetched_utc: str
    source: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "DatasetMetadata":
        return cls(**json.loads(text))


def meta_path(csv_path: str | Path) -> Path:
    csv_path = Path(csv_path)
    return csv_path.with_suffix(META_SUFFIX)


def _date_column(df: pd.DataFrame) -> str:
    for column in ("draw_date", "date"):
        if column in df.columns:
            return column
    raise ValueError(f"history has no draw_date/date column; got {list(df.columns)}")


def content_digest(df: pd.DataFrame) -> str:
    """SHA-256 over the frame's canonical CSV form, insensitive to column order and line endings."""
    ordered = df.reindex(sorted(df.columns), axis=1)
    return hashlib.sha256(ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def describe(
    csv_path: str | Path,
    *,
    game: str,
    source: str = "euromillions.get_draws",
    fetched_utc: str | None = None,
) -> DatasetMetadata:
    """Read a history CSV and derive its metadata. Does not write anything."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path} contains no rows")
    date_column = _date_column(df)
    dates = pd.to_datetime(df[date_column], errors="coerce").dropna()
    if dates.empty:
        raise ValueError(f"{csv_path} has no parseable dates in {date_column!r}")
    return DatasetMetadata(
        path=csv_path.as_posix(),
        game=game,
        rows=int(len(df)),
        first_draw=str(dates.min().date()),
        last_draw=str(dates.max().date()),
        columns=list(df.columns),
        content_sha256=content_digest(df),
        fetched_utc=fetched_utc or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source=source,
    )


def write(csv_path: str | Path, *, game: str, source: str = "euromillions.get_draws") -> DatasetMetadata:
    """Derive and persist metadata beside ``csv_path``. Returns what was written."""
    meta = describe(csv_path, game=game, source=source)
    meta_path(csv_path).write_text(meta.to_json(), encoding="utf-8")
    return meta


def read(csv_path: str | Path) -> DatasetMetadata | None:
    """Load the sidecar for ``csv_path``, or ``None`` when it has never been written."""
    path = meta_path(csv_path)
    if not path.exists():
        return None
    return DatasetMetadata.from_json(path.read_text(encoding="utf-8"))


def verify(csv_path: str | Path) -> tuple[bool, str]:
    """Check the CSV still matches its recorded digest. Returns ``(ok, human-readable reason)``."""
    recorded = read(csv_path)
    if recorded is None:
        return False, f"no metadata recorded for {csv_path}"
    actual = content_digest(pd.read_csv(csv_path))
    if actual != recorded.content_sha256:
        return False, (
            f"{csv_path} content changed since metadata was written "
            f"(recorded {recorded.content_sha256[:12]}, actual {actual[:12]})"
        )
    return True, f"{csv_path} matches recorded metadata ({recorded.rows} draws to {recorded.last_draw})"


def last_draw_date(csv_path: str | Path) -> pd.Timestamp:
    """Newest draw date, preferring the sidecar and falling back to reading the CSV."""
    meta = read(csv_path)
    if meta is not None:
        return pd.Timestamp(meta.last_draw)
    df = pd.read_csv(csv_path)
    return pd.to_datetime(df[_date_column(df)], errors="coerce").max()


def staleness_days(csv_path: str | Path, *, today: str | pd.Timestamp | None = None) -> int:
    """Whole days between the newest draw in the file and ``today`` (default: today, UTC)."""
    now = pd.Timestamp(today) if today is not None else pd.Timestamp(datetime.now(timezone.utc).date())
    return int((now.normalize() - last_draw_date(csv_path).normalize()).days)


def is_stale(
    csv_path: str | Path,
    *,
    max_age_days: int = DEFAULT_STALE_AFTER_DAYS,
    today: str | pd.Timestamp | None = None,
) -> bool:
    """True when the newest draw is older than ``max_age_days`` -- refresh before relying on it."""
    return staleness_days(csv_path, today=today) > max_age_days
