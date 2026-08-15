from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .archive_source import fetch_euromillions_archive
from .lottology import LOTTOLOGY_ARCHIVE_URL, EMRow, fetch_euromillions_lottology
from .schema import validate_df

PRIMARY_URL = "https://www.merseyworld.com/euromillions/resultsArchive.php?format=csv"
SECONDARY_URL = "https://www.national-lottery.co.uk/results/euromillions/draw-history/csv"
PEDRO_URL = SECONDARY_URL
CACHE_DIR = Path(".cache/euromillions")
_MIN_ROWS_FULL_HISTORY = 300
SOURCE_CHOICES = ("merseyworld", "pedro", "archive", "lottology", "auto")


class FetchError(RuntimeError):
    """Raised when fetching remote draws fails."""


class ContentTypeError(RuntimeError):
    """Raised when the remote endpoint does not return a text-like payload."""


class NormalizationError(ValueError):
    """Raised when raw CSV cannot be normalized into the expected schema."""


@dataclass(frozen=True)
class FetchResult:
    """Container returned by fetch_and_normalize."""

    raw_csv: str
    dataframe: pd.DataFrame
    cache_path: Path


def _session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update({"User-Agent": "lotteries/0.1 (github.com/kugguk2022/lotteries)"})
    return session


def _cache_dir(custom: Path | None = None) -> Path:
    override = os.environ.get("EUROMILLIONS_CACHE_DIR")
    if custom:
        return custom
    if override:
        return Path(override)
    return CACHE_DIR


def _cache_key(url: str, params: dict[str, str], cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw = f"{url}|{json.dumps(params, sort_keys=True)}".encode()
    digest = hashlib.sha256(raw).hexdigest()[:16]
    return cache_dir / f"{digest}.csv"


def fetch_raw_csv(
    date_from: str | None = None,
    date_to: str | None = None,
    *,
    session: requests.Session | None = None,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    urls: list[str] | None = None,
    min_rows_full_history: int = _MIN_ROWS_FULL_HISTORY,
    allow_partial: bool = False,
) -> str:
    """
    Fetch CSV text from EuroMillions sources with retries + on-disk caching.

    Set ``EUROMILLIONS_CACHE_DIR`` or pass ``cache_dir`` to control cache location.
    If ``use_cache`` is False, the network is always hit.
    """

    params: dict[str, str] = {}
    if date_from:
        params["from"] = date_from
    if date_to:
        params["to"] = date_to

    candidates = urls or [PRIMARY_URL, SECONDARY_URL]
    errors: list[str] = []

    def _looks_like_numeric_row(line: str) -> bool:
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) < 7:
            return False
        try:
            for p in parts[1:]:
                float(p)
            return True
        except ValueError:
            return False

    def _looks_like_draw_csv(text: str) -> bool:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return False
        header = lines[0].lower()
        if "ball1" in header or "ball 1" in header or "lucky" in header or "star" in header:
            return True
        return _looks_like_numeric_row(lines[0])

    def _looks_complete(text: str) -> bool:
        if not _looks_like_draw_csv(text):
            return False
        if date_from or date_to or allow_partial:
            return True
        return text.count("\n") + 1 >= min_rows_full_history

    for url in candidates:
        cache_file = _cache_key(url, params, _cache_dir(cache_dir))
        if use_cache and cache_file.exists():
            cached = cache_file.read_text(encoding="utf-8")
            if _looks_complete(cached):
                return cached
            if allow_partial and _looks_like_draw_csv(cached):
                return cached

        try:
            with (session or _session()).get(url, params=params, timeout=5) as response:
                response.raise_for_status()
                if "text" not in response.headers.get("Content-Type", ""):
                    raise ContentTypeError(
                        f"Unexpected content type: {response.headers.get('Content-Type')}"
                    )
                text = response.text
                if not _looks_like_draw_csv(text):
                    errors.append(f"{url}: unexpected payload (missing draw headers)")
                    continue
                cache_file.write_text(text, encoding="utf-8")
        except requests.RequestException as exc:  # network / HTTP errors
            errors.append(f"{url}: {exc}")
            if cache_file.exists() and use_cache:
                cached = cache_file.read_text(encoding="utf-8")
                if _looks_complete(cached):
                    return cached
                if allow_partial and _looks_like_draw_csv(cached):
                    return cached
            continue

        if _looks_complete(text):
            return text
        if allow_partial and _looks_like_draw_csv(text):
            return text
        errors.append(f"{url}: insufficient rows ({text.count(chr(10)) + 1})")

    raise FetchError("Failed to fetch EuroMillions CSV; attempts: " + "; ".join(errors))


def _finalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate, sort, and deduplicate draws."""

    df = validate_df(df)
    return df.sort_values("draw_date").drop_duplicates(subset=["draw_date"]).reset_index(drop=True)


def _apply_date_filters(df: pd.DataFrame, date_from: str | None, date_to: str | None) -> pd.DataFrame:
    """Return a copy of df filtered by optional inclusive date bounds."""

    filtered = df
    if date_from:
        filtered = filtered[filtered["draw_date"] >= pd.to_datetime(date_from)]
    if date_to:
        filtered = filtered[filtered["draw_date"] <= pd.to_datetime(date_to)]
    return filtered.reset_index(drop=True)


def _em_rows_to_df(rows: list[EMRow]) -> pd.DataFrame:
    """Convert EMRow list into the normalized draw dataframe shape."""

    df = pd.DataFrame(
        [
            {
                "draw_date": row.date,
                "ball_1": row.n1,
                "ball_2": row.n2,
                "ball_3": row.n3,
                "ball_4": row.n4,
                "ball_5": row.n5,
                "star_1": row.star1,
                "star_2": row.star2,
            }
            for row in rows
        ]
    )
    return _finalize_dataframe(df)


def _normalize_headerless(csv_text: str) -> pd.DataFrame:
    """Handle header-less CSV payloads by assigning canonical columns."""

    df = pd.read_csv(StringIO(csv_text), header=None)
    if df.shape[1] < 8:
        raise NormalizationError(f"Headerless payload has too few columns: {df.shape[1]}")

    df = df.iloc[:, :8].copy()
    df.columns = ["draw_date", "ball_1", "ball_2", "ball_3", "ball_4", "ball_5", "star_1", "star_2"]
    return _finalize_dataframe(df)


def normalize(csv_text: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(StringIO(csv_text))
    except Exception as exc:
        raise NormalizationError(f"Failed to parse CSV payload: {exc}") from exc

    rename = {}
    for column in df.columns:
        norm = column.lower().strip().replace(" ", "_")
        rename[column] = norm
    df = df.rename(columns=rename)

    mapping = {
        "date": "draw_date",
        "draw_date": "draw_date",
        "drawdate": "draw_date",
        "ball1": "ball_1",
        "ball_1": "ball_1",
        "ball2": "ball_2",
        "ball_2": "ball_2",
        "ball3": "ball_3",
        "ball_3": "ball_3",
        "ball4": "ball_4",
        "ball_4": "ball_4",
        "ball5": "ball_5",
        "ball_5": "ball_5",
        "lucky_star1": "star_1",
        "lucky_star_1": "star_1",
        "star_1": "star_1",
        "lucky_star2": "star_2",
        "lucky_star_2": "star_2",
        "star_2": "star_2",
    }
    df = df.rename(columns={key: value for key, value in mapping.items() if key in df.columns})

    keep = ["draw_date", "ball_1", "ball_2", "ball_3", "ball_4", "ball_5", "star_1", "star_2"]
    missing = [col for col in keep if col not in df.columns]
    if missing:
        # Retry assuming header-less numeric payload
        return _normalize_headerless(csv_text)
    df = df[keep]

    return _finalize_dataframe(df)


def write_csv(df: pd.DataFrame, out_path: Path, append: bool) -> None:
    """Persist normalized draws to CSV, optionally merging with an existing file."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if append and out_path.exists():
        previous = pd.read_csv(out_path, parse_dates=["draw_date"])
        previous = validate_df(previous)
        df = pd.concat([previous, df], axis=0, ignore_index=True)
        df = (
            df.sort_values("draw_date").drop_duplicates(subset=["draw_date"]).reset_index(drop=True)
        )

    df.to_csv(out_path, index=False)


def _fetch_http_source(
    *,
    urls: list[str],
    date_from: str | None,
    date_to: str | None,
    out_path: Path | None,
    append: bool,
    session: requests.Session | None,
    cache_dir: Path | None,
    use_cache: bool,
    allow_partial: bool,
    min_rows_full_history: int = _MIN_ROWS_FULL_HISTORY,
) -> FetchResult:
    raw_csv = fetch_raw_csv(
        date_from=date_from,
        date_to=date_to,
        session=session,
        cache_dir=cache_dir,
        use_cache=use_cache,
        urls=urls,
        min_rows_full_history=min_rows_full_history,
        allow_partial=allow_partial,
    )
    dataframe = _apply_date_filters(normalize(raw_csv), date_from, date_to)
    cache_params = {k: v for k, v in {"from": date_from, "to": date_to}.items() if v}
    cache_path = _cache_key(urls[0], cache_params, _cache_dir(cache_dir))

    if out_path:
        write_csv(dataframe, out_path, append=append)
    return FetchResult(raw_csv=raw_csv, dataframe=dataframe, cache_path=cache_path)


def _fetch_lottology_source(
    *,
    date_from: str | None,
    date_to: str | None,
    out_path: Path | None,
    append: bool,
    session: requests.Session | None,
    cache_dir: Path | None,
    use_cache: bool,
) -> FetchResult:
    cache_params = {k: v for k, v in {"from": date_from, "to": date_to}.items() if v}
    cache_path = _cache_key(LOTTOLOGY_ARCHIVE_URL, cache_params, _cache_dir(cache_dir))

    if use_cache and cache_path.exists():
        cached_csv = cache_path.read_text(encoding="utf-8")
        dataframe = _apply_date_filters(normalize(cached_csv), date_from, date_to)
        if out_path:
            write_csv(dataframe, out_path, append=append)
        return FetchResult(raw_csv=cached_csv, dataframe=dataframe, cache_path=cache_path)

    rows = fetch_euromillions_lottology(session=session)
    dataframe = _apply_date_filters(_em_rows_to_df(rows), date_from, date_to)
    raw_csv = dataframe.to_csv(index=False)
    if use_cache:
        cache_path.write_text(raw_csv, encoding="utf-8")

    if out_path:
        write_csv(dataframe, out_path, append=append)
    return FetchResult(raw_csv=raw_csv, dataframe=dataframe, cache_path=cache_path)


def _fetch_archive_source(
    *,
    date_from: str | None,
    date_to: str | None,
    out_path: Path | None,
    append: bool,
    session: requests.Session | None,
    cache_dir: Path | None,
    use_cache: bool,
) -> FetchResult:
    cache_params = {k: v for k, v in {"from": date_from, "to": date_to}.items() if v}
    cache_path = _cache_key("archive", cache_params, _cache_dir(cache_dir))

    if use_cache and cache_path.exists():
        cached_csv = cache_path.read_text(encoding="utf-8")
        dataframe = _apply_date_filters(normalize(cached_csv), date_from, date_to)
        if out_path:
            write_csv(dataframe, out_path, append=append)
        return FetchResult(raw_csv=cached_csv, dataframe=dataframe, cache_path=cache_path)

    start_year = pd.to_datetime(date_from).year if date_from else None
    end_year = pd.to_datetime(date_to).year if date_to else None
    rows = fetch_euromillions_archive(
        session=session,
        start_year=start_year,
        end_year=end_year,
    )
    dataframe = _apply_date_filters(_em_rows_to_df(rows), date_from, date_to)
    raw_csv = dataframe.to_csv(index=False)
    if use_cache:
        cache_path.write_text(raw_csv, encoding="utf-8")

    if out_path:
        write_csv(dataframe, out_path, append=append)
    return FetchResult(raw_csv=raw_csv, dataframe=dataframe, cache_path=cache_path)


def _load_existing(path: Path, out_path: Path | None = None) -> FetchResult:
    """Load an already-normalized CSV and return a FetchResult."""

    df = pd.read_csv(path, parse_dates=["draw_date"])
    df = _finalize_dataframe(df)
    raw_csv = df.to_csv(index=False)
    if out_path and out_path != path:
        write_csv(df, out_path, append=False)
    return FetchResult(raw_csv=raw_csv, dataframe=df, cache_path=path)


def fetch_and_normalize(
    *,
    date_from: str | None = None,
    date_to: str | None = None,
    out_path: Path | None = None,
    append: bool = False,
    session: requests.Session | None = None,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    allow_partial: bool = False,
    source: str = "auto",
    allow_stale: bool = False,
    sample_path: Path | None = None,
) -> FetchResult:
    """High-level helper used by the CLI and tests."""

    if source not in SOURCE_CHOICES:
        raise ValueError(f"Unsupported source {source!r}; choose from {SOURCE_CHOICES}.")

    candidates = ["merseyworld", "pedro", "archive", "lottology"] if source == "auto" else [source]
    errors: list[str] = []

    for candidate in candidates:
        try:
            if candidate == "merseyworld":
                return _fetch_http_source(
                    urls=[PRIMARY_URL],
                    date_from=date_from,
                    date_to=date_to,
                    out_path=out_path,
                    append=append,
                    session=session,
                    cache_dir=cache_dir,
                    use_cache=use_cache,
                    allow_partial=allow_partial,
                )
            if candidate == "pedro":
                return _fetch_http_source(
                    urls=[PEDRO_URL],
                    date_from=date_from,
                    date_to=date_to,
                    out_path=out_path,
                    append=append,
                    session=session,
                    cache_dir=cache_dir,
                    use_cache=use_cache,
                    allow_partial=allow_partial,
                )
            if candidate == "lottology":
                return _fetch_lottology_source(
                    date_from=date_from,
                    date_to=date_to,
                    out_path=out_path,
                    append=append,
                    session=session,
                    cache_dir=cache_dir,
                    use_cache=use_cache,
                )
            if candidate == "archive":
                return _fetch_archive_source(
                    date_from=date_from,
                    date_to=date_to,
                    out_path=out_path,
                    append=append,
                    session=session,
                    cache_dir=cache_dir,
                    use_cache=use_cache,
                )
        except Exception as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            errors.append(f"{candidate}: {exc}")
            continue

    if allow_stale:
        fallbacks: list[Path] = []
        if out_path and out_path.exists():
            fallbacks.append(out_path)
        bundled_full = Path("euromillions/euromillions_2016_2025.csv")
        if bundled_full.exists():
            fallbacks.append(bundled_full)
        default_sample = sample_path or Path("data/examples/euromillions_sample.csv")
        if default_sample.exists():
            fallbacks.append(default_sample)

        for path in fallbacks:
            try:
                return _load_existing(path, out_path=out_path)
            except Exception as exc:
                errors.append(f"fallback {path}: {exc}")

    raise FetchError("Failed to fetch EuroMillions CSV; attempts: " + "; ".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch EuroMillions draws and write normalized CSV"
    )
    parser.add_argument("--from", dest="date_from", default=None, help="YYYY-MM-DD (inclusive)")
    parser.add_argument("--to", dest="date_to", default=None, help="YYYY-MM-DD (inclusive)")
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--append", action="store_true", help="Append/dedup to existing CSV")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Override cache directory (.cache/euromillions)",
    )
    parser.add_argument(
        "--no-cache", dest="use_cache", action="store_false", help="Bypass local cache on fetch"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress summary print")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow truncated datasets if full history is unavailable",
    )
    parser.add_argument(
        "--allow-stale",
        action="store_true",
        help="If all sources fail, reuse existing --out file or bundled sample data if available.",
    )
    parser.add_argument(
        "--source",
        choices=SOURCE_CHOICES,
        default="auto",
        help="Select draw source: merseyworld, pedro (legacy National Lottery CSV), archive (full yearly history), lottology, or auto fallback.",
    )
    args = parser.parse_args()

    result = fetch_and_normalize(
        date_from=args.date_from,
        date_to=args.date_to,
        out_path=args.out,
        append=args.append,
        cache_dir=args.cache_dir,
        use_cache=args.use_cache,
        allow_partial=args.allow_partial,
        source=args.source,
        allow_stale=args.allow_stale,
    )

    if not args.quiet:
        print(
            f"Wrote {len(result.dataframe):,} rows -> {args.out} "
            f"(cache: {_cache_dir(args.cache_dir)})"
        )


__all__ = [
    "ContentTypeError",
    "FetchError",
    "FetchResult",
    "NormalizationError",
    "fetch_and_normalize",
    "fetch_raw_csv",
    "normalize",
    "write_csv",
]


if __name__ == "__main__":
    main()
