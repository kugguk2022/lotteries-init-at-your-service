"""EuroMillions retrieval from the Irish National Lottery and a validated history.

The official operator page publishes roughly 90 days of structured results. ``auto`` merges that
increment with a digest-pinned seed and requires at least one identical overlap draw. A changed
official result or a damaged seed therefore fails closed instead of silently rewriting benchmark
history. Repeated CLI/workflow runs may pass their last validated snapshot as the new baseline.

Retrieved payloads are cached on disk by URL so that repeated benchmark runs, tests, and offline
work do not re-hit the operator page. Historical archive adapters remain explicit diagnostic
choices; they are no longer part of the scheduled/default provider chain.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from importlib.resources import files
from io import StringIO
from pathlib import Path

import pandas as pd

from ..dataset import content_digest
from ..protocol import GameSpec
from .errors import ContentTypeError, FetchError, NormalizationError
from .schema import finalize

IRISH_OFFICIAL_URL = "https://www.lottery.ie/draw-games/results/view?game=euromillions"
PRIMARY_URL = "https://www.national-lottery.co.uk/results/euromillions/draw-history/csv"
SECONDARY_URL = "https://www.merseyworld.com/euromillions/resultsArchive.php?format=csv"
CSV_URLS = (PRIMARY_URL, SECONDARY_URL)

#: ``auto`` and ``irish-official`` use the validated-history strategy. Legacy sources remain
#: selectable for diagnosis and one-off baseline maintenance, never as silent fallbacks.
SOURCE_CHOICES = (
    "auto", "irish-official", "merseyworld", "national-lottery", "archive", "lottology",
)

CACHE_DIR = Path(".cache/euromillions")
USER_AGENT = (
    "lottobench/0.1 "
    "(+https://github.com/kugguk2022/lotteries-init-at-your-service)"
)

#: A full EuroMillions archive is thousands of draws. A handful of rows means a truncated or
#: error payload, which must not be mistaken for "the history is short".
MIN_ROWS_FULL_HISTORY = 300

_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})

CANONICAL_COLUMNS = [
    "draw_date", "ball_1", "ball_2", "ball_3", "ball_4", "ball_5", "star_1", "star_2",
]

_BASELINE_CSV = "data/euromillions_history.csv"
_BASELINE_METADATA = "data/euromillions_history.meta.json"

#: Every column spelling seen across upstream sources and bundled files, mapped to the canonical
#: schema. ``n1..n5``/``star1``/``star2`` is the bundled-file spelling.
COLUMN_ALIASES = {
    "date": "draw_date",
    "draw_date": "draw_date",
    "drawdate": "draw_date",
    "ball1": "ball_1", "ball_1": "ball_1", "n1": "ball_1",
    "ball2": "ball_2", "ball_2": "ball_2", "n2": "ball_2",
    "ball3": "ball_3", "ball_3": "ball_3", "n3": "ball_3",
    "ball4": "ball_4", "ball_4": "ball_4", "n4": "ball_4",
    "ball5": "ball_5", "ball_5": "ball_5", "n5": "ball_5",
    "lucky_star1": "star_1", "lucky_star_1": "star_1", "star_1": "star_1", "star1": "star_1",
    "lucky_star2": "star_2", "lucky_star_2": "star_2", "star_2": "star_2", "star2": "star_2",
}


def cache_dir(custom: Path | None = None) -> Path:
    if custom:
        return custom
    override = os.environ.get("LOTTOBENCH_CACHE_DIR") or os.environ.get("EUROMILLIONS_CACHE_DIR")
    return Path(override) if override else CACHE_DIR


def _cache_key(url: str, params: dict[str, str], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    raw = f"{url}|{json.dumps(params, sort_keys=True)}".encode()
    return directory / f"{hashlib.sha256(raw).hexdigest()[:16]}.csv"


def _get(url: str, params: dict[str, str], *, timeout: float, attempts: int) -> str:
    """One URL, with backoff on the retryable statuses. Returns decoded text."""
    target = url
    if params:
        joiner = "&" if urllib.parse.urlparse(url).query else "?"
        target = f"{url}{joiner}{urllib.parse.urlencode(params)}"

    last: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(target, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                content_type = response.headers.get("Content-Type", "")
                if "text" not in content_type and "csv" not in content_type:
                    raise ContentTypeError(f"Unexpected content type: {content_type or 'none'}")
                charset = response.headers.get_content_charset() or "utf-8"
                return response.read().decode(charset, errors="replace")
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in _RETRY_STATUS:
                raise
        except urllib.error.URLError as exc:
            last = exc
        except TimeoutError as exc:  # urlopen raises this directly on socket timeout
            last = exc
        if attempt + 1 < attempts:
            time.sleep(0.5 * (2**attempt))
    raise FetchError(f"{target}: {last}")


class _NextDataParser(HTMLParser):
    """Extract the JSON script without coupling the adapter to presentation CSS classes."""

    def __init__(self) -> None:
        super().__init__()
        self._inside = False
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "script" and dict(attrs).get("id") == "__NEXT_DATA__":
            self._inside = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._inside:
            self._inside = False

    def handle_data(self, data: str) -> None:
        if self._inside:
            self.parts.append(data)


def parse_irish_official(html: str, spec: GameSpec | None = None) -> pd.DataFrame:
    """Parse EuroMillions draws from the operator page's structured Next.js payload."""
    parser = _NextDataParser()
    parser.feed(html)
    if not parser.parts:
        raise NormalizationError("Irish official page has no __NEXT_DATA__ payload")
    try:
        payload = json.loads("".join(parser.parts))
        entries = payload["props"]["pageProps"]["list"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise NormalizationError(f"Irish official result payload changed shape: {exc}") from exc
    if not isinstance(entries, list) or not entries:
        raise NormalizationError("Irish official result payload contains no draws")

    rows: list[dict[str, object]] = []
    try:
        for entry in entries:
            result = entry["standard"]
            grid = result["grids"][0]
            mains = grid["standard"][0]
            stars = grid["additional"][0]
            if len(mains) != 5 or len(stars) != 2:
                raise ValueError("expected five main numbers and two Lucky Stars")
            rows.append(
                dict(
                    zip(
                        CANONICAL_COLUMNS,
                        [str(result["drawDates"][0])[:10], *mains, *stars],
                        strict=True,
                    )
                )
            )
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise NormalizationError(f"Irish official result entry changed shape: {exc}") from exc
    return finalize(pd.DataFrame(rows, columns=CANONICAL_COLUMNS), spec or GameSpec.euromillions())


def fetch_irish_official(
    *,
    cache: Path | None = None,
    use_cache: bool = True,
    timeout: float = 15.0,
    attempts: int = 3,
) -> pd.DataFrame:
    """Fetch and validate the current Irish National Lottery result window."""
    directory = cache_dir(cache)
    cache_file = _cache_key(IRISH_OFFICIAL_URL, {}, directory)
    if use_cache and cache_file.exists():
        try:
            return parse_irish_official(cache_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, NormalizationError, ValueError):
            cache_file.unlink(missing_ok=True)
    try:
        html = _get(IRISH_OFFICIAL_URL, {}, timeout=timeout, attempts=attempts)
        frame = parse_irish_official(html)
    except (ContentTypeError, FetchError, NormalizationError, urllib.error.HTTPError) as exc:
        raise FetchError(f"Irish official EuroMillions results unavailable: {exc}") from exc
    cache_file.write_text(html, encoding="utf-8")
    return frame


def load_validated_history() -> pd.DataFrame:
    """Load the immutable seed after verifying its canonical digest and declared span."""
    package = files("lotteries_core")
    try:
        csv_text = package.joinpath(_BASELINE_CSV).read_text(encoding="utf-8")
        metadata = json.loads(package.joinpath(_BASELINE_METADATA).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise NormalizationError(f"Validated EuroMillions baseline is unavailable: {exc}") from exc
    frame = normalize(csv_text)
    actual = {
        "rows": len(frame),
        "first_draw": str(frame["draw_date"].min().date()),
        "last_draw": str(frame["draw_date"].max().date()),
        "content_sha256": content_digest(frame),
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in actual.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise NormalizationError(f"Validated EuroMillions baseline metadata mismatch: {mismatches}")
    return frame


def merge_validated_history(baseline: pd.DataFrame, official: pd.DataFrame) -> pd.DataFrame:
    """Append official draws only after an exact overlap proves continuity."""
    spec = GameSpec.euromillions()
    baseline = finalize(baseline[CANONICAL_COLUMNS], spec)
    official = finalize(official[CANONICAL_COLUMNS], spec)
    overlap = baseline.merge(official, on="draw_date", suffixes=("_baseline", "_official"))
    if overlap.empty:
        raise NormalizationError(
            "Irish official window does not overlap the validated EuroMillions history"
        )
    conflicts: list[str] = []
    for column in CANONICAL_COLUMNS[1:]:
        different = overlap[f"{column}_baseline"] != overlap[f"{column}_official"]
        conflicts.extend(str(value.date()) for value in overlap.loc[different, "draw_date"])
    if conflicts:
        raise NormalizationError(
            "Irish official results conflict with validated history for draw date(s): "
            + ", ".join(sorted(set(conflicts)))
        )
    return finalize(pd.concat([baseline, official], ignore_index=True), spec)


def _looks_like_draw_csv(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    header = lines[0].lower()
    if any(token in header for token in ("ball1", "ball 1", "ball_1", "lucky", "star")):
        return True
    parts = [part.strip() for part in lines[0].split(",") if part.strip()]
    if len(parts) < 7:
        return False
    try:
        for part in parts[1:]:
            float(part)
    except ValueError:
        return False
    return True


def fetch_raw_csv(
    date_from: str | None = None,
    date_to: str | None = None,
    *,
    urls: tuple[str, ...] | list[str] | None = None,
    cache: Path | None = None,
    use_cache: bool = True,
    timeout: float = 15.0,
    attempts: int = 3,
    min_rows: int = MIN_ROWS_FULL_HISTORY,
    allow_partial: bool = False,
) -> str:
    """Return raw CSV text from the first archive that answers with a plausible full history."""
    params: dict[str, str] = {}
    if date_from:
        params["from"] = date_from
    if date_to:
        params["to"] = date_to

    def complete_enough(text: str) -> bool:
        if not _looks_like_draw_csv(text):
            return False
        if date_from or date_to or allow_partial:
            return True
        return text.count("\n") + 1 >= min_rows

    directory = cache_dir(cache)
    errors: list[str] = []
    for url in urls or CSV_URLS:
        cache_file = _cache_key(url, params, directory)
        if use_cache and cache_file.exists():
            cached = cache_file.read_text(encoding="utf-8")
            if complete_enough(cached):
                return cached

        try:
            text = _get(url, params, timeout=timeout, attempts=attempts)
        except (FetchError, ContentTypeError, urllib.error.HTTPError) as exc:
            errors.append(f"{url}: {exc}")
            # A stale cache beats no data, but only when it is itself plausible.
            if use_cache and cache_file.exists():
                cached = cache_file.read_text(encoding="utf-8")
                if complete_enough(cached):
                    return cached
            continue

        if not _looks_like_draw_csv(text):
            errors.append(f"{url}: unexpected payload (missing draw headers)")
            continue
        cache_file.write_text(text, encoding="utf-8")
        if complete_enough(text):
            return text
        errors.append(f"{url}: insufficient rows ({text.count(chr(10)) + 1})")

    raise FetchError("Failed to fetch EuroMillions CSV; attempts: " + "; ".join(errors))


def canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Lower-case, de-space, and alias column names onto the canonical schema."""
    renamed = frame.rename(
        columns={column: str(column).lower().strip().replace(" ", "_") for column in frame.columns}
    )
    return renamed.rename(
        columns={key: value for key, value in COLUMN_ALIASES.items() if key in renamed.columns}
    )


def _normalize_headerless(csv_text: str, spec: GameSpec) -> pd.DataFrame:
    frame = pd.read_csv(StringIO(csv_text), header=None)
    if frame.shape[1] < len(CANONICAL_COLUMNS):
        raise NormalizationError(f"Headerless payload has too few columns: {frame.shape[1]}")
    frame = frame.iloc[:, : len(CANONICAL_COLUMNS)].copy()
    frame.columns = CANONICAL_COLUMNS
    return finalize(frame, spec)


def normalize(csv_text: str, spec: GameSpec | None = None) -> pd.DataFrame:
    """Parse raw CSV text into validated, sorted, de-duplicated canonical history."""
    spec = spec or GameSpec.euromillions()
    try:
        frame = pd.read_csv(StringIO(csv_text))
    except Exception as exc:
        raise NormalizationError(f"Failed to parse CSV payload: {exc}") from exc

    frame = canonicalize_columns(frame)
    missing = [column for column in CANONICAL_COLUMNS if column not in frame.columns]
    if missing:
        return _normalize_headerless(csv_text, spec)
    return finalize(frame[CANONICAL_COLUMNS], spec)


def fetch_euromillions(
    *,
    source: str = "auto",
    date_from: str | None = None,
    date_to: str | None = None,
    cache: Path | None = None,
    use_cache: bool = True,
    timeout: float = 15.0,
    allow_partial: bool = False,
    baseline: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Retrieve and normalize EuroMillions history.

    The default strategy extends ``baseline`` (or the bundled seed) with the Irish operator's
    recent window. ``archive`` and ``lottology`` remain explicit maintenance sources.
    """
    if source not in SOURCE_CHOICES:
        raise ValueError(f"unknown source {source!r}; choose from {list(SOURCE_CHOICES)}")

    spec = GameSpec.euromillions()
    if source in ("archive", "lottology"):
        from .html_archive import fetch_html_archive

        return fetch_html_archive(source, spec=spec)

    if source in ("auto", "irish-official"):
        history = load_validated_history() if baseline is None else baseline
        frame = merge_validated_history(
            history,
            fetch_irish_official(cache=cache, use_cache=use_cache, timeout=timeout),
        )
        if date_from:
            frame = frame[frame["draw_date"] >= pd.to_datetime(date_from)]
        if date_to:
            frame = frame[frame["draw_date"] <= pd.to_datetime(date_to)]
        return frame.reset_index(drop=True)

    urls = {
        "national-lottery": (PRIMARY_URL,),
        "merseyworld": (SECONDARY_URL,),
    }[source]
    raw = fetch_raw_csv(
        date_from,
        date_to,
        urls=urls,
        cache=cache,
        use_cache=use_cache,
        timeout=timeout,
        allow_partial=allow_partial,
    )
    frame = normalize(raw, spec)
    if date_from:
        frame = frame[frame["draw_date"] >= pd.to_datetime(date_from)]
    if date_to:
        frame = frame[frame["draw_date"] <= pd.to_datetime(date_to)]
    return frame.reset_index(drop=True)
