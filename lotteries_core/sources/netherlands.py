"""Official Netherlands Lotto history adapter.

The playable line is six distinct integers from 1 through 45. The reserve number and the
jackpot-machine colour are result metadata, not user-selected pools. Lotto XL and the second draw
on Super Saturday are separate series and are deliberately not mixed into the primary Lotto
training history returned here.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import date
from html.parser import HTMLParser

import pandas as pd

from ..protocol import GameSpec
from .errors import FetchError, NormalizationError
from .schema import finalize

RESULTS_PAGE = "https://lotto.nederlandseloterij.nl/trekkingsuitslag"
RESULT_API = "https://lotto-api.nederlandseloterij.nl/api/draws/results/{draw_date}"
USER_AGENT = "lottobench/0.1 (+https://github.com/kugguk2022/lotteries)"


class _PublishedDrawParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.dates: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag != "a" or attributes.get("data-test") != "date-slider-item":
            return
        prefix = "/trekkingsuitslag/"
        href = attributes.get("href") or ""
        if not href.startswith(prefix):
            return
        candidate = href.removeprefix(prefix)
        try:
            self.dates.add(date.fromisoformat(candidate).isoformat())
        except ValueError:
            return


def _get_text(url: str, *, timeout: float) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode(response.headers.get_content_charset() or "utf-8")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        raise FetchError(f"{url}: {exc}") from exc


def published_dates(html: str) -> list[str]:
    """Extract completed draws from the official result slider, excluding page metadata."""
    parser = _PublishedDrawParser()
    parser.feed(html)
    today = date.today().isoformat()
    return sorted(value for value in parser.dates if value <= today)


def normalize_result(payload: dict) -> list[dict]:
    """Return primary Lotto rows from one official API response.

    A Super Saturday response contains two regular and two XL results. The first regular result is
    the primary weekly Lotto series; later regular results are separate promotional series and are
    not silently folded into this benchmark.
    """
    results = payload.get("results")
    if not isinstance(results, list):
        raise NormalizationError("Netherlands Lotto response has no results list")
    regular = [item for item in results if item.get("isXlDraw") is False]
    if not regular:
        return []
    item = regular[0]
    draw_date = item.get("draw", {}).get("drawDate")
    winning = item.get("winningNumbers", {})
    numbers = winning.get("numbers")
    if not isinstance(draw_date, str) or not isinstance(numbers, list) or len(numbers) != 6:
        raise NormalizationError("Netherlands Lotto response has an invalid primary draw")
    numbers = sorted(numbers)
    row = {
        "draw_date": draw_date,
        **{f"ball_{index + 1}": value for index, value in enumerate(numbers)},
        "reserve_number": winning.get("bonusNumber"),
        "jackpot_color": "yellow" if item.get("jackpotGuaranteed") else "black",
        "jackpot_won": bool(item.get("isJackpotWon")),
        "jackpot_amount_gross_cents": item.get("jackpotAmountGross"),
        "source_draw_count": len(regular),
    }
    return [row]


def fetch_netherlands(
    *,
    date_from: str | None = None,
    date_to: str | None = None,
    timeout: float = 15.0,
) -> pd.DataFrame:
    """Retrieve the published primary Lotto history from the official operator API."""
    html = _get_text(RESULTS_PAGE, timeout=timeout)
    dates = published_dates(html)
    if date_from:
        dates = [value for value in dates if value >= date_from]
    if date_to:
        dates = [value for value in dates if value <= date_to]
    if not dates:
        raise FetchError("Official Netherlands Lotto page exposed no draw dates in the requested range")

    rows: list[dict] = []
    errors: list[str] = []
    for draw_date in dates:
        try:
            raw = _get_text(RESULT_API.format(draw_date=draw_date), timeout=timeout)
            rows.extend(normalize_result(json.loads(raw)))
        except (FetchError, json.JSONDecodeError, NormalizationError) as exc:
            errors.append(f"{draw_date}: {exc}")
    if errors:
        detail = "; ".join(errors[-3:])
        raise FetchError(
            f"Netherlands Lotto retrieval was incomplete ({len(errors)}/{len(dates)} dates failed); "
            f"refusing a history with silent gaps: {detail}"
        )
    if not rows:
        raise FetchError("No Netherlands Lotto draws retrieved")

    frame = pd.DataFrame(rows)
    return finalize(frame, GameSpec("nl-lotto", main_n=45, main_k=6))
