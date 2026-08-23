"""HTML archive fallbacks used when the official EuroMillions CSV is unavailable.

``requests`` and ``beautifulsoup4`` are base dependencies because automatic retrieval must also
work in a clean installation. Imports are still checked up front to give a useful repair message
if an installation is incomplete.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from ..protocol import GameSpec
from .schema import finalize

ARCHIVE_URL_TEMPLATE = "https://www.euro-millions.com/results-history-{year}"
ARCHIVE_MIN_YEAR = 2004
LOTTOLOGY_ARCHIVE_URL = "https://www.lottology.com/europe/euromillions/past-draws-archive/"
UA = {"User-Agent": "lottobench/0.1 (+https://github.com/kugguk2022/lotteries)"}

MIN_LOTTOLOGY_ROWS = 300


@dataclass(frozen=True)
class DrawRow:
    """One parsed draw: ISO date plus five main numbers and two stars."""

    date: str
    n1: int
    n2: int
    n3: int
    n4: int
    n5: int
    star1: int
    star2: int


def _require_scrape_dependencies():
    """Return ``(requests, BeautifulSoup)`` or explain how to repair the installation."""
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ImportError(
            "LottoBench retrieval dependencies are missing; run: "
            "pip install --force-reinstall lottobench"
        ) from exc
    return requests, BeautifulSoup


def _normalize_archive_date(text: str) -> str:
    cleaned = re.sub(r"(\d+)\s+(st|nd|rd|th)\b", r"\1", text.strip(), flags=re.I)
    return datetime.strptime(cleaned, "%A %d %B %Y").date().isoformat()


def _parse_archive_year(html: str, soup_factory) -> list[DrawRow]:
    soup = soup_factory(html, "html.parser")
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if not rows:
            continue
        header = " ".join(rows[0].get_text(" ", strip=True).split())
        if "Result Date" not in header or "Numbers" not in header:
            continue

        parsed: list[DrawRow] = []
        for row in rows[1:]:
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            date_text = " ".join(cells[0].get_text(" ", strip=True).split())
            if not date_text.startswith(
                ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
            ):
                continue
            numbers = [
                int(value)
                for value in re.findall(r"\d+", cells[1].get_text(" ", strip=True))
            ]
            if len(numbers) < 7:
                continue
            parsed.append(DrawRow(_normalize_archive_date(date_text), *numbers[:7]))
        if parsed:
            return parsed
    raise RuntimeError("Archive table not found or parse yielded no EuroMillions rows.")


def fetch_archive(*, start_year: int | None = None, end_year: int | None = None) -> list[DrawRow]:
    """Scrape euro-millions.com year pages."""
    requests, soup_factory = _require_scrape_dependencies()
    current = date.today().year
    start = max(ARCHIVE_MIN_YEAR, start_year or ARCHIVE_MIN_YEAR)
    end = min(current, end_year or current)
    if start > end:
        raise ValueError(f"Invalid archive range: start_year={start}, end_year={end}")

    session = requests.Session()
    by_date: dict[str, DrawRow] = {}
    for year in range(start, end + 1):
        response = session.get(ARCHIVE_URL_TEMPLATE.format(year=year), headers=UA, timeout=30)
        response.raise_for_status()
        for row in _parse_archive_year(response.text, soup_factory):
            by_date[row.date] = row
    if not by_date:
        raise RuntimeError("Archive fetch returned no EuroMillions rows.")
    return [by_date[key] for key in sorted(by_date)]


def _absolute(url: str) -> str:
    if url.startswith("http"):
        return url
    return "https://www.lottology.com" + (url if url.startswith("/") else "/" + url)


def _find_text_export(html: str) -> str | None:
    hrefs = re.findall(r'href="([^"]+)"', html, flags=re.I)
    for extension in (".txt", ".csv"):
        for href in hrefs:
            if extension in href.lower() and "euromillions" in href.lower():
                return _absolute(href)
    for extension in (".txt", ".csv"):
        for href in hrefs:
            if extension in href.lower():
                return _absolute(href)
    return None


def _parse_loose_date(text: str) -> str:
    text = text.strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d.%m.%Y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    match = re.search(r"(\d{1,2})\D+(\d{1,2})\D+(\d{4})", text)
    if not match:
        raise ValueError(f"Unparseable date: {text!r}")
    day, month, year = map(int, match.groups())
    return datetime(year, month, day).date().isoformat()


def fetch_lottology() -> list[DrawRow]:
    """Follow lottology.com's archive page to its text export and parse it."""
    requests, _ = _require_scrape_dependencies()
    session = requests.Session()
    html = session.get(LOTTOLOGY_ARCHIVE_URL, headers=UA, timeout=30).text
    export_url = _find_text_export(html)
    if not export_url:
        raise RuntimeError("Lottology export link not found (no .txt/.csv on archive page).")

    text = session.get(export_url, headers=UA, timeout=60).content.decode(
        "utf-8", errors="replace"
    )
    rows: list[DrawRow] = []
    for line in text.splitlines():
        numbers = [int(value) for value in re.findall(r"\b\d+\b", line)]
        if len(numbers) < 8:
            continue
        match = re.search(r"(\d{1,2}\D+\d{1,2}\D+\d{4}|\d{4}-\d{2}-\d{2})", line)
        if not match:
            continue
        rows.append(DrawRow(_parse_loose_date(match.group(0)), *numbers[-7:]))

    if len(rows) < MIN_LOTTOLOGY_ROWS:
        raise RuntimeError(
            f"Lottology parsed only {len(rows)} rows (<{MIN_LOTTOLOGY_ROWS}); "
            "export format may have changed."
        )
    unique = {row.date: row for row in rows}
    return sorted(unique.values(), key=lambda row: row.date)


def rows_to_frame(rows: list[DrawRow], spec: GameSpec | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "draw_date": row.date,
                "ball_1": row.n1, "ball_2": row.n2, "ball_3": row.n3,
                "ball_4": row.n4, "ball_5": row.n5,
                "star_1": row.star1, "star_2": row.star2,
            }
            for row in rows
        ]
    )
    return finalize(frame, spec)


def fetch_html_archive(
    source: str, *, spec: GameSpec | None = None, start_year: int | None = None
) -> pd.DataFrame:
    """Dispatch to an HTML source and return canonical history."""
    rows = fetch_archive(start_year=start_year) if source == "archive" else fetch_lottology()
    return rows_to_frame(rows, spec)
