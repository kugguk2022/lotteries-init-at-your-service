from __future__ import annotations

import re
from datetime import date, datetime
from typing import List, Optional

import requests
from bs4 import BeautifulSoup

from .lottology import EMRow

ARCHIVE_URL_TEMPLATE = "https://www.euro-millions.com/results-history-{year}"
ARCHIVE_MIN_YEAR = 2004
UA = {"User-Agent": "lotteries/0.1 (+https://github.com/kugguk2022/lotteries)"}


def _normalize_archive_date(text: str) -> str:
    cleaned = re.sub(r"(\d+)\s+(st|nd|rd|th)\b", r"\1", text.strip(), flags=re.I)
    return datetime.strptime(cleaned, "%A %d %B %Y").date().isoformat()


def _parse_archive_year(html: str) -> List[EMRow]:
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        if not rows:
            continue
        header = " ".join(rows[0].get_text(" ", strip=True).split())
        if "Result Date" not in header or "Numbers" not in header:
            continue

        parsed: List[EMRow] = []
        for row in rows[1:]:
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            date_text = " ".join(cells[0].get_text(" ", strip=True).split())
            if not date_text.startswith(("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")):
                continue
            numbers = [int(value) for value in re.findall(r"\d+", cells[1].get_text(" ", strip=True))]
            if len(numbers) < 7:
                continue
            draw_date = _normalize_archive_date(date_text)
            n1, n2, n3, n4, n5, star1, star2 = numbers[:7]
            parsed.append(EMRow(draw_date, n1, n2, n3, n4, n5, star1, star2))

        if parsed:
            return parsed

    raise RuntimeError("Archive table not found or parse yielded no EuroMillions rows.")


def fetch_euromillions_archive(
    session: Optional[requests.Session] = None,
    *,
    start_year: int | None = None,
    end_year: int | None = None,
) -> List[EMRow]:
    current_year = date.today().year
    start = max(ARCHIVE_MIN_YEAR, start_year or ARCHIVE_MIN_YEAR)
    end = min(current_year, end_year or current_year)
    if start > end:
        raise ValueError(f"Invalid archive range: start_year={start}, end_year={end}")

    s = session or requests.Session()
    rows_by_date: dict[str, EMRow] = {}
    for year in range(start, end + 1):
        url = ARCHIVE_URL_TEMPLATE.format(year=year)
        response = s.get(url, headers=UA, timeout=30)
        response.raise_for_status()
        for row in _parse_archive_year(response.text):
            rows_by_date[row.date] = row

    if not rows_by_date:
        raise RuntimeError("Archive fetch returned no EuroMillions rows.")
    return [rows_by_date[key] for key in sorted(rows_by_date)]
