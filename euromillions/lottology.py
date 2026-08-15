# euromillions/sources/lottology.py
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

import requests

LOTTOLOGY_ARCHIVE_URL = "https://www.lottology.com/europe/euromillions/past-draws-archive/"
UA = {"User-Agent": "lotteries-bot/1.0 (+https://github.com/kugguk2022/lotteries)"}


@dataclass(frozen=True)
class EMRow:
    date: str  # YYYY-MM-DD
    n1: int
    n2: int
    n3: int
    n4: int
    n5: int
    star1: int
    star2: int


def _abs(url: str) -> str:
    if url.startswith("http"):
        return url
    if url.startswith("/"):
        return "https://www.lottology.com" + url
    return "https://www.lottology.com/" + url


def _find_txt_export(html: str) -> str | None:
    # naive but effective: get all hrefs, pick a .txt/.csv first
    hrefs = re.findall(r'href="([^"]+)"', html, flags=re.IGNORECASE)
    for ext in (".txt", ".csv"):
        for h in hrefs:
            if ext in h.lower() and "euromillions" in h.lower():
                return _abs(h)
    # fallback: any txt/csv link if euromillions not in URL
    for ext in (".txt", ".csv"):
        for h in hrefs:
            if ext in h.lower():
                return _abs(h)
    return None


def _parse_date(s: str) -> str:
    s = s.strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d.%m.%Y"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            pass
    # last resort: extract 3 numbers and guess d/m/y
    m = re.search(r"(\d{1,2})\D+(\d{1,2})\D+(\d{4})", s)
    if not m:
        raise ValueError(f"Unparseable date: {s!r}")
    d, mo, y = map(int, m.groups())
    return datetime(y, mo, d).date().isoformat()


def fetch_euromillions_lottology(session: requests.Session | None = None) -> list[EMRow]:
    s = session or requests.Session()
    html = s.get(LOTTOLOGY_ARCHIVE_URL, headers=UA, timeout=30).text
    export_url = _find_txt_export(html)
    if not export_url:
        raise RuntimeError("Lottology export link not found (no .txt/.csv on archive page).")

    raw = s.get(export_url, headers=UA, timeout=60).content
    text = raw.decode("utf-8", errors="replace")

    rows: list[EMRow] = []
    # Heuristic parser: each line should contain a date + 7 numbers (5 + 2 stars)
    for line in text.splitlines():
        nums = list(map(int, re.findall(r"\b\d+\b", line)))
        if len(nums) < 8:
            continue
        # try: first token is date-ish; but text exports vary, so we locate a date substring
        m = re.search(r"(\d{1,2}\D+\d{1,2}\D+\d{4}|\d{4}-\d{2}-\d{2})", line)
        if not m:
            continue
        date_iso = _parse_date(m.group(0))
        # take the last 7 numbers on the line as draw numbers (common pattern)
        n1, n2, n3, n4, n5, s1, s2 = nums[-7:]
        rows.append(EMRow(date_iso, n1, n2, n3, n4, n5, s1, s2))

    if len(rows) < 300:
        raise RuntimeError(f"Lottology parsed only {len(rows)} rows (<300). Export format may have changed.")
    # dedupe by date
    uniq = {}
    for r in rows:
        uniq[r.date] = r
    return sorted(uniq.values(), key=lambda r: r.date)
