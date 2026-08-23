#!/usr/bin/env python3
"""
Fetch Portuguese Totoloto historical draws and save as CSV/JSON.

Primary source (per-year archive pages):
  https://www.euro-millions.com/pt/totoloto/arquivo-de-resultados-YYYY

Output columns (chronological, earliest -> latest):
  draw_date,weekday,ball_1,ball_2,ball_3,ball_4,ball_5,bonus,draw_code,source_url

Notes
- Some pages list 6 or 7 integers per draw. We assume 5 mains + 1 bonus (last). If only 5 are present, bonus is empty.
- Portuguese dates like 'Sábado 17 de fevereiro de 2024' are parsed to ISO YYYY-MM-DD.
- Ranges are configurable (defaults: balls 1..49, bonus 1..13). Use flags to override if rules differ.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

YEAR_URL = "https://www.euro-millions.com/pt/totoloto/arquivo-de-resultados-{year}"

FIELDS = [
    "draw_date",
    "weekday",
    "ball_1",
    "ball_2",
    "ball_3",
    "ball_4",
    "ball_5",
    "bonus",
    "draw_code",
    "source_url",
]

MONTHS_PT = {
    "janeiro": 1,
    "fevereiro": 2,
    "março": 3,
    "marco": 3,
    "abril": 4,
    "maio": 5,
    "junho": 6,
    "julho": 7,
    "agosto": 8,
    "setembro": 9,
    "outubro": 10,
    "novembro": 11,
    "dezembro": 12,
}


def _weekday_name(iso_date: str) -> str:
    y, m, d = map(int, iso_date.split("-"))
    return dt.date(y, m, d).strftime("%a")


def _session_with_retry() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {"User-Agent": "lotteries-totoloto-scraper/1.0 (+https://github.com/kugguk2022/lotteries)"}
    )
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=["GET"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s


def _parse_pt_date(text: str) -> Optional[str]:
    """Parse 'Quinta-feira 20 de novembro de 2025' -> '2025-11-20'."""
    m = re.search(r"(\d{1,2})\s+de\s+([A-Za-zÀ-ÿ]+)\s+de\s+(\d{4})", text, flags=re.I)
    if not m:
        return None
    day = int(m.group(1))
    month_name = m.group(2).strip().lower()
    year_val = int(m.group(3))
    month = MONTHS_PT.get(month_name) or MONTHS_PT.get(month_name.capitalize())
    if not month:
        return None
    try:
        return dt.date(year_val, month, day).isoformat()
    except Exception:
        return None


@dataclass
class Ranges:
    ball_min: int = 1
    ball_max: int = 49
    bonus_min: int = 1
    bonus_max: int = 13

    def check(self, balls: List[int], bonus: Optional[int]) -> bool:
        if len(balls) != 5:
            return False
        if not all(self.ball_min <= b <= self.ball_max for b in balls):
            return False
        if len(set(balls)) != 5:
            return False
        if bonus is None:
            return True
        return self.bonus_min <= bonus <= self.bonus_max


def fetch_year_html(session: requests.Session, year: int) -> str:
    url = YEAR_URL.format(year=year)
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return r.text


def parse_year(html: str, year: int, ranges: Ranges) -> List[Dict]:
    """Extract draws from a per-year page.
    Heuristic: for each date+UL pair, read 6-7 integers; the last is 'bonus', first five are 'balls'.
    If only 5 numbers exist, set bonus=None.
    Also try to capture a draw code like 'NNN/YYYY' if present.
    """
    out: List[Dict] = []

    # Find blocks with a Portuguese date followed by a UL with numbers; optional code NNN/YYYY nearby.
    patt = re.compile(
        r"(?:>([A-Za-zÀ-ÿ]+\s+\d{1,2}\s+de\s+[A-Za-zÀ-ÿ]+\s+de\s+\d{4})<).*?(\d{3}/\d{4})?.*?(<ul[^>]*>.*?</ul>)",
        flags=re.S | re.I,
    )

    for m in patt.finditer(html):
        date_pt, code, ul_html = m.group(1), m.group(2), m.group(3)
        date_iso = _parse_pt_date(date_pt) or ""
        if not date_iso:
            continue
        nums = [int(x) for x in re.findall(r"<li>\s*(\d{1,2})\s*</li>", ul_html)]
        if len(nums) < 5:
            continue

        if len(nums) >= 6:
            mains, bonus = nums[:5], nums[-1]
        else:
            mains, bonus = nums[:5], None

        if not ranges.check(mains, bonus):
            # Try alternative: sometimes sites list bonus not last; fallback to 5 mains and ignore extras
            mains = mains[:5]
            bonus = bonus if bonus is None else bonus
            if not ranges.check(mains, bonus):
                continue

        rec = {
            "draw_date": date_iso,
            "weekday": _weekday_name(date_iso),
            "ball_1": mains[0],
            "ball_2": mains[1],
            "ball_3": mains[2],
            "ball_4": mains[3],
            "ball_5": mains[4],
            "bonus": bonus,
            "draw_code": code or "",
            "source_url": YEAR_URL.format(year=year),
        }
        out.append(rec)

    return out


def dedupe_sort(recs: List[Dict]) -> List[Dict]:
    best: Dict[Tuple[str, Tuple[int, ...], Optional[int]], Dict] = {}
    for r in recs:
        key = (
            r["draw_date"],
            (
                int(r["ball_1"]),
                int(r["ball_2"]),
                int(r["ball_3"]),
                int(r["ball_4"]),
                int(r["ball_5"]),
            ),
            int(r["bonus"]) if r["bonus"] is not None else None,
        )
        prev = best.get(key)
        if not prev:
            best[key] = r
            continue
        # Prefer record with draw_code
        if (not prev.get("draw_code")) and r.get("draw_code"):
            best[key] = r
    return sorted(best.values(), key=lambda x: x["draw_date"])


def filter_year_range(recs: List[Dict], start_year: int, end_year: int) -> List[Dict]:
    out = []
    for r in recs:
        y = int(r["draw_date"][:4])
        if start_year <= y <= end_year:
            out.append(r)
    return out


def save_csv(recs: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in recs:
            row = {k: r.get(k) for k in FIELDS}
            w.writerow(row)


def save_json(recs: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch Totoloto draws and write normalized CSV/JSON")
    ap.add_argument("--out", type=Path, required=True, help="Output path")
    ap.add_argument("--format", choices=["csv", "json"], default="csv")
    ap.add_argument("--start-year", type=int, default=2000)
    ap.add_argument("--end-year", type=int, default=9999)
    ap.add_argument("--ball-range", type=int, nargs=2, default=[1, 49], metavar=("MIN", "MAX"))
    ap.add_argument("--bonus-range", type=int, nargs=2, default=[1, 13], metavar=("MIN", "MAX"))
    ap.add_argument(
        "--years", type=int, nargs="*", default=None, help="Explicit list of years to fetch"
    )
    args = ap.parse_args()

    rng = Ranges(args.ball_range[0], args.ball_range[1], args.bonus_range[0], args.bonus_range[1])

    session = _session_with_retry()
    errors, recs = [], []

    years = args.years
    if not years:
        years = list(range(args.start_year, min(args.end_year, dt.date.today().year) + 1))

    for y in years:
        try:
            html = fetch_year_html(session, y)
            recs.extend(parse_year(html, y, rng))
        except Exception as e:
            errors.append(f"year {y}: {e}")

    if not recs:
        raise SystemExit("No records fetched.\n" + "\n".join(errors))

    recs = filter_year_range(dedupe_sort(recs), args.start_year, args.end_year)

    if args.format == "csv":
        save_csv(recs, args.out)
    else:
        save_json(recs, args.out)

    print(
        f"OK: wrote {len(recs)} draws to {args.out} "
        f"({recs[0]['draw_date']} → {recs[-1]['draw_date']})."
    )
    if errors:
        print("Warnings:\n  " + "\n  ".join(errors))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Aborted.", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
