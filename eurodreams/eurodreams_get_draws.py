#!/usr/bin/env python3
"""
Get EuroDreams historical draws (2023-present) and save as a CSV or JSON.

Primary source (full archive): https://www.irishlottery.com/eurodreams-archive
Secondary source (per-year pages): https://www.euro-millions.com/pt/eurodreams/arquivo-de-resultados-YYYY
Tertiary (recent 90 days only): https://www.lottery.ie/results/eurodreams/history

Output columns (chronological, earliest -> latest):
date,weekday,n1,n2,n3,n4,n5,n6,dream,draw_code,source_url

Notes
- draw_code is like '093/2025' when available (from euro-millions.com); otherwise empty.
- Weekday is computed from the date in UTC (ISO YYYY-MM-DD). EuroDreams draws are Mon/Thu ~21:00 CET.
- This script does not include prize breakdowns or annuity details.
- Dependencies: requests
"""

import argparse
import csv
import datetime as dt
import json
import re
import sys
import traceback
from pathlib import Path

import requests

IRISH_ARCHIVE_URL = "https://www.irishlottery.com/eurodreams-archive"
EUROMILLIONS_YEAR_URL = "https://www.euro-millions.com/pt/eurodreams/arquivo-de-resultados-{year}"
LOTTERY_IE_HISTORY_URL = "https://www.lottery.ie/results/eurodreams/history"  # last ~90 days


FIELDS = ["date", "weekday", "n1", "n2", "n3", "n4", "n5", "n6", "dream", "draw_code", "source_url"]


def _weekday_name(iso_date: str) -> str:
    y, m, d = map(int, iso_date.split("-"))
    return dt.date(y, m, d).strftime("%a")


def _iso_from_eng_date(text: str) -> str | None:
    """Convert strings like 'November 20th 2025' -> '2025-11-20'."""
    m = re.search(r"\b([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?\s+(\d{4})\b", text)
    if not m:
        return None
    month_name, day, year = m.group(1), m.group(2), m.group(3)
    try:
        dt_obj = dt.datetime.strptime(f"{day} {month_name} {year}", "%d %B %Y")
        return dt_obj.date().isoformat()
    except Exception:
        return None


def _get_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })
    return s

def fetch_irish_archive() -> str:
    s = _get_session()
    r = s.get(IRISH_ARCHIVE_URL, timeout=30)
    r.raise_for_status()
    return r.text


def parse_irish_archive(html: str) -> list[dict]:
    """Scan for date markers, then read the next 7 <li> integers (6 mains + dream)."""
    out: list[dict] = []
    # Date anchors like >November 20th 2025<
    date_iter = list(re.finditer(r">\s*([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?\s+\d{4})\s*<", html))
    # All numbers in order of appearance
    num_iter = list(re.finditer(r"<li>\s*(\d{1,2})\s*</li>", html))
    nums_positions = [(m.start(), int(m.group(1))) for m in num_iter]

    for dmatch in date_iter:
        date_text = dmatch.group(1)
        date_iso = _iso_from_eng_date(date_text)
        if not date_iso:
            continue
        # Take the next 7 numbers after this date location
        start_pos = dmatch.end()
        following = [n for (pos, n) in nums_positions if pos > start_pos][:7]
        if len(following) != 7:
            continue
        mains, dream = following[:6], following[6]
        rec = {
            "date": date_iso,
            "weekday": _weekday_name(date_iso),
            "n1": mains[0],
            "n2": mains[1],
            "n3": mains[2],
            "n4": mains[3],
            "n5": mains[4],
            "n6": mains[5],
            "dream": dream,
            "draw_code": "",
            "source_url": IRISH_ARCHIVE_URL,
        }
        out.append(rec)
    return out


def fetch_euromillions_year(year: int) -> str:
    url = EUROMILLIONS_YEAR_URL.format(year=year)
    s = _get_session()
    r = s.get(url, timeout=30)
    r.raise_for_status()
    return r.text


def parse_euromillions_year(html: str, year: int) -> list[dict]:
    """Extract Portuguese date, draw_code 'NNN/YYYY', and 7 numbers (6 + dream)."""
    out: list[dict] = []

    for m in re.finditer(
        r"(?:>([A-Za-zÀ-ÿ]+\s+\d{1,2}\s+de\s+[A-Za-zÀ-ÿ]+\s+\d{4})<).*?(\d{3}/\d{4}).*?(<ul[^>]*>.*?</ul>)",
        html,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        date_pt = m.group(1)
        draw_code = m.group(2)
        ul_html = m.group(3)

        md = re.search(r"(\d{1,2})\s+de\s+([A-Za-zÀ-ÿ]+)\s+(\d{4})", date_pt, flags=re.IGNORECASE)
        if not md:
            continue
        day = int(md.group(1))
        month_name = md.group(2).strip().lower()

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
        month = MONTHS_PT.get(month_name) or MONTHS_PT.get(month_name.capitalize(), None)
        year_val = int(md.group(3))
        try:
            date_iso = dt.date(year_val, month, day).isoformat()
        except Exception:
            continue

        nums = [int(x) for x in re.findall(r"<li>\s*(\d{1,2})\s*</li>", ul_html)]
        if len(nums) < 7:
            continue
        mains, dream = nums[:6], nums[6]

        rec = {
            "date": date_iso,
            "weekday": _weekday_name(date_iso),
            "n1": mains[0],
            "n2": mains[1],
            "n3": mains[2],
            "n4": mains[3],
            "n5": mains[4],
            "n6": mains[5],
            "dream": dream,
            "draw_code": draw_code,
            "source_url": EUROMILLIONS_YEAR_URL.format(year=year),
        }
        out.append(rec)
    return out


def fetch_lottery_ie_recent() -> str:
    s = _get_session()
    r = s.get(LOTTERY_IE_HISTORY_URL, timeout=30)
    r.raise_for_status()
    return r.text


def parse_lottery_ie_recent(html: str) -> list[dict]:
    """Parse recent (last ~90 days) results from lottery.ie."""
    out: list[dict] = []
    for m in re.finditer(r"(\d{1,2})/(\d{1,2})/(\d{4}).*?<ul[^>]*>(.*?)</ul>", html, flags=re.DOTALL):
        d, mth, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        ul_html = m.group(4)
        try:
            date_iso = dt.date(y, mth, d).isoformat()
        except Exception:
            continue
        nums = [int(x) for x in re.findall(r"<li>\s*(\d{1,2})\s*</li>", ul_html)]
        if len(nums) < 7:
            continue
        mains, dream = nums[:6], nums[6]
        rec = {
            "date": date_iso,
            "weekday": _weekday_name(date_iso),
            "n1": mains[0],
            "n2": mains[1],
            "n3": mains[2],
            "n4": mains[3],
            "n5": mains[4],
            "n6": mains[5],
            "dream": dream,
            "draw_code": "",
            "source_url": LOTTERY_IE_HISTORY_URL,
        }
        out.append(rec)
    return out


def dedupe_keep_best(recs: list[dict]) -> list[dict]:
    best: dict[str, dict] = {}
    for r in recs:
        k = r["date"]
        prev = best.get(k)
        if not prev:
            best[k] = r
            continue
        # Prefer entries with draw_code; otherwise keep existing
        if (not prev.get("draw_code")) and r.get("draw_code"):
            best[k] = r
    return sorted(best.values(), key=lambda x: x["date"])


def filter_year_range(recs: list[dict], start_year: int, end_year: int) -> list[dict]:
    out = []
    for r in recs:
        y = int(r["date"][:4])
        if start_year <= y <= end_year:
            out.append(r)
    return out


def save_csv(recs: list[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in recs:
            w.writerow({k: r.get(k) for k in FIELDS})


def save_json(recs: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False, indent=2)


def _load_existing(path: str) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _validate_structured_rows(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for r in rows:
        # Normalize keys
        mapped = {}
        # Date
        if "date" in r:
            mapped["date"] = r["date"]
        elif "draw_date" in r:
            mapped["date"] = r["draw_date"]
        
        # Balls
        for i in range(1, 7):
            k_std = f"n{i}"
            k_alt = f"ball_{i}"
            if k_std in r:
                mapped[k_std] = r[k_std]
            elif k_alt in r:
                mapped[k_std] = r[k_alt]
        
        # Handle the weird fallback case where header is ball_1..5, star_1, star_2
        if "n6" not in mapped and "star_1" in r:
            # Assume this is actually ball 6
            mapped["n6"] = r["star_1"]
        
        # Dream
        if "dream" in r:
            mapped["dream"] = r["dream"]
        elif "star_2" in r and "n6" in mapped and mapped["n6"] == r.get("star_1"):
             # If we used star_1 as n6, star_2 is likely dream
             mapped["dream"] = r["star_2"]
        elif "star_1" in r and "n6" in mapped and mapped["n6"] != r["star_1"]:
             # regular case?
             mapped["dream"] = r["star_1"]
        elif "bonus" in r:
             mapped["dream"] = r["bonus"]
             
        # Check required
        if not {"date", "n1", "n2", "n3", "n4", "n5", "n6", "dream"}.issubset(mapped.keys()):
            # print(f"Skipping row missing keys: {mapped.keys()} vs required. Orig: {r}")
            continue
            
        try:
             # Fix date format D/M/Y -> YYYY-MM-DD
             d_str = mapped["date"]
             if "/" in d_str:
                 try:
                     parts = d_str.split("/")
                     if len(parts) == 3:
                         # assume D/M/Y if part[0] <= 31
                         d, m, y = map(int, parts)
                         d_iso = dt.date(y, m, d).isoformat()
                         mapped["date"] = d_iso
                 except ValueError:
                     pass

             rec = {
                "date": mapped["date"],
                "weekday": r.get("weekday") or _weekday_name(mapped["date"]),
                "n1": int(mapped["n1"]),
                "n2": int(mapped["n2"]),
                "n3": int(mapped["n3"]),
                "n4": int(mapped["n4"]),
                "n5": int(mapped["n5"]),
                "n6": int(mapped["n6"]),
                "dream": int(mapped["dream"]),
                "draw_code": r.get("draw_code", ""),
                "source_url": r.get("source_url", "user-supplied"),
            }
             out.append(rec)
        except Exception:
            continue
            
    return out



def main():
    ap = argparse.ArgumentParser(
        description="Compile EuroDreams draws (2023-present) into CSV/JSON (chronological)."
    )
    ap.add_argument(
        "--out", default="eurodreams_all.csv", help="Output file path (default: eurodreams_all.csv)"
    )
    ap.add_argument(
        "--format", choices=["csv", "json"], default="csv", help="Output format (default: csv)"
    )
    ap.add_argument("--start-year", type=int, default=2023, help="Start year (default: 2023)")
    ap.add_argument("--end-year", type=int, default=9999, help="End year (default: no upper bound)")
    ap.add_argument(
        "--source",
        choices=["auto", "irish", "euro", "lottery_ie"],
        default="auto",
        help="Source preference: auto (try all), irish (primary), euro (euromillions per-year), lottery_ie (recent only)",
    )
    ap.add_argument(
        "--allow-stale",
        action="store_true",
        default=True,
        help="If remote fetches fail, reuse an existing output file or a bundled sample dataset (default: on).",
    )
    ap.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Optional local CSV with columns date,n1..n6,dream[,draw_code,source_url] to ingest before fetching.",
    )
    ap.add_argument(
        "--csv-url",
        type=str,
        default=None,
        help="Optional CSV URL (same columns as --input-csv) to ingest before fetching.",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress warnings/summary; still prints errors on failure.",
    )
    args = ap.parse_args()

    recs: list[dict] = []
    errors: list[str] = []

    def _extend(fn_fetch, fn_parse, label):
        nonlocal recs
        try:
            html = fn_fetch()
            recs.extend(fn_parse(html))
        except Exception as e:
            errors.append(f"{label} failed: {e}")

    # Ingest user-provided CSV first (local or URL).
    if args.input_csv:
        try:
            recs.extend(_validate_structured_rows(_load_existing(args.input_csv)))
            errors.append(f"used input_csv {args.input_csv}")
        except Exception as e:
            errors.append(f"input_csv failed: {e}")
    if args.csv_url:
        try:
            r = requests.get(args.csv_url, timeout=20)
            r.raise_for_status()
            tmp_path = Path(".cache/eurodreams")
            tmp_path.mkdir(parents=True, exist_ok=True)
            tmp_file = tmp_path / "remote.csv"
            tmp_file.write_text(r.text, encoding="utf-8")
            recs.extend(_validate_structured_rows(_load_existing(str(tmp_file))))
            errors.append(f"used csv_url {args.csv_url}")
        except Exception as e:
            errors.append(f"csv_url failed: {e}")

    if args.source in ("auto", "irish"):
        _extend(fetch_irish_archive, parse_irish_archive, "irishlottery.com archive")

    if args.source in ("auto", "euro"):
        current_year = dt.date.today().year
        # Check from 2023 up to current year (inclusive)
        for y in range(2023, current_year + 1):
            try:
                html = fetch_euromillions_year(y)
                recs.extend(parse_euromillions_year(html, y))
            except Exception as e:
                errors.append(f"euro-millions.com {y} failed: {e}")

    if args.source in ("auto", "lottery_ie"):
        _extend(fetch_lottery_ie_recent, parse_lottery_ie_recent, "lottery.ie recent")

    if not recs and args.allow_stale:
        repo_root = Path(__file__).resolve().parents[1]
        fallback_candidates = [
            Path(args.out),
            Path("data/eurodreams.csv"),
            Path("data/eurodreams_draws_2023_to_2025.csv"),
            Path("data/examples/eurodreams_sample.csv"),
            repo_root / "data/eurodreams.csv",
            repo_root / "data/eurodreams_draws_2023_to_2025.csv",
            repo_root / "data/examples/eurodreams_sample.csv",
        ]
        seen = set()
        uniq_paths = []
        for p in fallback_candidates:
            if not p:
                continue
            try:
                key = p.resolve()
            except Exception:
                key = p
            if key in seen:
                continue
            seen.add(key)
            uniq_paths.append(p)

        for path in uniq_paths:
            if path.exists():
                try:
                    recs = _validate_structured_rows(_load_existing(str(path)))
                    errors.append(f"used fallback {path}")
                    break
                except Exception as e:  # pragma: no cover - fallback path
                    errors.append(f"fallback {path} failed: {e}")

    if not recs:
        raise SystemExit("No records fetched.\n" + "\n".join(errors))

    recs = dedupe_keep_best(recs)
    recs = filter_year_range(recs, args.start_year, args.end_year)

    if not recs:
        raise SystemExit("No records in requested year range.")

    if args.format == "csv":
        save_csv(recs, args.out)
    else:
        save_json(recs, args.out)

    if not args.quiet:
        print(f"OK: wrote {len(recs)} draws to {args.out} ({recs[0]['date']} -> {recs[-1]['date']}).")
        if errors:
            print('Warnings:\n  ' + '\n  '.join(errors))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Aborted.", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
