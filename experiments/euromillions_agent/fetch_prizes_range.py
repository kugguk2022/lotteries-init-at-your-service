#!/usr/bin/env python3
"""
fetch_prizes_range.py
Build a multi-draw prize JSON for EuroMillions: {"steps":[{"prizes": {...}}, ...]}

Two modes:
  A) Latest N draws:
     python3 fetch_prizes_range.py --latest-count 200 --out prizes_multi.json
  B) Explicit draw ids:
     python3 fetch_prizes_range.py --ids 1860 1861 1862 --out prizes_multi.json
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

HIST_URL = "https://www.national-lottery.co.uk/results/euromillions/draw-history"


def parse_money(s: str) -> float:
    s = s.strip().replace(",", "")
    m = re.search(r"([\d\.]+)", s)
    return float(m.group(1)) if m else 0.0


def scrape_breakdown(draw_id: int):
    url = f"https://www.national-lottery.co.uk/results/euromillions/draw-history/prize-breakdown/{draw_id}"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table")
    if not table:
        raise RuntimeError(f"Breakdown table not found for draw {draw_id}")
    data = {}
    for tr in table.find_all("tr"):
        cols = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        if not cols or "Match" not in cols[0]:
            continue
        cat = cols[0]
        m = re.search(r"Match\s+(\d)(?:\s*\+\s*(\d)\s*Star[s]?)?", cat, flags=re.I)
        if not m:
            continue
        km = int(m.group(1))
        ks = int(m.group(2)) if m.group(2) else 0
        prize = 0.0
        for c in reversed(cols):
            if "£" in c or "€" in c:
                prize = parse_money(c)
                break
        data[f"{km}_{ks}"] = prize
    return {"draw_id": draw_id, "prizes": data, "source": url}


def find_latest_ids(n: int):
    # Crawl history page(s) to collect draw ids (limited by n). Assumes descending order.
    r = requests.get(HIST_URL, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    ids = []
    for a in soup.find_all("a", href=True):
        if "prize-breakdown" in a["href"]:
            m = re.search(r"/prize-breakdown/(\d+)", a["href"])
            if m:
                ids.append(int(m.group(1)))
    ids = sorted(set(ids))  # ascending
    return ids[-n:] if n < len(ids) else ids


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--latest-count", type=int, help="Fetch latest N draw ids")
    g.add_argument("--ids", type=int, nargs="+", help="Explicit draw ids")
    ap.add_argument("--out", type=str, default="prizes_multi.json")
    args = ap.parse_args()

    draw_ids = args.ids if args.ids else find_latest_ids(args.latest_count)
    draws = []
    for i, did in enumerate(draw_ids):
        try:
            rec = scrape_breakdown(did)
            draws.append(rec)
            time.sleep(0.2)  # be polite
        except Exception as e:
            print(f"[warn] failed draw {did}: {e}", file=sys.stderr)
    # Output chronological (oldest -> newest)
    draws = sorted(draws, key=lambda r: r["draw_id"])
    Path(args.out).write_text(
        json.dumps(
            {
                "steps": [
                    {"prizes": d["prizes"], "draw_id": d["draw_id"], "source": d["source"]}
                    for d in draws
                ]
            },
            indent=2,
        )
    )
    print(f"Wrote {args.out} with {len(draws)} steps")


if __name__ == "__main__":
    main()
