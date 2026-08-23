#!/usr/bin/env python3
"""
fetch_prizes.py
Fetch EuroMillions prize breakdown for a given draw (or the latest) from the UK National Lottery
and save a prizes.json mapping like {"5_2": amount, "5_1": amount, ..., "2_0": amount}.
Values are in GBP if scraped from the UK site. Keep your --ticket-cost in the same currency.

Examples:
  python3 fetch_prizes.py --latest --out prizes.json
  python3 fetch_prizes.py --draw-id 1867 --out prizes.json
  python3 fetch_prizes.py --url https://www.national-lottery.co.uk/results/euromillions/draw-history/prize-breakdown/1867 --out prizes.json
"""

import argparse
import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

HIST_URL = "https://www.national-lottery.co.uk/results/euromillions/draw-history"


def parse_money(s: str) -> float:
    s = s.strip().replace(",", "")
    m = re.search(r"([\d\.]+)", s)
    return float(m.group(1)) if m else 0.0


def scrape_breakdown(url: str):
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table")
    if not table:
        raise SystemExit("Could not find breakdown table on page.")
    data = {}
    draw_date = None
    title = soup.find(["h1", "h2"])
    if title:
        m = re.search(r"(\d{1,2}\s\w+\s\d{4})", title.get_text(" ", strip=True))
        if m:
            draw_date = m.group(1)
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
    meta = {"source_url": url, "currency": "GBP", "draw_date": draw_date}
    return data, meta


def find_latest_draw_id():
    r = requests.get(HIST_URL, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for a in soup.find_all("a", href=True):
        if "prize-breakdown" in a["href"]:
            m = re.search(r"/prize-breakdown/(\d+)", a["href"])
            if m:
                return int(m.group(1))
    raise SystemExit("No prize-breakdown links found.")


def main():
    ap = argparse.ArgumentParser(
        description="Fetch EuroMillions prize breakdown and save as prizes.json"
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--latest", action="store_true")
    g.add_argument("--draw-id", type=int)
    g.add_argument("--url", type=str)
    ap.add_argument("--out", type=str, default="prizes.json")
    args = ap.parse_args()
    if args.url:
        url = args.url
    else:
        draw_id = find_latest_draw_id() if args.latest else args.draw_id
        if not draw_id:
            raise SystemExit("Need --latest or --draw-id or --url.")
        url = f"https://www.national-lottery.co.uk/results/euromillions/draw-history/prize-breakdown/{draw_id}"
    data, meta = scrape_breakdown(url)
    Path(args.out).write_text(json.dumps({"meta": meta, "prizes": data}, indent=2))
    print(
        f"Wrote {args.out} with {len(data)} keys from {meta['source_url']} ({meta.get('draw_date')})"
    )


if __name__ == "__main__":
    main()
