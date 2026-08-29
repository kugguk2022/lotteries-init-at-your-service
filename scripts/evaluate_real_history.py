"""Forward-only POI-G evaluation on a real, locally-cached draw history.

The synthetic community benchmark cannot test whether POI-G finds structure: its draws come from a
uniform RNG, so no method can exceed containment lift 1.0 on it in expectation. This script runs the
same forward-only protocol against a real history instead.

It measures **match depth**, not just exact containment. Exact containment of a 5-of-50 combination
in a 500-ticket shortlist has a random baseline of 0.024%, so it is the wrong instrument -- it
reports zero for a reducer that is doing real work. The tiered question "does the shortlist hold a
ticket matching at least 4 of the 5 drawn numbers, ignoring stars" is both answerable and the one
that corresponds to an actual prize tier.

Raw draw rows are READ locally and never written to ``publishing/``. Only derived aggregate metrics
leave this script, which keeps third-party history out of the redistributable bundles.
"""

from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

from lotteries_core.likely_set_generator import (
    GameConfig,
    build_comatrices,
    detect_columns,
    generate_sets,
    observed_poi_series,
)
from lotteries_core.protocol import GameSpec

sys.path.insert(0, str(Path(__file__).resolve().parent))
from poi_g_controls import frequency_matched_shortlist, match_rates  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY = ROOT / ".cache" / "euromillions" / "d06faf6752215252.csv"
DEFAULT_OUT = ROOT / "publishing" / "common" / "real_history_poi_g_results.csv"
SUBSET_SIZES = (20, 100, 500)
WINDOW = 26
PAIRING = "cross"
SEED = 20260828
CONTROL_REPEATS = 25
# EuroMillions ran 9 stars to 2011, 11 to 2016-09, 12 since. Mixing eras would score a shortlist
# built for one rule set against draws from another, so the default run uses the current era only.
TWELVE_STAR_ERA_START = "2016-09-24"


def match_depth_stats(shortlist_mains: np.ndarray, actual_main: set[int]) -> tuple[int, int, int]:
    """(best match count, tickets matching >=4, tickets matching >=3) over the shortlist."""
    hits = np.isin(shortlist_mains, list(actual_main)).sum(axis=1)
    return int(hits.max()), int((hits >= 4).sum()), int((hits >= 3).sum())


def random_at_least(k: int, spec: GameSpec) -> float:
    """P(one uniformly random ticket matches >= k of the drawn main numbers)."""
    total = comb(spec.main_n, spec.main_k)
    favourable = sum(
        comb(spec.main_k, m) * comb(spec.main_n - spec.main_k, spec.main_k - m)
        for m in range(k, spec.main_k + 1)
    )
    return favourable / total


def evaluate(history: pd.DataFrame, spec: GameSpec, holdout: int) -> pd.DataFrame:
    cfg = GameConfig(spec.name, spec.main_n, spec.main_k, spec.star_n, spec.star_k)
    main_cols, star_cols = detect_columns(history, cfg)
    max_size = max(SUBSET_SIZES)
    rng = np.random.default_rng(SEED)

    # Per-draw records keyed by subset size.
    records: dict[int, list[dict]] = {size: [] for size in SUBSET_SIZES}
    start = len(history) - holdout

    for position, index in enumerate(range(start, len(history)), start=1):
        train = history.iloc[:index]
        matrices = build_comatrices(train, cfg, main_cols, star_cols)
        poi = observed_poi_series(train, cfg, matrices, main_cols, star_cols, PAIRING)
        target = float(pd.Series(poi[-WINDOW:]).mean()) if len(poi) else 0.0
        # One ranking per draw, sliced per size: the top-n of a longer ranked list is the same
        # ranking, and re-ranking per size would triple an already 9s-per-draw enumeration.
        _level, ranked = generate_sets(
            cfg, matrices, target, pairing=PAIRING, max_out=0, top_n=max_size
        )
        row = history.iloc[index]
        actual_main = {int(row[column]) for column in main_cols}
        actual_star = tuple(sorted(int(row[column]) for column in star_cols))
        actual_main_key = tuple(sorted(actual_main))

        for size in SUBSET_SIZES:
            window = ranked[:size]
            mains = np.array([ticket[0] for ticket in window], dtype=np.int64)
            best, at_least_4, at_least_3 = match_depth_stats(mains, actual_main)
            main_keys = {tuple(sorted(ticket[0])) for ticket in window}
            full_keys = {(tuple(sorted(t[0])), tuple(sorted(t[1]))) for t in window}
            # POI-G's tickets reuse a favoured number pool, so they are not `size` independent
            # trials. Comparing only against a uniform baseline conflates "the pair ranking works"
            # with "the shortlist is concentrated". The frequency-matched control keeps the
            # shortlist's own marginals and destroys only the pairing, isolating the mechanism.
            control_hits = {3: 0, 4: 0}
            for _ in range(CONTROL_REPEATS):
                control = frequency_matched_shortlist(rng, mains, spec.main_n, spec.main_k)
                rates = match_rates(control, actual_main, thresholds=(3, 4))
                control_hits[3] += rates[3] > 0
                control_hits[4] += rates[4] > 0
            records[size].append(
                {
                    "best_match": best,
                    "hits_ge4": at_least_4,
                    "hits_ge3": at_least_3,
                    "has_ge4": float(at_least_4 > 0),
                    "has_ge3": float(at_least_3 > 0),
                    "ctrl_ge4": control_hits[4] / CONTROL_REPEATS,
                    "ctrl_ge3": control_hits[3] / CONTROL_REPEATS,
                    "exact_main": float(actual_main_key in main_keys),
                    "exact_full": float((actual_main_key, actual_star) in full_keys),
                    "distinct_mains": len(main_keys),
                }
            )
        if position % 10 == 0:
            print(f"  {position}/{holdout} draws", flush=True)

    main_universe = comb(spec.main_n, spec.main_k)
    p4 = random_at_least(4, spec)
    p3 = random_at_least(3, spec)
    rows = []
    for size in SUBSET_SIZES:
        frame = pd.DataFrame(records[size])
        distinct = frame["distinct_mains"].mean()
        rows.append(
            {
                "subset_size": size,
                "holdout": holdout,
                "main_universe_size": main_universe,
                "mean_distinct_main_sets": distinct,
                "mean_best_match": frame["best_match"].mean(),
                "max_best_match": int(frame["best_match"].max()),
                "draws_with_ge4": int(frame["has_ge4"].sum()),
                "rate_ge4": frame["has_ge4"].mean(),
                "random_rate_ge4": 1.0 - (1.0 - p4) ** size,
                "lift_ge4": frame["has_ge4"].mean() / (1.0 - (1.0 - p4) ** size),
                "draws_with_ge3": int(frame["has_ge3"].sum()),
                "rate_ge3": frame["has_ge3"].mean(),
                "random_rate_ge3": 1.0 - (1.0 - p3) ** size,
                "lift_ge3": frame["has_ge3"].mean() / (1.0 - (1.0 - p3) ** size),
                "mean_tickets_ge4": frame["hits_ge4"].mean(),
                "random_mean_tickets_ge4": size * p4,
                "freqmatched_rate_ge4": frame["ctrl_ge4"].mean(),
                "lift_ge4_vs_freqmatched": (
                    frame["has_ge4"].mean() / frame["ctrl_ge4"].mean()
                    if frame["ctrl_ge4"].mean() else float("nan")
                ),
                "freqmatched_rate_ge3": frame["ctrl_ge3"].mean(),
                "lift_ge3_vs_freqmatched": (
                    frame["has_ge3"].mean() / frame["ctrl_ge3"].mean()
                    if frame["ctrl_ge3"].mean() else float("nan")
                ),
                "exact_main_hits": int(frame["exact_main"].sum()),
                "random_exact_main_rate": distinct / main_universe,
                "exact_full_hits": int(frame["exact_full"].sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--holdout", type=int, default=120)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--since", default=TWELVE_STAR_ERA_START)
    args = parser.parse_args()

    history = pd.read_csv(args.history)
    history = history[pd.to_datetime(history["draw_date"]) >= args.since].reset_index(drop=True)
    spec = GameSpec("euromillions", main_n=50, main_k=5, star_n=12, star_k=2)
    print(
        f"history: {len(history)} draws {history['draw_date'].iloc[0]} -> "
        f"{history['draw_date'].iloc[-1]} | holdout {args.holdout}",
        flush=True,
    )

    frame = evaluate(history, spec, args.holdout)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False, lineterminator="\n")

    provenance = {
        "game": "euromillions",
        "era_start": args.since,
        "draws_available": int(len(history)),
        "holdout": args.holdout,
        "first_draw": str(history["draw_date"].iloc[0]),
        "last_draw": str(history["draw_date"].iloc[-1]),
        "window": WINDOW,
        "pairing": PAIRING,
        "note": (
            "Derived metrics only. The underlying draw history is third-party data held locally "
            "and is deliberately not redistributed in this repository."
        ),
    }
    args.out.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
