#!/usr/bin/env python3
"""outcome_tracker.py -- forward-only, long-horizon evidence ledger for the lottery repo.

Purpose (stated plainly): this repo's hypothesis is that coordinated, coverage-and-co-occurrence-aware
set selection produces *better tracked outcomes than random chance*. That is an empirical claim, and
the only honest way to judge it is to record predictions BEFORE each draw and score them AFTER, over a
long window (the owner's plan: ~3 years), then read off whether there is a consistent edge -- and PARK
the project if there is not. This module is that ledger.

It never bets, never handles money, and makes no promise of winnings. Against a fair draw the expected
edge is zero; the point is to measure, not to assume. Every draw is also scored against a matched
uniform-random control generated from the same draw key, so "did the method beat luck?" is answerable.

Workflow
--------
    # 1. Before a draw: generate and log this draw's portfolio (history must NOT contain the draw).
    python -m lotteries_core.outcome_tracker record --history history.csv --preset euromillions \
        --draw-key 2026-08-18 --ledger ./ledger --n-sets 20

    # 2. After the draw: settle it with the official result.
    python -m lotteries_core.outcome_tracker settle --ledger ./ledger --draw-key 2026-08-18 \
        --actual-main 4,17,23,38,45 --actual-stars 3,9

    # 3. Any time: read the cumulative verdict.
    python -m lotteries_core.outcome_tracker report --ledger ./ledger
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .likely_set_generator import PRESETS, GameConfig, build_portfolio, resolve_config

PENDING = "pending_predictions.jsonl"
SETTLED = "settled_predictions.jsonl"
RESULTS = "results.csv"


# ------------------------------------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------------------------------------


def _score_portfolio(tickets, actual_main: set[int], actual_stars: set[int], cfg: GameConfig) -> dict:
    best_main = 0
    best_star = 0
    mean_main = 0.0
    jackpot = False
    top_tier = (0, 0)
    for mains, stars in tickets:
        mh = len(set(mains) & actual_main)
        sh = len(set(stars) & actual_stars)
        best_main = max(best_main, mh)
        best_star = max(best_star, sh)
        mean_main += mh
        if (mh, sh) > top_tier:
            top_tier = (mh, sh)
        if mh == cfg.main_k and sh == cfg.star_k:
            jackpot = True
    n = max(len(tickets), 1)
    return {
        "n_sets": len(tickets),
        "best_main": best_main,
        "best_star": best_star,
        "mean_main": mean_main / n,
        "top_tier_main": top_tier[0],
        "top_tier_star": top_tier[1],
        "jackpot": int(jackpot),
    }


def _random_control(cfg: GameConfig, n_sets: int, draw_key: str) -> list[tuple]:
    seed = int(abs(hash(("control", draw_key))) % (2**32))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_sets):
        mains = tuple(sorted(int(x) for x in rng.choice(np.arange(1, cfg.main_n + 1), cfg.main_k, replace=False)))
        stars = (
            tuple(sorted(int(x) for x in rng.choice(np.arange(1, cfg.star_n + 1), cfg.star_k, replace=False)))
            if cfg.star_k > 0 else ()
        )
        out.append((mains, stars))
    return out


# ------------------------------------------------------------------------------------------------
# Ledger IO
# ------------------------------------------------------------------------------------------------


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rewrite_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records), encoding="utf-8")


# ------------------------------------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------------------------------------


def cmd_record(args) -> None:
    cfg = resolve_config(args)
    ledger = Path(args.ledger)
    df = pd.read_csv(args.history)
    pf = build_portfolio(
        df, cfg, target_mode=args.target_mode, window=args.window,
        pairing=args.pairing, n_sets=args.n_sets,
    )
    record = {
        "draw_key": args.draw_key,
        "game": cfg.name,
        "config": asdict(cfg),
        "method": f"cooccurrence:{args.target_mode}:{args.pairing}",
        "target": pf["target"],
        "n_sets": len(pf["tickets"]),
        "tickets": [[list(m), list(s)] for m, s in pf["tickets"]],
        "generated_from_rows": int(len(df)),
    }
    # guard against double-recording the same draw key
    pending = _read_jsonl(ledger / PENDING)
    if any(r["draw_key"] == args.draw_key and r["method"] == record["method"] for r in pending):
        print(f"[skip] draw_key {args.draw_key} already recorded for this method")
        return
    _append_jsonl(ledger / PENDING, record)
    print(f"[record] {record['n_sets']} sets logged for draw {args.draw_key} "
          f"({cfg.name}, target={pf['target']:.1f}) -> {ledger / PENDING}")


def _parse_nums(s: str) -> set[int]:
    return {int(x) for x in str(s).replace(" ", "").split(",") if x != ""}


def cmd_settle(args) -> None:
    ledger = Path(args.ledger)
    pending = _read_jsonl(ledger / PENDING)
    actual_main = _parse_nums(args.actual_main)
    actual_stars = _parse_nums(args.actual_stars) if args.actual_stars else set()

    remaining, settled_now = [], []
    for rec in pending:
        if rec["draw_key"] != args.draw_key:
            remaining.append(rec)
            continue
        cfg = GameConfig(**rec["config"])
        tickets = [(tuple(m), tuple(s)) for m, s in rec["tickets"]]
        method = _score_portfolio(tickets, actual_main, actual_stars, cfg)
        control_tickets = _random_control(cfg, len(tickets), rec["draw_key"])
        control = _score_portfolio(control_tickets, actual_main, actual_stars, cfg)
        row = {
            "draw_key": rec["draw_key"],
            "game": rec["game"],
            "method": rec["method"],
            "n_sets": method["n_sets"],
            "m_best_main": method["best_main"],
            "m_best_star": method["best_star"],
            "m_mean_main": round(method["mean_main"], 4),
            "m_top_tier": f"{method['top_tier_main']}+{method['top_tier_star']}",
            "m_jackpot": method["jackpot"],
            "c_best_main": control["best_main"],
            "c_mean_main": round(control["mean_main"], 4),
            "c_jackpot": control["jackpot"],
            "lift_mean_main": round(method["mean_main"] - control["mean_main"], 4),
        }
        _append_results(ledger / RESULTS, row)
        rec["actual_main"] = sorted(actual_main)
        rec["actual_stars"] = sorted(actual_stars)
        settled_now.append(rec)

    if not settled_now:
        print(f"[settle] no pending predictions for draw_key {args.draw_key}")
        return
    for rec in settled_now:
        _append_jsonl(ledger / SETTLED, rec)
    _rewrite_jsonl(ledger / PENDING, remaining)
    print(f"[settle] scored {len(settled_now)} prediction(s) for {args.draw_key}; "
          f"appended to {ledger / RESULTS}")


def _append_results(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([row])
    header = not path.exists()
    df_row.to_csv(path, mode="a", header=header, index=False)


def _permutation_pvalue(diffs: np.ndarray, iters: int = 10000, seed: int = 0) -> float:
    """One-sided p-value that mean(diffs) > 0 via sign-flip permutation (paired test)."""
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    observed = diffs.mean()
    count = 0
    for _ in range(iters):
        signs = rng.choice([-1.0, 1.0], size=len(diffs))
        if (signs * np.abs(diffs)).mean() >= observed:
            count += 1
    return (count + 1) / (iters + 1)


def cmd_report(args) -> None:
    ledger = Path(args.ledger)
    path = ledger / RESULTS
    if not path.exists():
        print("[report] no settled results yet. Record and settle some draws first.")
        return
    df = pd.read_csv(path)
    n = len(df)
    diffs = df["lift_mean_main"].to_numpy(dtype=float)
    method_best = df["m_best_main"].mean()
    control_best = df["c_best_main"].mean()
    mean_lift = float(np.nanmean(diffs))
    pval = _permutation_pvalue(diffs)
    jackpots = int(df["m_jackpot"].sum())

    print("=" * 60)
    print(f"Outcome report  ({ledger})")
    print("=" * 60)
    print(f"settled draws              : {n}")
    print(f"method mean best-main hits : {method_best:.3f}")
    print(f"control mean best-main hits: {control_best:.3f}")
    print(f"mean per-draw lift (method-control, mean-main): {mean_lift:+.4f}")
    print(f"one-sided permutation p (lift>0): {pval:.4f}")
    print(f"method jackpots            : {jackpots}")
    print("-" * 60)
    print("verdict:", _verdict(n, mean_lift, pval))
    print("=" * 60)


def _verdict(n: int, mean_lift: float, pval: float) -> str:
    if n < 30:
        return f"INSUFFICIENT DATA ({n} draws) — keep tracking; a 3-year window is the goal."
    if np.isnan(pval):
        return "INSUFFICIENT DATA — keep tracking."
    if mean_lift <= 0:
        return "NO EDGE vs random so far — trending toward PARK if this persists."
    if pval < 0.05:
        return "POSITIVE SIGNAL (p<0.05) — keep tracking to confirm stability over the full window."
    return "WEAK/INCONCLUSIVE positive — keep tracking; not yet distinguishable from luck."


# ------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------


def _add_game_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--preset", choices=sorted(PRESETS))
    p.add_argument("--main-n", type=int)
    p.add_argument("--main-k", type=int)
    p.add_argument("--star-n", type=int)
    p.add_argument("--star-k", type=int)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    rec = sub.add_parser("record", help="Generate and log a draw's portfolio (before the draw).")
    rec.add_argument("--history", required=True)
    rec.add_argument("--draw-key", required=True)
    rec.add_argument("--ledger", default="./ledger")
    rec.add_argument("--n-sets", type=int, default=20)
    rec.add_argument("--target-mode", choices=["observed", "predicted"], default="predicted")
    rec.add_argument("--pairing", choices=["cross", "main", "pooled"], default="cross")
    rec.add_argument("--window", type=int, default=26)
    _add_game_args(rec)
    rec.set_defaults(func=cmd_record)

    st = sub.add_parser("settle", help="Score a logged draw against the official result (after the draw).")
    st.add_argument("--ledger", default="./ledger")
    st.add_argument("--draw-key", required=True)
    st.add_argument("--actual-main", required=True, help="comma-separated main numbers")
    st.add_argument("--actual-stars", default="", help="comma-separated star numbers (optional)")
    st.set_defaults(func=cmd_settle)

    rep = sub.add_parser("report", help="Cumulative verdict across all settled draws.")
    rep.add_argument("--ledger", default="./ledger")
    rep.set_defaults(func=cmd_report)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
