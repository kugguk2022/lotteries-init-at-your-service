#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

# Sobol preferred (SciPy). Fallback uses a simple Halton (still low discrepancy).
try:
    from scipy.stats import qmc  # type: ignore
    SOBOL_OK = True
except Exception:
    SOBOL_OK = False


# --------------------------
# Combinadic rank / unrank
# --------------------------
def nCk(n: int, k: int) -> int:
    return math.comb(n, k)

def unrank_comb_lex(n: int, k: int, r: int) -> list[int]:
    """Unrank r in [0, C(n,k)) to a 1-based sorted combination (lex order by first element)."""
    if r < 0 or r >= nCk(n, k):
        raise ValueError("rank out of range")
    out = []
    x = 1
    for i in range(1, k + 1):
        while True:
            c = nCk(n - x, k - i) if (n - x) >= (k - i) else 0
            if c <= r:
                r -= c
                x += 1
            else:
                break
        out.append(x)
        x += 1
    return out

def rank_comb_lex(n: int, k: int, comb_list: list[int]) -> int:
    """Rank a 1-based sorted combination into [0, C(n,k))."""
    if len(comb_list) != k or sorted(comb_list) != comb_list:
        raise ValueError("comb must be sorted length k")
    r = 0
    prev = 0
    for i, a in enumerate(comb_list, start=1):
        for x in range(prev + 1, a):
            c = nCk(n - x, k - i) if (n - x) >= (k - i) else 0
            r += c
        prev = a
    return r

def rank_ticket(main: list[int], main_n: int, main_k: int,
                star: list[int] | None, star_n: int, star_k: int) -> int:
    """Map full ticket (mains + optional stars) into 1D index."""
    midx = rank_comb_lex(main_n, main_k, main)
    if star_k <= 0 or star is None:
        return midx
    sidx = rank_comb_lex(star_n, star_k, star)
    return midx * nCk(star_n, star_k) + sidx

def unrank_ticket(idx: int, main_n: int, main_k: int,
                  star_n: int, star_k: int) -> tuple[list[int], list[int] | None]:
    """Inverse of rank_ticket."""
    if star_k <= 0:
        return unrank_comb_lex(main_n, main_k, idx), None
    base = nCk(star_n, star_k)
    midx = idx // base
    sidx = idx % base
    main = unrank_comb_lex(main_n, main_k, midx)
    star = unrank_comb_lex(star_n, star_k, sidx)
    return main, star


# --------------------------
# Low discrepancy samplers
# --------------------------
def _van_der_corput(i: int, base: int) -> float:
    vdc, denom = 0.0, 1.0
    while i:
        i, rem = divmod(i, base)
        denom *= base
        vdc += rem / denom
    return vdc

def halton_2d(n: int, start: int = 1) -> np.ndarray:
    """Fallback if SciPy Sobol isn't available."""
    pts = np.zeros((n, 2), dtype=float)
    for t in range(n):
        pts[t, 0] = _van_der_corput(start + t, 2)
        pts[t, 1] = _van_der_corput(start + t, 3)
    return pts

def sobol_2d(n: int, seed: int = 0) -> np.ndarray:
    if SOBOL_OK:
        eng = qmc.Sobol(d=2, scramble=True, seed=seed)
        return eng.random(n)
    return halton_2d(n, start=seed + 1)


# --------------------------
# R-to-Python feature engine
# (pair co-occurrence history -> output matrix -> poi)
# --------------------------
@dataclass
class FeatureOut:
    output_pairs: np.ndarray   # (T, 10) pair-counts for each draw's 10 pairs
    poi: np.ndarray            # (T,) sum of output_pairs rows
    g: np.ndarray              # (T,) Euler phi of time index 1..T
    g1: np.ndarray             # (T,) fitted baseline E[poi|g] (GLM/linear)
    resid: np.ndarray          # (T,) poi - g1
    pair_counts: np.ndarray    # (main_n+1, main_n+1) final pair count matrix

def euler_phi_upto(m: int) -> np.ndarray:
    phi = np.arange(m + 1)
    for i in range(2, m + 1):
        if phi[i] == i:
            phi[i::i] -= phi[i::i] // i
    return phi[1:]  # 1..m

def load_int_draws(path: Path, main_k: int = 5, star_k: int = 2) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Loads a CSV file of draws. 
    Robustly handles headers, date columns, and column names like 'ball_1', 'n1', etc.
    """
    try:
        # Try reading with header first
        df = pd.read_csv(path)
    except Exception:
        # Fallback for completely messy files
        df = pd.read_csv(path, header=None)
    
    # Identify numeric columns
    # We prefer columns named like ball_*, n*, star_*, dream, etc.
    
    col_map = {c: str(c).lower().strip() for c in df.columns}
    
    mains = []
    stars = []
    
    # Heuristic for main balls: ball_1..ball_N or n1..nN
    # Try ball_X pattern
    for i in range(1, main_k + 1):
        pat = f"ball_{i}"
        found = next((k for k, v in col_map.items() if v == pat or v == f"ball{i}"), None)
        if found:
            mains.append(found)
        else:
            # Try nX pattern
            pat_n = f"n{i}"
            found_n = next((k for k, v in col_map.items() if v == pat_n), None)
            if found_n:
                mains.append(found_n)
    
    # Heuristic for stars: star_X or lucky_star_X or dream
    for i in range(1, star_k + 1):
        pat = f"star_{i}"
        found = next((k for k, v in col_map.items() if v == pat or v == f"star{i}" or v == f"lucky_star_{i}" or v == f"lucky_star{i}"), None)
        if found:
            stars.append(found)
        else:
             # dream? (usually single star)
             if star_k == 1 and i == 1:
                 found_d = next((k for k, v in col_map.items() if v == "dream"), None)
                 if found_d:
                     stars.append(found_d)
                     
    if len(mains) != main_k or (star_k > 0 and len(stars) != star_k):
        # Fallback: find ALL numeric columns
        # Drop columns that are definitely dates/strings
        df_num = df.select_dtypes(include=[np.number])
        
        # If select_dtypes failed to find enough (maybe read as object due to dirty data), force coerce everything
        if df_num.shape[1] < main_k + star_k:
             df_coerced = df.apply(pd.to_numeric, errors='coerce')
             # Drop columns that are all NaN (likely dates/strings)
             df_num = df_coerced.dropna(axis=1, how='all')
        
        # Now take first main_k as mains, next star_k as stars
        if df_num.shape[1] >= main_k:
            cols = df_num.columns.tolist()
            mains = cols[:main_k]
            if star_k > 0 and len(cols) >= main_k + star_k:
                stars = cols[main_k : main_k + star_k]
            else:
                stars = [] # Not enough stars found
    
    if len(mains) < main_k:
        raise ValueError(f"Could not identify {main_k} main ball columns in {path}")
        
    df_mains = df[mains].apply(pd.to_numeric, errors='coerce').dropna().astype(int)
    mains_arr = df_mains.to_numpy()
    
    stars_arr = None
    if star_k > 0:
        if len(stars) == star_k:
            # Align indices with mains
            df_stars = df[stars].loc[df_mains.index].apply(pd.to_numeric, errors='coerce').astype(int)
            stars_arr = df_stars.to_numpy()
        else:
             # stars_arr will remain None — not enough star columns found
             stars_arr = None
    
    return mains_arr, stars_arr


def build_pair_features(mains: np.ndarray, main_n: int | None = None, include_current: bool = True) -> FeatureOut:
    """
    Replicates your R logic:
      For each time t, compute co-occurrence counts F(i,j) over history,
      then for the current draw emit the 10 F-values for its 10 pairs -> output[t, :]
      poi[t] = sum(output[t, :])
    The include_current=True matches your R behavior (counts include current draw).
    """
    mains = np.asarray(mains, dtype=int)
    T, k = mains.shape
    if k != 5:
        raise ValueError("This feature code mirrors your R: 5 mains per draw. (Extend if needed.)")

    main_n = int(main_n) if main_n is not None else int(mains.max())
    M = np.zeros((main_n + 1, main_n + 1), dtype=np.int32)

    pair_list = list(combinations(range(5), 2))  # 10 pairs by position
    out = np.zeros((T, len(pair_list)), dtype=np.int32)

    for t in range(T):
        draw = np.sort(mains[t])
        if include_current:
            for a, b in combinations(draw.tolist(), 2):
                M[a, b] += 1
                M[b, a] += 1
        # emit 10 pair counts for this draw
        for j, (i1, i2) in enumerate(pair_list):
            a, b = int(draw[i1]), int(draw[i2])
            out[t, j] = M[a, b]
        if not include_current:
            for a, b in combinations(draw.tolist(), 2):
                M[a, b] += 1
                M[b, a] += 1

    poi = out.sum(axis=1).astype(np.float64)
    g = euler_phi_upto(T).astype(np.float64)  # time feature

    # "GLM": your R glm(poi ~ g) defaults to gaussian => linear regression baseline
    X = np.c_[np.ones(T), g]
    beta, *_ = np.linalg.lstsq(X, poi, rcond=None)
    g1 = (X @ beta).astype(np.float64)
    resid = (poi - g1).astype(np.float64)

    return FeatureOut(output_pairs=out, poi=poi, g=g, g1=g1, resid=resid, pair_counts=M)


# --------------------------
# Sobol ticket generation (low overlap coverage)
# --------------------------
def ticket_overlap(a_main: list[int], a_star: list[int] | None,
                   b_main: list[int], b_star: list[int] | None) -> tuple[int, int]:
    shared_main = len(set(a_main) & set(b_main))
    shared_star = len(set(a_star or []) & set(b_star or []))
    return shared_main, shared_star

def select_low_overlap(cands: list[tuple[list[int], list[int] | None]], n_pick: int,
                       max_shared_main: int = 2, max_shared_star: int = 1) -> list[tuple[list[int], list[int] | None]]:
    """
    Greedy filter: keep tickets that don't overlap too much with those already selected.
    """
    chosen: list[tuple[list[int], list[int] | None]] = []
    for m, s in cands:
        ok = True
        for cm, cs in chosen:
            sm, ss = ticket_overlap(m, s, cm, cs)
            if sm > max_shared_main or ss > max_shared_star:
                ok = False
                break
        if ok:
            chosen.append((m, s))
            if len(chosen) >= n_pick:
                break
    return chosen

def generate_sobol_tickets(n_tickets: int,
                           main_n: int, main_k: int,
                           star_n: int, star_k: int,
                           seed: int = 0,
                           oversample: int = 12,
                           max_shared_main: int = 2,
                           max_shared_star: int = 1) -> list[tuple[list[int], list[int] | None]]:
    """
    Uses 2D Sobol/low-discrepancy points to sample combination indices:
      u0 -> main combo index in [0, C(main_n,main_k))
      u1 -> star combo index in [0, C(star_n,star_k))
    This reduces duplicates/collisions vs naive random sampling and helps space coverage.
    """
    total_main = nCk(main_n, main_k)
    total_star = nCk(star_n, star_k) if star_k > 0 else 1
    total_main * total_star

    pts = sobol_2d(n_tickets * oversample, seed=seed)

    seen = set()
    cands: list[tuple[list[int], list[int] | None]] = []
    for u0, u1 in pts:
        midx = min(total_main - 1, int(u0 * total_main))
        sidx = min(total_star - 1, int(u1 * total_star)) if star_k > 0 else 0
        idx = midx * total_star + sidx
        if idx in seen:
            continue
        seen.add(idx)
        m, s = unrank_ticket(idx, main_n, main_k, star_n, star_k)
        cands.append((m, s))
        if len(cands) >= n_tickets * oversample:
            break

    # Greedy low-overlap selection
    chosen = select_low_overlap(cands, n_tickets, max_shared_main=max_shared_main, max_shared_star=max_shared_star)
    return chosen


# --------------------------
# CLI
# --------------------------
def main():
    ap = argparse.ArgumentParser(description="Phase 2: R-style poi/g1 + Sobol combinadic ticket generation.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    fe = sub.add_parser("features", help="Compute output_pairs/poi/g1/resid from a draws file.")
    fe.add_argument("--infile", required=True, type=Path)
    fe.add_argument("--outdir", required=True, type=Path)
    fe.add_argument("--main-k", type=int, default=5)
    fe.add_argument("--star-k", type=int, default=2)
    fe.add_argument("--main-n", type=int, default=None)
    fe.add_argument("--include-current", action="store_true", help="Match your R behavior (counts include current draw).")

    tk = sub.add_parser("tickets", help="Generate low-overlap tickets using Sobol + combinadic unranking.")
    tk.add_argument("--out", required=True, type=Path)
    tk.add_argument("--n", type=int, default=50)
    tk.add_argument("--main-n", type=int, default=50)
    tk.add_argument("--main-k", type=int, default=5)
    tk.add_argument("--star-n", type=int, default=12)
    tk.add_argument("--star-k", type=int, default=2)
    tk.add_argument("--seed", type=int, default=0)
    tk.add_argument("--oversample", type=int, default=12)
    tk.add_argument("--max-shared-main", type=int, default=2)
    tk.add_argument("--max-shared-star", type=int, default=1)

    args = ap.parse_args()

    if args.cmd == "features":
        mains, _stars = load_int_draws(args.infile, main_k=args.main_k, star_k=args.star_k)
        out = build_pair_features(mains, main_n=args.main_n, include_current=bool(args.include_current))
        args.outdir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(out.output_pairs).to_csv(args.outdir / "output_pairs.csv", index=False, header=False)
        pd.Series(out.poi).to_csv(args.outdir / "poi.csv", index=False, header=False)
        pd.Series(out.g).to_csv(args.outdir / "g.csv", index=False, header=False)
        pd.Series(out.g1).to_csv(args.outdir / "g1.csv", index=False, header=False)
        pd.Series(out.resid).to_csv(args.outdir / "resid.csv", index=False, header=False)
        print(f"Wrote: output_pairs.csv, poi.csv, g.csv, g1.csv, resid.csv -> {args.outdir}")
        return

    if args.cmd == "tickets":
        tickets = generate_sobol_tickets(
            n_tickets=args.n,
            main_n=args.main_n, main_k=args.main_k,
            star_n=args.star_n, star_k=args.star_k,
            seed=args.seed,
            oversample=args.oversample,
            max_shared_main=args.max_shared_main,
            max_shared_star=args.max_shared_star,
        )
        rows = []
        for m, s in tickets:
            row = m + (s or [])
            rows.append(row)
        df = pd.DataFrame(rows)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False, header=False)
        print(f"Wrote {len(df)} tickets -> {args.out}")
        if not SOBOL_OK:
            print("[note] SciPy Sobol not available; used Halton fallback (still low-discrepancy).")
        return


if __name__ == "__main__":
    main()
