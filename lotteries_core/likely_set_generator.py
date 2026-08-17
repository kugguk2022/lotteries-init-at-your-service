"""likely_set_generator.py -- generalized, vectorized "co-occurrence level-set" ticket generator.

A clean, format-agnostic evolution of the EuroDreams script (the lineage: R pair-co-occurrence ->
`Edreams` 21-pair pooled level-set -> this). It works for ANY draw game of the form ``main_k + star_k``
(5+1, 5+2, 6+1, 6+2, 6+0, ...), fixes a token-collision bug, vectorizes the scoring so it runs in
seconds instead of hours, and offers a genuinely forward-only prediction mode.

------------------------------------------------------------------------------------------------
WHAT IT DOES
------------------------------------------------------------------------------------------------
1. From draw history it builds pairwise co-occurrence counts W over the numbers.
2. Every candidate ticket gets a score G(ticket) = sum of W over all pairs inside the ticket.
3. It returns the *level set*: all tickets whose G equals a `target` (within a tolerance), plus a
   top-N ranked view. The drawn ticket is always included in observed mode.

`target` comes from one of two modes:
* ``observed``  -- target = G of a specific drawn ticket (your original script's behavior). This
  *characterizes* the co-occurrence "siblings" of a draw you have already seen.
* ``predicted`` -- target = a causal, trailing-mean forecast of the next draw's score, built ONLY
  from draws before it (no look-ahead). This is the forward-only "n+1" generator.

------------------------------------------------------------------------------------------------
TWO CORRECTNESS FIXES over the original
------------------------------------------------------------------------------------------------
* **Separate number spaces.** Mains (1..main_n) and stars (1..star_n) live in different pools, so a
  main "3" and a star "3" are different tokens. `pairing="cross"` (default) still counts main-main,
  main-star and star-star pairs -- matching the "all 21 pairs" intent -- without conflating values.
  Use `pairing="pooled"` to reproduce the original (collision-prone) behavior, or `pairing="main"`
  for main-main only (the older R behavior).
* **Vectorized scoring.** Candidate scores are computed by numpy fancy-indexing over a precomputed
  W matrix; the star part is added per (small) star-combo without rescanning the mains.

------------------------------------------------------------------------------------------------
SCOPE (read docs/SCOPE.md in the repo)
------------------------------------------------------------------------------------------------
Under the fair-draw null hypothesis, past co-occurrence should not change a ticket's mechanical odds.
G is therefore an experimental ranking signal, not proof of a more probable winner. This tool exists
to generate portfolios before draws and test whether the signal beats matched random controls over
time. Positive, negative and inconclusive outcomes are all retained; it promises no wins and moves no
money.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations, islice
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

from .protocol import GameSpec, InferenceProvider, ProviderResult

# ------------------------------------------------------------------------------------------------
# Game configuration + presets
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GameConfig:
    name: str
    main_n: int
    main_k: int
    star_n: int = 0
    star_k: int = 0

    def n_main_combos(self) -> int:
        return comb(self.main_n, self.main_k)

    def n_star_combos(self) -> int:
        return comb(self.star_n, self.star_k) if self.star_k > 0 else 1


PRESETS: dict[str, GameConfig] = {
    "euromillions": GameConfig("euromillions", 50, 5, 12, 2),  # 5 + 2
    "eurodreams": GameConfig("eurodreams", 40, 6, 5, 1),       # 6 + 1
    "totoloto": GameConfig("totoloto", 49, 5, 13, 1),          # 5 + 1
    "5+2": GameConfig("5+2", 50, 5, 12, 2),
    "5+1": GameConfig("5+1", 49, 5, 10, 1),
    "6+1": GameConfig("6+1", 40, 6, 5, 1),
    "6+2": GameConfig("6+2", 50, 6, 12, 2),
}


# ------------------------------------------------------------------------------------------------
# Column detection
# ------------------------------------------------------------------------------------------------


def detect_columns(df: pd.DataFrame, cfg: GameConfig) -> tuple[list[str], list[str]]:
    cols = list(df.columns)
    main = [c for c in cols if str(c).lower().startswith(("ball_", "ball", "n"))][: cfg.main_k]
    star = [
        c for c in cols if str(c).lower().startswith(("star", "dream", "lucky", "bonus"))
    ][: cfg.star_k]
    if len(main) < cfg.main_k:
        # fall back to first numeric columns
        num = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all").columns.tolist()
        main = num[: cfg.main_k]
        if cfg.star_k:
            star = num[cfg.main_k : cfg.main_k + cfg.star_k]
    return main, star


# ------------------------------------------------------------------------------------------------
# Co-occurrence matrices (0-based indexing)
# ------------------------------------------------------------------------------------------------


@dataclass
class CoMatrices:
    Wmain: np.ndarray   # (main_n, main_n) main-main co-occurrence, symmetric, zero diagonal
    Wcross: np.ndarray  # (main_n, star_n) main-star co-occurrence
    Wstar: np.ndarray   # (star_n, star_n) star-star co-occurrence, symmetric


def build_comatrices(
    df: pd.DataFrame, cfg: GameConfig, main_cols: list[str], star_cols: list[str]
) -> CoMatrices:
    Wmain = np.zeros((cfg.main_n, cfg.main_n), dtype=np.int64)
    Wcross = np.zeros((cfg.main_n, max(cfg.star_n, 1)), dtype=np.int64)
    Wstar = np.zeros((max(cfg.star_n, 1), max(cfg.star_n, 1)), dtype=np.int64)

    mains = df[main_cols].apply(pd.to_numeric, errors="coerce")
    stars = df[star_cols].apply(pd.to_numeric, errors="coerce") if star_cols else None

    for i in range(len(df)):
        m = sorted({int(v) for v in mains.iloc[i].to_numpy() if 1 <= v <= cfg.main_n})
        for a, b in combinations(m, 2):
            Wmain[a - 1, b - 1] += 1
            Wmain[b - 1, a - 1] += 1
        if stars is not None:
            s = sorted({int(v) for v in stars.iloc[i].to_numpy() if 1 <= v <= cfg.star_n})
            for mv in m:
                for sv in s:
                    Wcross[mv - 1, sv - 1] += 1
            for a, b in combinations(s, 2):
                Wstar[a - 1, b - 1] += 1
                Wstar[b - 1, a - 1] += 1
    return CoMatrices(Wmain, Wcross, Wstar)


# ------------------------------------------------------------------------------------------------
# Ticket scoring
# ------------------------------------------------------------------------------------------------


def score_ticket(main0: tuple[int, ...], star0: tuple[int, ...], W: CoMatrices, pairing: str) -> int:
    """G for a single ticket given 0-based main/star indices."""
    g = 0
    for a, b in combinations(main0, 2):
        g += int(W.Wmain[a, b])
    if pairing in ("cross", "pooled") and star0:
        for m in main0:
            for s in star0:
                g += int(W.Wcross[m, s])
        for a, b in combinations(star0, 2):
            g += int(W.Wstar[a, b])
    return g


def _main_scores(cfg: GameConfig, W: CoMatrices, chunk: int = 250_000):
    """Yield (main_index_array, main_pair_score_array) chunks over all main combinations."""
    pos_pairs = list(combinations(range(cfg.main_k), 2))
    gen = combinations(range(cfg.main_n), cfg.main_k)
    while True:
        block = list(islice(gen, chunk))
        if not block:
            break
        arr = np.asarray(block, dtype=np.int32)
        gm = np.zeros(len(arr), dtype=np.int64)
        for a, b in pos_pairs:
            gm += W.Wmain[arr[:, a], arr[:, b]]
        yield arr, gm


def enumerate_all_main_scores(cfg: GameConfig, W: CoMatrices):
    """Return (all_main_arr, all_gm) held in memory (fine up to a few million combos)."""
    arrs, gms = [], []
    for arr, gm in _main_scores(cfg, W):
        arrs.append(arr)
        gms.append(gm)
    return np.vstack(arrs), np.concatenate(gms)


def observed_poi_series(
    df: pd.DataFrame, cfg: GameConfig, W: CoMatrices, main_cols, star_cols, pairing
) -> np.ndarray:
    """G of each actually-drawn ticket under W (used to form the target / its trend)."""
    mains = df[main_cols].apply(pd.to_numeric, errors="coerce").astype("Int64")
    stars = df[star_cols].apply(pd.to_numeric, errors="coerce").astype("Int64") if star_cols else None
    out = np.zeros(len(df), dtype=np.int64)
    for i in range(len(df)):
        m0 = tuple(int(v) - 1 for v in mains.iloc[i].to_numpy())
        s0 = tuple(int(v) - 1 for v in stars.iloc[i].to_numpy()) if stars is not None else ()
        out[i] = score_ticket(m0, s0, W, pairing)
    return out


# ------------------------------------------------------------------------------------------------
# Level-set / top-N generation for one target
# ------------------------------------------------------------------------------------------------


def generate_sets(
    cfg: GameConfig,
    W: CoMatrices,
    target: float,
    *,
    pairing: str = "cross",
    tolerance: float = 0.0,
    max_out: int = 5000,
    top_n: int = 0,
):
    """Return (level_set_tickets, ranked_tickets).

    level_set: tickets with |G - target| <= tolerance (up to max_out).
    ranked:    the top_n tickets by |G - target| ascending then G descending (0 disables).
    Tickets are ((mains...), (stars...)) with 1-based numbers.
    """
    all_main, all_gm = enumerate_all_main_scores(cfg, W)
    star_combos = (
        [tuple(s) for s in combinations(range(cfg.star_n), cfg.star_k)] if cfg.star_k > 0 else [()]
    )

    level: list[tuple] = []
    # For ranking we keep the best few by distance to target.
    ranked_pool_g: list[int] = []
    ranked_pool_ticket: list[tuple] = []

    for s0 in star_combos:
        # star contribution is constant across all mains for this star-combo
        if pairing in ("cross", "pooled") and s0:
            star_const = sum(int(W.Wstar[a, b]) for a, b in combinations(s0, 2))
            cross_cols = W.Wcross[:, list(s0)]  # (main_n, star_k)
        else:
            star_const = 0
            cross_cols = None

        # G for every main set with this star combo
        if cross_cols is not None:
            # Sum over the star axis first: colsum[m] = sum_{s in s0} Wcross[m, s]. Gathering after
            # the reduction gives an identical result while materialising (num_main, main_k) instead
            # of (num_main, main_k, star_k) -- the latter is 162 MiB per star combo for EuroMillions.
            colsum = cross_cols.sum(axis=1)  # (main_n,)
            cross = colsum[all_main].sum(axis=1)  # (num_main,)
        else:
            cross = 0
        g = all_gm + cross + star_const

        # level set
        if tolerance <= 0:
            hit = np.where(g == round(target))[0]
        else:
            hit = np.where(np.abs(g - target) <= tolerance)[0]
        for idx in hit[: max(0, max_out - len(level))]:
            mains = tuple(int(x) + 1 for x in all_main[idx])
            stars = tuple(int(x) + 1 for x in s0)
            level.append((mains, stars, int(g[idx])))
            if len(level) >= max_out:
                break

        # ranked pool (by closeness to target)
        if top_n > 0:
            dist = np.abs(g - target)
            k = min(top_n, len(g))
            best = np.argpartition(dist, k - 1)[:k]
            for idx in best:
                ranked_pool_g.append(int(g[idx]))
                ranked_pool_ticket.append(
                    (tuple(int(x) + 1 for x in all_main[idx]), tuple(int(x) + 1 for x in s0))
                )
        if len(level) >= max_out and top_n == 0:
            break

    ranked = []
    if top_n > 0 and ranked_pool_g:
        order = sorted(
            range(len(ranked_pool_g)),
            key=lambda j: (abs(ranked_pool_g[j] - target), -ranked_pool_g[j]),
        )[:top_n]
        ranked = [
            (ranked_pool_ticket[j][0], ranked_pool_ticket[j][1], ranked_pool_g[j]) for j in order
        ]
    return level, ranked


def _diversify(tickets: list[tuple], max_shared_main: int, limit: int) -> list[tuple]:
    """Greedy low-overlap filter over ranked tickets (keeps a spread portfolio)."""
    kept: list[tuple] = []
    kept_sets: list[set] = []
    for t in tickets:
        ms = set(t[0])
        if any(len(ms & prev) > max_shared_main for prev in kept_sets):
            continue
        kept.append(t)
        kept_sets.append(ms)
        if len(kept) >= limit:
            break
    if len(kept) < limit:  # top up ignoring overlap if diversification starved us
        for t in tickets:
            if t not in kept:
                kept.append(t)
                if len(kept) >= limit:
                    break
    return kept


def build_portfolio(
    df: pd.DataFrame,
    cfg: GameConfig,
    *,
    target_mode: str = "predicted",
    window: int = 26,
    pairing: str = "cross",
    n_sets: int = 20,
    max_shared_main: int | None = None,
    main_cols: list[str] | None = None,
    star_cols: list[str] | None = None,
) -> dict:
    """Programmatic API: return a bounded, diversified portfolio + the target used.

    Returns ``{"tickets": [(mains, stars), ...], "target": float, "main_cols":..., "star_cols":...}``.
    In ``predicted`` mode every supplied row is treated as already-known history. The caller must
    invoke this before the target draw is appended. No historical row is silently discarded.
    """
    if main_cols is None or star_cols is None:
        main_cols, star_cols = detect_columns(df, cfg)
    if target_mode == "observed":
        W = build_comatrices(df, cfg, main_cols, star_cols)
        poi = observed_poi_series(df, cfg, W, main_cols, star_cols, pairing)
        target = float(poi[-1]) if len(poi) else 0.0
    else:
        train = df
        W = build_comatrices(train, cfg, main_cols, star_cols)
        poi = observed_poi_series(train, cfg, W, main_cols, star_cols, pairing)
        target = float(np.mean(poi[-window:])) if len(poi) else 0.0

    _level, ranked = generate_sets(
        cfg, W, target, pairing=pairing, tolerance=0.0, max_out=1, top_n=max(n_sets * 6, n_sets)
    )
    if target_mode == "observed" and len(df):
        drawn_main = tuple(sorted(int(v) for v in df.iloc[-1][main_cols].to_numpy()))
        drawn_star = (
            tuple(sorted(int(v) for v in df.iloc[-1][star_cols].to_numpy())) if star_cols else ()
        )
        observed = (drawn_main, drawn_star, round(target))
        ranked = [observed] + [t for t in ranked if (t[0], t[1]) != (drawn_main, drawn_star)]
    if max_shared_main is None:
        max_shared_main = max(1, cfg.main_k - 2)
    portfolio = _diversify(ranked, max_shared_main, n_sets)
    tickets = [(t[0], t[1]) for t in portfolio]
    return {
        "tickets": tickets,
        "scored_tickets": portfolio,
        "target": target,
        "main_cols": main_cols,
        "star_cols": star_cols,
    }


class CooccurrenceLevelSetProvider(InferenceProvider):
    """Expose the owner's level-set method through the repository provider protocol.

    ``fit`` receives only the history available before the evaluated draw. ``propose`` ranks legal
    tickets by distance from the forecast co-occurrence level and returns a diversified fixed-budget
    portfolio. The score is evidence for evaluation, not a claim that draw odds changed.
    """

    name = "cooccurrence_level_set"
    description = "Forward-only pair-co-occurrence level-set generator with diversified selection."

    def __init__(self, pairing: str = "cross", window: int = 26) -> None:
        if pairing not in {"cross", "main", "pooled"}:
            raise ValueError("pairing must be cross, main, or pooled")
        self.pairing = pairing
        self.window = int(window)
        self._history: pd.DataFrame | None = None

    def fit(
        self, history: pd.DataFrame, spec: GameSpec | None = None
    ) -> CooccurrenceLevelSetProvider:
        self._history = history.copy()
        return self

    def propose(
        self, spec: GameSpec, budget: int, rng: np.random.Generator
    ) -> ProviderResult:
        del rng  # the ranked level set is deterministic for a fixed history/config
        if self._history is None:
            raise RuntimeError("CooccurrenceLevelSetProvider.propose called before fit()")
        cfg = GameConfig(spec.name, spec.main_n, spec.main_k, spec.star_n, spec.star_k)
        result = build_portfolio(
            self._history,
            cfg,
            target_mode="predicted",
            window=self.window,
            pairing=self.pairing,
            n_sets=budget,
        )
        scored = result["scored_tickets"]
        scores = np.asarray([-abs(float(g) - result["target"]) for _m, _s, g in scored])
        return ProviderResult(
            tickets=result["tickets"],
            scores=scores,
            diagnostics={
                "pairing": self.pairing,
                "window": self.window,
                "target": result["target"],
                "history_rows": len(self._history),
            },
        )


# ------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------


def resolve_config(args) -> GameConfig:
    if args.preset:
        base = PRESETS[args.preset]
        return GameConfig(
            base.name,
            args.main_n or base.main_n,
            args.main_k or base.main_k,
            args.star_n if args.star_n is not None else base.star_n,
            args.star_k if args.star_k is not None else base.star_k,
        )
    if not (args.main_n and args.main_k):
        raise SystemExit("Provide --preset OR both --main-n and --main-k (and star-n/star-k).")
    return GameConfig("custom", args.main_n, args.main_k, args.star_n or 0, args.star_k or 0)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history", required=True)
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--main-n", type=int)
    ap.add_argument("--main-k", type=int)
    ap.add_argument("--star-n", type=int)
    ap.add_argument("--star-k", type=int)
    ap.add_argument("--pairing", choices=["cross", "main", "pooled"], default="cross",
                    help="cross=main+star+cross pairs (default); main=main-main only; pooled=legacy conflated")
    ap.add_argument("--target-mode", choices=["observed", "predicted"], default="predicted")
    ap.add_argument("--window", type=int, default=26, help="trailing window for predicted target")
    ap.add_argument("--tolerance", type=float, default=0.0)
    ap.add_argument("--max-out", type=int, default=5000)
    ap.add_argument("--top-n", type=int, default=25)
    ap.add_argument("--out", default=None, help="CSV path for the generated sets")
    args = ap.parse_args(argv)

    cfg = resolve_config(args)
    df = pd.read_csv(args.history)
    main_cols, star_cols = detect_columns(df, cfg)
    print(f"[cfg] {cfg.name} {cfg.main_k}+{cfg.star_k} from ({cfg.main_n},{cfg.star_n}); "
          f"mains={main_cols} stars={star_cols}; main combos={cfg.n_main_combos():,}")

    if args.target_mode == "observed":
        # W built from all history; target = last drawn ticket's score (your original behavior).
        W = build_comatrices(df, cfg, main_cols, star_cols)
        poi = observed_poi_series(df, cfg, W, main_cols, star_cols, args.pairing)
        target = float(poi[-1])
    else:
        # Forward-only: the input contains only completed draws; predict the next unseen draw.
        train = df
        W = build_comatrices(train, cfg, main_cols, star_cols)
        poi = observed_poi_series(train, cfg, W, main_cols, star_cols, args.pairing)
        target = float(np.mean(poi[-args.window:])) if len(poi) else 0.0

    print(f"[target] mode={args.target_mode} target={target:.2f} tolerance={args.tolerance}")
    level, ranked = generate_sets(
        cfg, W, target, pairing=args.pairing, tolerance=args.tolerance,
        max_out=args.max_out, top_n=args.top_n,
    )
    print(f"[level-set] {len(level)} tickets with G within tolerance of target")
    if ranked:
        print(f"[top-{args.top_n}] closest tickets to target:")
        for mains, stars, g in ranked[:10]:
            print(f"   {mains} + {stars}  G={g}")

    rows = []
    for mains, stars, g in (level if level else ranked):
        rows.append(list(mains) + list(stars) + [g, round(target)])
    out_df = pd.DataFrame(rows)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out, index=False, header=False)
        print(f"[saved] {len(out_df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
