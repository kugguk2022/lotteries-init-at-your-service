#!/usr/bin/env python3
"""
lotto_lab.py — One-file lab for EuroMillions
Agent (forecaster) + Discriminator (pair-cooccurrence) + Grok (tiny transformer) + RL Mixer (CEM).

Highlights
- Forward-only splits (no leakage): agent train/TEST, grok train/TEST matching agent, mixer re-splits TEST to VAL/HOLDOUT.
- Grok: OneCycleLR schedule + LR plot, train/val loss plots, robust to tiny data (falls back & still writes plots).
- Quality checks: reliability (calibration) plot + Brier score, permutation null histogram for mean-hits.
- Ablations: remove part of the dataset (head/tail/middle/random) to stress test like classic DL.
- Mixer objectives: mean hits (default) or ROI (needs prize table JSON).

Usage
  python3 lotto_lab.py --csv /path/euromillions.csv --mode all --outdir lotto_lab_out --debug
  python3 lotto_lab.py --csv /path/euromillions.csv --mode all --prizes /path/prizes.json --use-roi --outdir lotto_lab_out

Author: you + a friendly assistant.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Optional SciPy for chi-square p-values
try:
    from scipy.stats import chi2

    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# Optional torch for grokking
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    TORCH_OK = True
except Exception:
    TORCH_OK = False


# ---------------------- Utils ----------------------
def is_prime_vec(n: int) -> np.ndarray:
    is_p = np.ones(n + 1, dtype=bool)
    is_p[:2] = False
    for p in range(2, int(n**0.5) + 1):
        if is_p[p]:
            is_p[p * p : n + 1 : p] = False
    return is_p[1:]


def smart_load_draws(csv_path: str):
    """Load CSV (with/without header) and split main vs stars by max value (<=12 -> star)."""
    peek = pd.read_csv(csv_path, nrows=1, header=None)
    first7 = peek.iloc[0, : min(7, peek.shape[1])].astype(str)
    looks_like_header = any(not s.strip().isdigit() for s in first7)
    df = pd.read_csv(csv_path) if looks_like_header else pd.read_csv(csv_path, header=None)

    order = list(df.columns)
    num_cols, col_series = [], {}
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() >= 0.90:
            num_cols.append(c)
            col_series[c] = s

    if len(num_cols) < 5:
        raise ValueError("Could not find at least 5 numeric columns for main numbers.")

    num_cols_sorted = sorted(num_cols, key=lambda x: order.index(x))
    mains, stars = [], []
    for c in num_cols_sorted:
        mx = col_series[c].max()
        (mains if mx > 12 else stars).append(c)

    if len(mains) < 5:
        mains = sorted(num_cols, key=lambda c: col_series[c].max(), reverse=True)[:5]

    main_df = pd.concat([col_series[c] for c in mains[:5]], axis=1).dropna().astype(int)
    stars_df = (
        pd.concat([col_series[c] for c in stars[:2]], axis=1).dropna().astype(int)
        if len(stars) >= 2
        else None
    )
    main_df.columns = range(main_df.shape[1])
    if stars_df is not None:
        stars_df.columns = range(stars_df.shape[1])
    return main_df, stars_df


def trim_to_stars_12(main_df: pd.DataFrame, stars_df: pd.DataFrame):
    """Trim to first row where any star equals 12 -> ensures 12-star regime (2016+)."""
    arr = stars_df.to_numpy()
    mask12 = (arr == 12).any(axis=1)
    if mask12.any():
        start_idx = int(np.argmax(mask12))
        return (
            main_df.iloc[start_idx:].reset_index(drop=True),
            stars_df.iloc[start_idx:].reset_index(drop=True),
            start_idx,
        )
    return main_df.reset_index(drop=True), stars_df.reset_index(drop=True), 0


def ablate_df(df: pd.DataFrame, frac: float, mode: str) -> pd.DataFrame:
    """Remove a fraction of rows to stress-test. mode in {none, head, tail, middle, random}."""
    if frac <= 0 or mode == "none":
        return df.reset_index(drop=True)
    n = len(df)
    k = int(frac * n)
    if k <= 0:
        return df.reset_index(drop=True)
    if mode == "head":
        return df.iloc[k:].reset_index(drop=True)
    if mode == "tail":
        return df.iloc[: n - k].reset_index(drop=True)
    if mode == "middle":
        start = (n - k) // 2
        return pd.concat([df.iloc[:start], df.iloc[start + k :]], axis=0).reset_index(drop=True)
    if mode == "random":
        idx = np.arange(n)
        rs = np.random.RandomState(42)
        rs.shuffle(idx)
        keep = np.sort(idx[k:])
        return df.iloc[keep].reset_index(drop=True)
    return df.reset_index(drop=True)


# ----------------- Bias + Features (Agent) -----------------
@dataclass
class BiasConfig:
    windows: Tuple[int, ...] = (50, 200, 500)
    rec_cap: int = 1000
    decay_fast: float = 0.98
    decay_slow: float = 0.995


class FeatureEngine:
    def __init__(self, n_max: int, k: int, cfg: BiasConfig):
        self.N = n_max
        self.k = k
        self.cfg = cfg
        self.t = 0
        self.occ = np.zeros(self.N + 1, dtype=np.int64)
        self.last_seen = np.full(self.N + 1, 10**9, dtype=np.int64)
        self.ema_fast = np.zeros(self.N + 1, dtype=float)
        self.ema_slow = np.zeros(self.N + 1, dtype=float)
        self.cusum = np.zeros(self.N + 1, dtype=float)
        self.last_time = np.full(self.N + 1, -1, dtype=int)
        self.gap_sum = np.zeros(self.N + 1, dtype=float)
        self.gap_sqsum = np.zeros(self.N + 1, dtype=float)
        self.gap_cnt = np.zeros(self.N + 1, dtype=int)
        self.buffer = []
        self.is_prime = np.r_[False, is_prime_vec(self.N)]

    def _window_freq_z(self, w: int) -> np.ndarray:
        if self.t == 0:
            return np.zeros(self.N)
        w_buf = self.buffer[-w:] if len(self.buffer) >= w else self.buffer
        counts = np.zeros(self.N + 1, dtype=float)
        for draw in w_buf:
            counts[draw] += 1.0
        freq = counts[1:] / max(1, len(w_buf))
        mu, sd = freq.mean(), freq.std()
        return np.zeros_like(freq) if sd < 1e-12 else (freq - mu) / sd

    def _interarrival_z(self) -> np.ndarray:
        means = np.zeros(self.N + 1, dtype=float)
        mask = self.gap_cnt > 0
        means[mask] = self.gap_sum[mask] / np.maximum(1, self.gap_cnt[mask])
        vals = means[1:]
        mu, sd = vals.mean(), vals.std()
        return np.zeros_like(vals) if sd < 1e-12 else (vals - mu) / sd

    def snapshot_features(self) -> pd.DataFrame:
        N = self.N
        nums = np.arange(1, N + 1)
        denom = max(1, self.t)
        freq = self.occ[1:] / denom
        f_mu, f_sd = freq.mean(), freq.std()
        zfreq_all = (freq - f_mu) / f_sd if f_sd > 1e-12 else np.zeros_like(freq)
        rec = np.minimum(self.last_seen[1:], BiasConfig.rec_cap) / float(self.cfg.rec_cap)
        hot_fast = self.ema_fast[1:]
        hot_slow = self.ema_slow[1:]
        hot_fast_n = hot_fast / (np.mean(hot_fast) + 1e-12)
        hot_slow_n = hot_slow / (np.mean(hot_slow) + 1e-12)
        momentum = hot_fast_n - hot_slow_n
        zfreq_w = [self._window_freq_z(w) for w in self.cfg.windows]
        z_gap = self._interarrival_z()
        cusum = self.cusum[1:]
        is_prime = self.is_prime[1:].astype(float)
        parity = (nums % 2 == 0).astype(float)
        last_digit = nums % 10
        last_digit_oh = np.eye(10)[last_digit]
        digit_sum = np.array([sum(int(d) for d in str(x)) for x in nums], dtype=float)
        mod3 = (nums % 3).astype(float)
        mod5 = (nums % 5).astype(float)
        mod7 = (nums % 7).astype(float)
        base = {
            "num": nums,
            "zfreq_all": zfreq_all,
            "rec": rec,
            "hot_fast": hot_fast_n,
            "hot_slow": hot_slow_n,
            "momentum": momentum,
            "bias_cusum": cusum,
            "bias_gap_z": z_gap,
            "prime": is_prime,
            "parity": parity,
            "digit_sum": digit_sum,
            "mod3": mod3,
            "mod5": mod5,
            "mod7": mod7,
        }
        df = pd.DataFrame(base)
        for i, w in enumerate(self.cfg.windows):
            df[f"zfreq_w{w}"] = zfreq_w[i]
        for d in range(10):
            df[f"lastd_{d}"] = last_digit_oh[:, d]
        return df

    def consume_draw(self, draw_nums: np.ndarray) -> None:
        self.t += 1
        self.last_seen[1:] += 1
        self.last_seen[draw_nums] = 0
        self.occ[draw_nums] += 1
        self.ema_fast *= self.cfg.decay_fast
        self.ema_slow *= self.cfg.decay_slow
        self.ema_fast[draw_nums] += 1.0
        self.ema_slow[draw_nums] += 1.0
        expected = self.k / self.N
        self.cusum *= 0.999
        self.cusum[draw_nums] += 1.0 - expected
        not_drawn = np.setdiff1d(np.arange(1, self.N + 1), draw_nums, assume_unique=True)
        self.cusum[not_drawn] -= expected
        for n in draw_nums:
            if self.last_time[n] >= 0:
                gap = self.t - self.last_time[n]
                self.gap_sum[n] += gap
                self.gap_sqsum[n] += gap * gap
                self.gap_cnt[n] += 1
            self.last_time[n] = self.t
        self.buffer.append(draw_nums)


def build_dataset(draw_df: pd.DataFrame, k: int, bias_cfg: BiasConfig, min_warmup: int):
    main = draw_df.iloc[:, :k].copy().dropna().astype(int)
    N = int(main.to_numpy().max())
    T = len(main)
    if T <= min_warmup + 1:
        raise ValueError(
            f"Not enough rows for warmup. Have {T}, need > {min_warmup + 1}. "
            "Consider reducing --warmup or checking the input CSV."
        )
    eng = FeatureEngine(N, k, bias_cfg)
    for t in range(min_warmup):
        eng.consume_draw(main.iloc[t].to_numpy())
    X_rows, y_rows = [], []
    feat_names = None
    for t in range(min_warmup, T):
        feats = eng.snapshot_features()
        truth = main.iloc[t].to_numpy()
        y_t = np.isin(feats["num"].to_numpy(), truth).astype(int)
        cols = [c for c in feats.columns if c != "num"]
        feat_names = feat_names or cols
        X_rows.append(feats[cols].to_numpy())
        y_rows.append(y_t)
        eng.consume_draw(truth)
    X = np.vstack(X_rows)
    y = np.concatenate(y_rows)
    steps = T - min_warmup
    return X, y, steps, N, feat_names or []


def ts_block_splits(steps: int, N: int, n_splits: int):
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2.")
    fold = steps // n_splits
    splits = []
    for i in range(1, n_splits):
        tr_steps = i * fold
        va_steps = fold if i < n_splits - 1 else (steps - tr_steps)
        if va_steps <= 0:
            continue
        tr = np.arange(0, tr_steps * N)
        va = np.arange(tr_steps * N, (tr_steps + va_steps) * N)
        splits.append((tr, va))
    return splits


def select_and_fit(X_tr, y_tr, steps_tr, N, cs_grid, n_splits, debug=False):
    splits = ts_block_splits(steps_tr, N, n_splits)
    best_c, best_loss = None, np.inf
    for C in cs_grid:
        losses = []
        for tr, va in splits:
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "lr",
                        LogisticRegression(
                            penalty="l2",
                            C=C,
                            class_weight="balanced",
                            solver="lbfgs",
                            max_iter=2000,
                        ),
                    ),
                ]
            )
            pipe.fit(X_tr[tr], y_tr[tr])
            p = pipe.predict_proba(X_tr[va])[:, 1]
            p_clipped = np.clip(p, 1e-15, 1 - 1e-15)
            losses.append(log_loss(y_tr[va], p_clipped))
        m = float(np.mean(losses))
        if debug:
            print(f"[cv] C={C} mean logloss={m:.6f}")
        if m < best_loss:
            best_loss, best_c = m, C
    final = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    penalty="l2", C=best_c, class_weight="balanced", solver="lbfgs", max_iter=2000
                ),
            ),
        ]
    )
    final.fit(X_tr, y_tr)
    return final, float(best_c), float(best_loss)


def evaluate_blocks(p: np.ndarray, y: np.ndarray, N: int, top_k: int):
    steps = len(p) // N
    hits = np.zeros(steps, dtype=int)
    topk_probs = []
    for k in range(steps):
        a, b = k * N, (k + 1) * N
        probs_k = np.clip(p[a:b], 1e-15, 1 - 1e-15)
        truth_k = y[a:b]
        picks_idx = np.argsort(probs_k)[::-1][:top_k]
        truth_idx = np.where(truth_k == 1)[0]
        hits[k] = np.intersect1d(picks_idx, truth_idx).size
        topk_probs.append(probs_k[picks_idx])
    return hits, topk_probs


def cls_metrics(p: np.ndarray, y: np.ndarray):
    p_clipped = np.clip(p, 1e-15, 1 - 1e-15)
    return dict(
        logloss=float(log_loss(y, p_clipped)),
        roc_auc=float(roc_auc_score(y, p)),
        pr_auc=float(average_precision_score(y, p)),
    )


# ---------- Calibration + permutation null ----------
def plot_calibration_and_brier(p: np.ndarray, y: np.ndarray, N: int, outdir: Path):
    """Reliability curve + Brier score on TEST block."""
    steps = len(p) // N
    P = p.reshape(steps, N)
    Y = y.reshape(steps, N)
    pr = P.ravel()
    gt = Y.ravel().astype(float)
    brier = float(np.mean((pr - gt) ** 2))
    bins = np.linspace(0.0, 1.0, 11)
    which = np.digitize(pr, bins) - 1
    conf, acc, cnt = [], [], []
    for b in range(10):
        m = which == b
        if m.sum() == 0:
            continue
        conf.append(float(pr[m].mean()))
        acc.append(float(gt[m].mean()))
        cnt.append(int(m.sum()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot([0, 1], [0, 1], "--")
    plt.plot(conf, acc, marker="o")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Reliability (TEST)")
    plt.tight_layout()
    plt.savefig(outdir / "agent_calibration.png")
    plt.close()
    (outdir / "agent_calibration.json").write_text(
        json.dumps(
            {
                "brier": brier,
                "points": [{"conf": c, "acc": a, "count": n} for c, a, n in zip(conf, acc, cnt)],
            },
            indent=2,
        )
    )


def permutation_null_mean_hits(p: np.ndarray, y: np.ndarray, N: int, trials: int, outdir: Path):
    """Simulate null by permuting labels within each draw; record mean hits of top-5 policy."""
    if trials <= 0:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = len(p) // N
    P = p.reshape(steps, N)
    Y = y.reshape(steps, N)
    means = []
    rng = np.random.default_rng(123)
    picks = np.argsort(P, axis=1)[:, ::-1][:, :5]
    for t in range(trials):
        hits_list = []
        for s in range(steps):
            perm = rng.permutation(N)
            y_perm = Y[s, perm]
            truth_idx = np.where(y_perm == 1)[0]
            hit = np.intersect1d(picks[s], truth_idx).size
            hits_list.append(hit)
        means.append(float(np.mean(hits_list)))
    means = np.array(means, dtype=float)
    plt.figure()
    plt.hist(means, bins=20)
    plt.axvline(np.mean(means), linestyle="--")
    plt.xlabel("Mean hits (perm null)")
    plt.ylabel("Frequency")
    plt.title(f"Permutation null (n={trials})")
    plt.tight_layout()
    plt.savefig(outdir / "agent_perm_null_hist.png")
    plt.close()
    (outdir / "agent_perm_null.json").write_text(
        json.dumps(
            {"trials": trials, "mean": float(means.mean()), "std": float(means.std())}, indent=2
        )
    )


# --------------- Randomness tests ---------------
def chi_square_uniform(counts: np.ndarray):
    N = len(counts)
    total = counts.sum()
    expected = total / N
    stat = float(((counts - expected) ** 2 / (expected + 1e-12)).sum())
    dof = N - 1
    p = float(chi2.sf(stat, dof)) if SCIPY_OK else None
    return stat, p, dof


def runs_parity_chi(main_df: pd.DataFrame):
    evens_per_draw = (main_df % 2 == 0).sum(axis=1).to_numpy()
    obs = np.bincount(evens_per_draw, minlength=6)[:6]
    from math import comb

    total = obs.sum()
    exp = np.array([comb(5, k) * (0.5**5) * total for k in range(6)])
    stat = float(((obs - exp) ** 2 / (exp + 1e-12)).sum())
    dof = 5
    p = float(chi2.sf(stat, dof)) if SCIPY_OK else None
    return stat, p, dof


def last_digit_chi(all_nums: np.ndarray):
    last = all_nums % 10
    counts = np.bincount(last, minlength=10)
    return chi_square_uniform(counts)


# --------------- EV helpers ---------------
def poisson_binom_pmf(probs: np.ndarray):
    m = len(probs)
    pmf = np.zeros(m + 1)
    pmf[0] = 1.0
    for p in probs:
        pmf[1 : m + 1] = pmf[1 : m + 1] * (1 - p) + pmf[0:m] * p
        pmf[0] *= 1 - p
    pmf = np.clip(pmf, 0, 1)
    s = pmf.sum()
    if s > 0:
        pmf /= s
    return pmf


def load_prizes(path: Path):
    ext = path.suffix.lower()
    table: Dict[Tuple[int, int], float] = {}
    if ext == ".json":
        obj = json.loads(Path(path).read_text())
        if "prizes" in obj:
            obj = obj["prizes"]
        for k, v in obj.items():
            km, ks = map(int, k.split("_"))
            table[(km, ks)] = float(v)
    else:
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            table[(int(r["k_main"]), int(r["k_star"]))] = float(r["prize"])
    return table


def load_prizes_multi(path: Path):
    """Load per-draw prize list: {"steps":[{"prizes": {...}}, ...]}"""
    obj = json.loads(Path(path).read_text())
    steps = obj.get("steps") or obj.get("draws") or []
    out = []
    for e in steps:
        prizes = e.get("prizes") if isinstance(e, dict) else e
        if isinstance(prizes, dict):
            # normalize keys to (km, ks) tuples
            tbl = {}
            for k, v in prizes.items():
                try:
                    km, ks = map(int, str(k).split("_"))
                    tbl[(km, ks)] = float(v)
                except Exception:
                    continue
            out.append(tbl)
    return out


def ticket_ev(
    p_main: np.ndarray,
    p_star: Optional[np.ndarray],
    prize_table: Dict[Tuple[int, int], float],
    dep_corr: float = 0.95,
) -> float:
    p_m = np.clip(p_main * dep_corr, 0.0, 1.0)
    pmf_m = poisson_binom_pmf(p_m)
    if p_star is None or len(p_star) == 0:
        return float(sum(prize_table.get((km, 0), 0.0) * pmf_m[km] for km in range(len(pmf_m))))
    p_s = np.clip(p_star * dep_corr, 0.0, 1.0)
    pmf_s = poisson_binom_pmf(p_s)
    ev = 0.0
    for km, pm in enumerate(pmf_m):
        for ks, ps in enumerate(pmf_s):
            ev += pm * ps * prize_table.get((km, ks), 0.0)
    return float(ev)


# --------------- Discriminator ---------------
def euler_phi_upto(m: int) -> np.ndarray:
    phi = np.arange(m + 1)
    for i in range(2, m + 1):
        if phi[i] == i:
            phi[i::i] -= phi[i::i] // i
    return phi[1:]


def build_discriminator(main_df: pd.DataFrame, warmup: int, test_frac: float, debug: bool = False):
    from itertools import combinations

    main = main_df.copy().astype(int)
    N = int(main.to_numpy().max())
    T = len(main)
    if T <= warmup + 1:
        raise SystemExit("Not enough rows for warmup in discriminator.")
    M = np.zeros((N + 1, N + 1), dtype=np.int32)
    for t in range(warmup):
        draw = main.iloc[t].to_numpy()
        for a, b in combinations(sorted(draw), 2):
            M[a, b] += 1
            M[b, a] += 1
    poi = []
    centrality_rows = []
    steps = 0
    for t in range(warmup, T):
        steps += 1
        draw = main.iloc[t].to_numpy()
        s = 0
        for a, b in combinations(sorted(draw), 2):
            s += M[a, b]
        poi.append(s)
        cent = M.sum(axis=1)[1:]
        mu, sd = cent.mean(), cent.std()
        z = np.zeros_like(cent, dtype=np.float64) if sd < 1e-12 else (cent - mu) / sd
        centrality_rows.append(z)
        for a, b in combinations(sorted(draw), 2):
            M[a, b] += 1
            M[b, a] += 1
    centrality = np.vstack(centrality_rows)
    poi = np.array(poi, dtype=np.float64)
    test_steps = int(round(test_frac * steps))
    train_steps = steps - test_steps
    cent_test = centrality[train_steps:, :]
    if debug:
        print(
            f"[disc] steps={steps}, train={train_steps}, test={test_steps}, N={N}; cent_test={cent_test.shape}"
        )
    phi = euler_phi_upto(len(poi))
    return dict(
        N=N,
        steps=steps,
        train_steps=train_steps,
        test_steps=test_steps,
        centrality_test=cent_test,
        poi=poi,
        phi=phi,
    )


# --------------- Grok (tiny transformer) ---------------
class SeqDataset(Dataset):
    def __init__(self, X, y, seq_len=10):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.y) - self.seq_len)

    def __getitem__(self, i):
        x = self.X[i : i + self.seq_len]
        target = self.y[i + self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


class TinyTransformer(nn.Module):
    def __init__(self, d_model=8, nhead=2, num_layers=2, seq_len=10):
        super().__init__()
        self.inp = nn.Linear(2, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64, dropout=0.1, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model * seq_len, 1)

    def forward(self, x):
        h = self.inp(x)
        h = self.enc(h)
        h = h.reshape(h.shape[0], -1)
        return self.head(h).squeeze(-1)


def train_grok(
    phi,
    poi,
    steps,
    train_steps,
    test_steps,
    outdir: Path,
    seq_len=10,
    epochs=80,
    bs=16,
    lr=1e-3,
    debug=False,
    force=False,
):
    if not TORCH_OK:
        # still emit stub artifacts so downstream pipeline continues
        outdir.mkdir(parents=True, exist_ok=True)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for name in ["grok_lr.png", "grok_train_loss.png", "grok_val_loss.png"]:
            plt.figure()
            plt.plot([0], [0])
            plt.title("torch not installed - stub")
            plt.tight_layout()
            plt.savefig(outdir / name)
            plt.close()
        sig_full = np.zeros(test_steps, dtype=np.float64)
        pd.Series(sig_full).to_csv(
            outdir / "transformer_signal_test.csv", index=False, header=False
        )
        return sig_full, float("inf")

    # Build features: [phi_z, poi_prev_z]
    phi = np.asarray(phi, dtype=np.float64)
    poi = np.asarray(poi, dtype=np.float64)
    phi_z = (phi - phi.mean()) / (phi.std() + 1e-12)
    poi_prev = np.r_[0.0, poi[:-1]]
    poi_z = (poi_prev - poi_prev.mean()) / (poi_prev.std() + 1e-12)
    X = np.stack([phi_z, poi_z], axis=1)
    y = poi

    # Respect discriminator split
    X_tr, y_tr = X[:train_steps], y[:train_steps]
    X_te, y_te = X[train_steps:], y[train_steps:]

    # If there is not enough data, emit a safe constant signal and stub plots
    if (train_steps < 3 or test_steps < 1) and not force:
        outdir.mkdir(parents=True, exist_ok=True)
        sig_full = np.zeros(test_steps, dtype=np.float64)
        # minimal plots
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot([0, 1], [lr, lr])
        plt.xlabel("Update")
        plt.ylabel("Learning rate")
        plt.title("OneCycleLR schedule (stub)")
        plt.tight_layout()
        plt.savefig(outdir / "grok_lr.png")
        plt.close()
        plt.figure()
        plt.plot([0], [0])
        plt.title("Grok train loss (stub)")
        plt.tight_layout()
        plt.savefig(outdir / "grok_train_loss.png")
        plt.close()
        plt.figure()
        plt.plot([0], [0])
        plt.title("Grok val loss (stub)")
        plt.tight_layout()
        plt.savefig(outdir / "grok_val_loss.png")
        plt.close()
        (outdir / "grok_stats.json").write_text(
            json.dumps({"val_loss_min": None, "epochs": 0, "seq_len": None}, indent=2)
        )
        pd.Series(sig_full).to_csv(
            outdir / "transformer_signal_test.csv", index=False, header=False
        )
        if debug:
            print("[grok] too little data; emitted zero signal")
        return sig_full, float("inf")

    # Dynamic effective sequence length
    eff_seq_len = min(int(seq_len), max(2, min(max(2, train_steps - 1), max(2, test_steps - 1))))

    ds_tr = SeqDataset(X_tr, y_tr, seq_len=eff_seq_len)
    ds_te = SeqDataset(X_te, y_te, seq_len=eff_seq_len)
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True)
    dl_te = DataLoader(ds_te, batch_size=bs, shuffle=False)

    model = TinyTransformer(d_model=8, nhead=2, num_layers=2, seq_len=eff_seq_len)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # OneCycleLR for LR plot
    steps_per_epoch = max(1, len(dl_tr))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch
    )
    loss_fn = nn.MSELoss()
    hist = {"train_loss": [], "val_loss": [], "lr": []}
    best = 1e18
    for ep in range(epochs):
        model.train()
        run_loss = 0.0
        for xb, yb in dl_tr:
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            scheduler.step()
            hist["lr"].append(float(scheduler.get_last_lr()[0]))
            run_loss += float(loss.item())
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for xb, yb in dl_te:
                out = model(xb)
                vloss += float(loss_fn(out, yb).item())
        hist["train_loss"].append(run_loss / max(1, len(dl_tr)))
        hist["val_loss"].append(vloss / max(1, len(dl_te)))
        if debug:
            print(
                f"[grok] ep {ep + 1}/{epochs} train={hist['train_loss'][-1]:.6f} val={hist['val_loss'][-1]:.6f}"
            )
        best = min(best, hist["val_loss"][-1])

    # Emit a scalar per TEST step. If too short, fallback to a constant.
    outdir.mkdir(parents=True, exist_ok=True)
    if len(X_te) <= eff_seq_len:
        const = float(np.mean(y_tr[-min(5, len(y_tr)) :]))
        sig_full = np.full(test_steps, const, dtype=np.float64)
    else:
        sig = []
        model.eval()
        with torch.no_grad():
            for i in range(len(X_te) - eff_seq_len):
                xb = torch.tensor(X_te[i : i + eff_seq_len], dtype=torch.float32).unsqueeze(0)
                yhat = model(xb).item()
                sig.append(yhat)
        if len(sig) == 0:
            const = float(np.mean(y_tr[-min(5, len(y_tr)) :]))
            sig_full = np.full(test_steps, const, dtype=np.float64)
        else:
            pad = [sig[0]] * max(0, test_steps - len(sig))
            sig_full = np.array(pad + sig, dtype=np.float64)[:test_steps]

    # Plots
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(hist["lr"])
    plt.xlabel("Update")
    plt.ylabel("Learning rate")
    plt.title("OneCycleLR schedule")
    plt.tight_layout()
    plt.savefig(outdir / "grok_lr.png")
    plt.close()
    plt.figure()
    plt.plot(hist["train_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title("Grok train loss")
    plt.tight_layout()
    plt.savefig(outdir / "grok_train_loss.png")
    plt.close()
    plt.figure()
    plt.plot(hist["val_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Val loss")
    plt.title("Grok val loss")
    plt.tight_layout()
    plt.savefig(outdir / "grok_val_loss.png")
    plt.close()

    (outdir / "grok_stats.json").write_text(
        json.dumps({"val_loss_min": best, "epochs": epochs, "seq_len": eff_seq_len}, indent=2)
    )
    pd.Series(sig_full).to_csv(outdir / "transformer_signal_test.csv", index=False, header=False)
    # Optional next-step forecast beyond TEST
    try:
        last_seq_len = min(len(X), eff_seq_len)
        if last_seq_len >= 2:
            last_seq = torch.tensor(X[-last_seq_len:], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                y_next = model(last_seq).item()
            pd.Series([y_next]).to_csv(
                outdir / "transformer_signal_next.csv", index=False, header=False
            )
    except Exception:
        pass
    return sig_full, best


# --------------- RL Mixer ---------------
def zscore(a, axis=1, eps=1e-12):
    m = a.mean(axis=axis, keepdims=True)
    s = a.std(axis=axis, keepdims=True)
    s = np.where(s < eps, 1.0, s)
    return (a - m) / s


def softmax(x, axis=1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def eval_policy(
    p_agent,
    truth,
    p_pair,
    sig=None,
    w=(0.7, 0.3),
    tau=1.0,
    prize_table=None,
    prizes_multi=None,
    ticket_cost=2.5,
    ev_mult=1.5,
):
    steps, N = p_agent.shape
    pa = zscore(p_agent, axis=1)
    pp = zscore(p_pair, axis=1)
    scores = (w[0] * pa + w[1] * pp) / max(1e-6, tau)
    probs = softmax(scores, axis=1)
    picks_idx = np.argsort(probs, axis=1)[:, ::-1][:, :5]
    hits = []
    cost = 0.0
    ret = 0.0
    for t in range(steps):
        truth_idx = np.where(truth[t] == 1)[0]
        hit = np.intersect1d(picks_idx[t], truth_idx).size
        hits.append(hit)
        # Per-step realized reward using prizes_multi if provided; fallback to single table.
        table_t = None
        if prizes_multi is not None and len(prizes_multi) > 0:
            if len(prizes_multi) == steps:
                table_t = prizes_multi[t]
            else:
                # best-effort align from the end
                off = max(0, len(prizes_multi) - steps)
                idx = min(len(prizes_multi) - 1, off + t)
                table_t = prizes_multi[idx]
        elif prize_table is not None:
            table_t = prize_table
        if table_t is not None:
            local_ev_mult = ev_mult
            if sig is not None and len(sig) == steps:
                s = (sig[t] - np.mean(sig)) / (np.std(sig) + 1e-12)
                local_ev_mult *= float(1.0 + 0.05 * np.tanh(s))
            payout = table_t.get((hit, 0), 0.0)
            if payout >= local_ev_mult * ticket_cost:
                cost += ticket_cost
                ret += payout
    hits = np.array(hits, dtype=int)
    roi = (ret - cost) / cost if cost > 0 else float("nan")
    return hits, (cost, ret, roi), picks_idx


def cross_entropy_search(
    p_agent,
    truth,
    p_pair,
    sig=None,
    prize_table=None,
    prizes_multi=None,
    ticket_cost=2.5,
    ev_mult=1.5,
    iters=25,
    samples=60,
    init_mean=(0.7, 0.3, 1.0),
    init_std=(0.25, 0.25, 0.35),
    topk_frac=0.2,
    use_roi=False,
):
    mean = np.array(init_mean, dtype=np.float64)
    std = np.array(init_std, dtype=np.float64)
    for it in range(iters):
        cand = np.random.randn(samples, 3) * std + mean
        cand[:, :2] = np.clip(cand[:, :2], 0.0, 1.5)
        cand[:, 2] = np.clip(cand[:, 2], 0.2, 3.0)
        scores = []
        for w1, w2, tau in cand:
            hits, (cost, ret, roi), _ = eval_policy(
                p_agent,
                truth,
                p_pair,
                sig=sig,
                w=(w1, w2),
                tau=tau,
                prize_table=prize_table,
                prizes_multi=prizes_multi,
                ticket_cost=ticket_cost,
                ev_mult=ev_mult,
            )
            score = (
                roi if (use_roi and prize_table is not None and not np.isnan(roi)) else hits.mean()
            )
            scores.append(score)
        scores = np.array(scores)
        elite_idx = scores.argsort()[::-1][: max(1, int(topk_frac * len(cand)))]
        elite = cand[elite_idx]
        mean = elite.mean(axis=0)
        std = elite.std(axis=0) + 1e-6
    return tuple(mean.tolist())


# --------------- Orchestrator ---------------
@dataclass
class LabConfig:
    csv_path: Path
    outdir: Path
    warmup: int = 200
    test_frac: float = 0.20
    splits: int = 5
    with_stars: bool = False
    ticket_cost: float = 2.50
    ev_mult: float = 1.5
    dep_corr: float = 0.95
    prizes_path: Optional[Path] = None


def dump_block(arr: np.ndarray, N: int, path: Path) -> int:
    steps = len(arr) // N
    M = arr.reshape(steps, N)
    pd.DataFrame(M).to_csv(path, index=False, header=False)
    return steps


def run_agent(main_df: pd.DataFrame, cfg: LabConfig, debug=False) -> Dict:
    bias_cfg = BiasConfig()
    X, y, steps, N, feat_names = build_dataset(
        main_df, k=5, bias_cfg=bias_cfg, min_warmup=cfg.warmup
    )
    tr_steps = int((1.0 - cfg.test_frac) * steps)
    split_idx = tr_steps * N
    X_tr, y_tr, X_te, y_te = X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]

    if tr_steps < cfg.splits:
        raise ValueError(
            f"Not enough training steps ({tr_steps}) for {cfg.splits}-fold time-series CV. "
            f"Try reducing --warmup, --test-frac, or --splits."
        )
    model, C, cv_loss = select_and_fit(
        X_tr,
        y_tr,
        tr_steps,
        N,
        cs_grid=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0),
        n_splits=cfg.splits,
        debug=debug,
    )
    p_te = model.predict_proba(X_te)[:, 1]
    metrics = cls_metrics(p_te, y_te)
    hits, _ = evaluate_blocks(p_te, y_te, N, top_k=5)
    if debug:
        print(f"[agent] steps={steps} tr={tr_steps} te={steps - tr_steps} N={N}")
        print(f"[agent] C={C} cv_logloss={cv_loss:.6f}")
        print(f"[agent] test metrics: {metrics}")
        print(f"[agent] mean hits (top-5)={hits.mean():.4f} baseline={5 * (5 / N):.4f}")
    # calibration + permutation null
    outA = cfg.outdir / "agent_out"
    outA.mkdir(parents=True, exist_ok=True)
    plot_calibration_and_brier(p_te, y_te, N, outA)
    permutation_null_mean_hits(p_te, y_te, N, trials=getattr(cfg, "permute_null", 0), outdir=outA)
    # dump for mixer
    dump_block(p_te, N, outA / "probs_main_test.csv")
    dump_block(y_te, N, outA / "truth_main_test.csv")
    # Randomness tests summary
    diag = {}
    all_mains = main_df.to_numpy().ravel()
    N_guess = int(all_mains.max())
    counts = np.bincount(all_mains, minlength=N_guess + 1)[1:]
    stat1, p1, d1 = chi_square_uniform(counts)
    stat2, p2, d2 = last_digit_chi(all_mains)
    stat3, p3, d3 = runs_parity_chi(main_df)
    diag["uniform_chi2"] = {"stat": stat1, "p": p1, "dof": d1}
    diag["last_digit_chi2"] = {"stat": stat2, "p": p2, "dof": d2}
    diag["parity_chi2"] = {"stat": stat3, "p": p3, "dof": d3}
    (cfg.outdir / "agent_metrics.json").write_text(
        json.dumps(
            {"metrics": metrics, "mean_hits": float(hits.mean()), "N": N, "diag": diag}, indent=2
        )
    )
    return {"p_te": p_te, "y_te": y_te, "N": N, "tr_steps": tr_steps, "steps": steps}


def run_all(
    cfg: LabConfig,
    mode: str,
    debug=False,
    ablate_mode="none",
    ablate_frac=0.0,
    grok_seq_len=10,
    grok_epochs=80,
    mixer_use_roi=False,
    force_grok=False,
):
    # Load & ablate
    main_df, stars_df = smart_load_draws(str(cfg.csv_path))
    if cfg.with_stars and stars_df is not None:
        main_df, stars_df, start_idx = trim_to_stars_12(main_df, stars_df)
        if debug and start_idx > 0:
            print(f"[info] trimmed to 12-star regime @ row {start_idx}")
    main_df0 = main_df.copy()
    main_df = ablate_df(main_df0, ablate_frac, ablate_mode)
    if debug and ablate_mode != "none":
        print(
            f"[ablate] mode={ablate_mode} frac={ablate_frac} rows {len(main_df0)} -> {len(main_df)}"
        )

    # Agent
    agent = run_agent(main_df, cfg, debug=debug)
    if mode == "agent":
        return

    # Discriminator
    disc = build_discriminator(main_df, warmup=cfg.warmup, test_frac=cfg.test_frac, debug=debug)
    disc_dir = cfg.outdir / "disc_out"
    disc_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(disc["centrality_test"]).to_csv(
        disc_dir / "pair_centrality_test.csv", index=False, header=False
    )
    pd.Series(disc["poi"]).to_csv(disc_dir / "poi.csv", index=False, header=False)
    pd.Series(disc["phi"]).to_csv(disc_dir / "phi.csv", index=False, header=False)
    (disc_dir / "meta.json").write_text(
        json.dumps(
            {
                k: int(disc[k]) if isinstance(disc[k], (int, np.integer)) else disc[k]
                for k in ["N", "steps", "train_steps", "test_steps"]
            },
            indent=2,
        )
    )
    if mode == "agent+disc":
        return

    # Grok
    grok_dir = cfg.outdir / "grok_out"
    grok_dir.mkdir(parents=True, exist_ok=True)
    sig, best = train_grok(
        disc["phi"],
        disc["poi"],
        disc["steps"],
        disc["train_steps"],
        disc["test_steps"],
        outdir=grok_dir,
        seq_len=grok_seq_len,
        epochs=grok_epochs,
        debug=debug,
        force=force_grok,
    )
    if mode == "agent+disc+grok":
        return

    # Mixer
    agent_dir = cfg.outdir / "agent_out"
    p_agent = pd.read_csv(agent_dir / "probs_main_test.csv", header=None).to_numpy(np.float64)
    truth = pd.read_csv(agent_dir / "truth_main_test.csv", header=None).to_numpy(np.int32)
    p_pair = pd.read_csv(disc_dir / "pair_centrality_test.csv", header=None).to_numpy(np.float64)
    T = min(p_agent.shape[0], truth.shape[0], p_pair.shape[0])
    p_agent, truth, p_pair = p_agent[:T], truth[:T], p_pair[:T]
    sig_path = grok_dir / "transformer_signal_test.csv"
    sig = (
        pd.read_csv(sig_path, header=None).squeeze("columns").to_numpy(np.float64)[:T]
        if sig_path.exists()
        else None
    )
    prize_table = (
        load_prizes(cfg.prizes_path) if cfg.prizes_path and cfg.prizes_path.exists() else None
    )
    prizes_multi = None
    if (
        hasattr(cfg, "prizes_multi_path")
        and cfg.prizes_multi_path
        and Path(cfg.prizes_multi_path).exists()
    ):
        prizes_multi = load_prizes_multi(Path(cfg.prizes_multi_path))

    # Split TEST into VAL/HOLD
    val_T = int(0.6 * T)
    pA_val, tru_val, pP_val = p_agent[:val_T], truth[:val_T], p_pair[:val_T]
    pA_hold, tru_hold, pP_hold = p_agent[val_T:], truth[val_T:], p_pair[val_T:]
    sig_val = sig[:val_T] if sig is not None else None
    sig_hold = sig[val_T:] if sig is not None else None

    w1, w2, tau = cross_entropy_search(
        pA_val,
        tru_val,
        pP_val,
        sig=sig_val,
        prize_table=prize_table,
        prizes_multi=prizes_multi,
        ticket_cost=cfg.ticket_cost,
        ev_mult=cfg.ev_mult,
        use_roi=mixer_use_roi,
    )
    hits, (cost, ret, roi), picks_idx = eval_policy(
        pA_hold,
        tru_hold,
        pP_hold,
        sig=sig_hold,
        w=(w1, w2),
        tau=tau,
        prize_table=prize_table,
        prizes_multi=prizes_multi,
        ticket_cost=cfg.ticket_cost,
        ev_mult=cfg.ev_mult,
    )
    outM = cfg.outdir / "mixer_out"
    outM.mkdir(parents=True, exist_ok=True)
    (outM / "mixer_report.json").write_text(
        json.dumps(
            {
                "params": {"w1": w1, "w2": w2, "tau": tau},
                "holdout": {
                    "steps": int(pA_hold.shape[0]),
                    "mean_hits": float(hits.mean()),
                    "cost": float(cost),
                    "ret": float(ret),
                    "roi": float(roi),
                },
            },
            indent=2,
        )
    )
    pd.DataFrame(picks_idx + 1).to_csv(outM / "mixer_picks.csv", index=False, header=False)
    if debug:
        print(
            f"[mixer] w1={w1:.3f} w2={w2:.3f} tau={tau:.3f} holdout mean_hits={hits.mean():.4f} roi={roi}"
        )


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(
        description="Lotto Lab: Agent + Discriminator + Grok + RL Mixer (one-file)"
    )
    ap.add_argument("--csv", required=True, help="Path to draws CSV")
    ap.add_argument("--outdir", default="lotto_lab_out", help="Output directory")
    ap.add_argument(
        "--mode",
        default="all",
        choices=["agent", "agent+disc", "agent+disc+grok", "all"],
        help="What to run",
    )
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--test-frac", type=float, default=0.20)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--with-stars", action="store_true")
    ap.add_argument("--ticket-cost", type=float, default=2.50)
    ap.add_argument("--ev-mult", type=float, default=1.5)
    ap.add_argument("--dep-corr", type=float, default=0.95)
    ap.add_argument(
        "--prizes", type=str, help="Single prize table JSON/CSV (keys like 5_0, 4_0, ...)"
    )
    ap.add_argument(
        "--prizes-multi",
        type=str,
        help="Per-draw prizes JSON with structure {'steps': [{'prizes': {...}}, ...]}",
    )
    ap.add_argument("--debug", action="store_true", help="Verbose debug prints")
    # Ablations
    ap.add_argument(
        "--ablate-mode",
        default="none",
        choices=["none", "head", "tail", "middle", "random"],
        help="Remove part of the dataset to test robustness",
    )
    ap.add_argument("--ablate-frac", type=float, default=0.0, help="Fraction to remove (0..0.8)")
    # Grok
    ap.add_argument("--grok-seq-len", type=int, default=10)
    ap.add_argument("--grok-epochs", type=int, default=80)
    ap.add_argument(
        "--force-grok",
        action="store_true",
        help="Always emit grok artifacts even with tiny data (stub/constant).",
    )
    # Mixer
    ap.add_argument(
        "--use-roi",
        action="store_true",
        help="Mixer optimizes holdout ROI (needs --prizes). Default=mean hits.",
    )
    # Stats
    ap.add_argument(
        "--permute-null",
        type=int,
        default=0,
        help="Permutation-null runs for mean-hits sanity (0=off).",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = LabConfig(
        csv_path=Path(args.csv),
        outdir=outdir,
        warmup=args.warmup,
        test_frac=args.test_frac,
        splits=args.splits,
        with_stars=bool(args.with_stars),
        ticket_cost=args.ticket_cost,
        ev_mult=args.ev_mult,
        dep_corr=args.dep_corr,
        prizes_path=Path(args.prizes) if args.prizes else None,
    )
    setattr(cfg, "permute_null", args.permute_null)

    if args.ablate_frac < 0 or args.ablate_frac > 0.8:
        print("[warn] ablate-frac should be in [0, 0.8]. Clipping.")
        args.ablate_frac = max(0.0, min(0.8, args.ablate_frac))

    run_all(
        cfg,
        mode=args.mode,
        debug=args.debug,
        ablate_mode=args.ablate_mode,
        ablate_frac=args.ablate_frac,
        grok_seq_len=args.grok_seq_len,
        grok_epochs=args.grok_epochs,
        mixer_use_roi=args.use_roi,
        force_grok=args.force_grok,
    )


if __name__ == "__main__":
    main()
