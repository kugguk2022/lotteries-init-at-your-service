"""Fixed-size, forward-only benchmark of the owner's main-pair PoI method.

This preserves the R main-pair statistic, not the cross-pool GARCH statistic.
The target is the existing PoI trailing-26 mean under the training-cutoff matrix.
Equal-score tickets use a versioned seeded hash order, never auxiliary-number order.
Historical replays are not public pre-draw predictions and are never cash ROI.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from itertools import chain, combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .likely_set_generator import (
    GameConfig,
    build_comatrices,
    detect_columns,
    observed_poi_series,
)
from .protocol import GameSpec

VERSION = "poi-main-fixed-budget-v1"
BUDGETS = (12, 100, 1000, 1_000_000)
WINDOW = 26
HOLDOUT = 52
SEED = 20261109
FILES = (
    "poi_benchmark_replay.csv", "poi_benchmark_summary.json",
    "poi_next_preview.csv", "poi_next_candidates.csv.gz", "poi_next_manifest.json",
)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=1)
def main_combinations(n: int, k: int) -> np.ndarray:
    array = np.fromiter(chain.from_iterable(combinations(range(n), k)), dtype=np.uint16,
                        count=math.comb(n, k) * k).reshape(-1, k)
    array.flags.writeable = False
    return array


def tie_keys(ids: np.ndarray, seed: int) -> np.ndarray:
    """SplitMix64 bijection: deterministic, unique keys for distinct ticket IDs."""
    with np.errstate(over="ignore"):
        keys = ids.astype(np.uint64) + np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
        keys = (keys ^ (keys >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        keys = (keys ^ (keys >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return keys ^ (keys >> np.uint64(31))


def normalized_history(history: pd.DataFrame, spec: GameSpec) -> pd.DataFrame:
    cfg = GameConfig(spec.name, spec.main_n, spec.main_k, spec.star_n, spec.star_k)
    mains, stars = detect_columns(history, cfg)
    if len(mains) != spec.main_k or len(stars) != spec.star_k:
        raise ValueError("Missing required draw columns")
    frame = history[["draw_date", *mains, *stars]].copy()
    frame["draw_date"] = pd.to_datetime(frame.draw_date, errors="raise").dt.strftime("%Y-%m-%d")
    if frame.draw_date.duplicated().any():
        raise ValueError("Duplicate draw dates")
    for col in [*mains, *stars]:
        numeric = pd.to_numeric(frame[col], errors="raise")
        if not np.isfinite(numeric).all() or not (numeric == np.floor(numeric)).all():
            raise ValueError("Noninteger or missing draw numbers")
        frame[col] = numeric.astype(int)
    for row in frame.to_dict("records"):
        spec.validate_ticket((tuple(sorted(row[c] for c in mains)),
                              tuple(sorted(row[c] for c in stars))))
    frame = frame.sort_values("draw_date").reset_index(drop=True)
    # Fixed 12-star rule era only; older draws have a different legal universe.
    if spec.name == "euromillions":
        frame = frame[frame.draw_date >= "2016-09-27"].reset_index(drop=True)
    return frame


@dataclass
class RankedPoi:
    ids: np.ndarray
    main_scores: np.ndarray
    target: float
    spec: GameSpec
    seed: int

    def decode(self, ids: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        ids = self.ids if ids is None else ids
        aux = auxiliary_combinations(self.spec)
        mains = main_combinations(self.spec.main_n, self.spec.main_k)[ids // len(aux)] + 1
        return mains, aux[ids % len(aux)]


def auxiliary_combinations(spec: GameSpec) -> np.ndarray:
    if not spec.star_k:
        return np.zeros((1, 0), dtype=np.int64)
    return np.array(list(combinations(range(1, spec.star_n + 1), spec.star_k)), dtype=np.int64)


def rank_poi(history: pd.DataFrame, spec: GameSpec, size: int, *, seed: int = SEED,
             window: int = WINDOW) -> RankedPoi:
    """Exact full-universe ranking, bounded output; receives training rows only.

    Primary key: |G-target|; secondary: descending G, as in existing PoI.
    Final key: seeded hash of the full-ticket ID (including auxiliary numbers).
    Smaller budgets are exact prefixes, regardless of requested output size.
    """
    if size < 1 or size > spec.n_tickets() or window < 1 or len(history) < window:
        raise ValueError("Invalid size/window or insufficient training history")
    cfg = GameConfig(spec.name, spec.main_n, spec.main_k, spec.star_n, spec.star_k)
    mc, sc = detect_columns(history, cfg)
    matrices = build_comatrices(history, cfg, mc, sc)
    series = observed_poi_series(history.iloc[-window:], cfg, matrices, mc, sc, "main")
    target = float(np.mean(series))
    mains = main_combinations(spec.main_n, spec.main_k)
    scores = np.zeros(len(mains), dtype=np.int64)
    for a, b in combinations(range(spec.main_k), 2):
        scores += matrices.Wmain[mains[:, a], mains[:, b]]
    levels = sorted(np.unique(scores), key=lambda g: (abs(float(g) - target), -int(g)))
    aux_count = len(auxiliary_combinations(spec))
    selected = []
    remaining = size
    for level in levels:
        main_ids = np.flatnonzero(scores == level)
        full_ids = (main_ids[:, None] * aux_count + np.arange(aux_count)).ravel()
        keys = tie_keys(full_ids, seed)
        take = min(remaining, len(full_ids))
        chosen = np.argpartition(keys, take - 1)[:take]
        chosen = chosen[np.argsort(keys[chosen])]
        selected.append(full_ids[chosen])
        remaining -= take
        if not remaining:
            break
    return RankedPoi(np.concatenate(selected), scores, target, spec, seed)


def favourable_tickets(spec: GameSpec, threshold: int | None) -> int:
    """Number of legal full tickets with >= threshold mains, or the exact winner."""
    if threshold is None:
        return 1
    return sum(math.comb(spec.main_k, h) * math.comb(spec.main_n - spec.main_k, spec.main_k-h)
               for h in range(threshold, spec.main_k + 1)) * len(auxiliary_combinations(spec))


@lru_cache(maxsize=128)
def null_any(universe: int, favourable: int, budget: int) -> float:
    """Exact hypergeometric P(at least one): unique tickets, no replacement.

    Use the shorter product over the sample or favourable set for stable small-tail arithmetic.
    """
    if min(universe, budget) < 1 or not 0 <= favourable <= universe or budget > universe:
        raise ValueError("Invalid null parameters")
    if favourable == 0:
        return 0.0
    if budget > universe - favourable:
        return 1.0
    a, b = sorted((budget, favourable))
    logs = 0.0
    for start in range(0, a, 100_000):
        offsets = np.arange(start, min(start + 100_000, a), dtype=float)
        logs += float(np.log1p(-b / (universe - offsets)).sum())
    return float(-np.expm1(logs))


def wilson(hits: int, draws: int) -> list[float]:
    z = 1.959963984540054
    p = hits / draws
    denominator = 1 + z*z/draws
    center = (p + z*z/(2*draws)) / denominator
    radius = z * math.sqrt(p*(1-p)/draws + z*z/(4*draws*draws)) / denominator
    return [max(0.0, center-radius), min(1.0, center+radius)]


def evaluate_draw(ranking: RankedPoi, outcome: pd.Series, budgets: tuple[int, ...],
                  training_hash: str, target_date: str) -> list[dict]:
    """Score only after candidate generation; count mains and stars on the same ticket."""
    spec = ranking.spec
    cfg = GameConfig(spec.name, spec.main_n, spec.main_k, spec.star_n, spec.star_k)
    mc, sc = detect_columns(pd.DataFrame([outcome]), cfg)
    actual_main, actual_aux = [int(outcome[c]) for c in mc], [int(outcome[c]) for c in sc]
    mains, aux = ranking.decode()
    mh = np.isin(mains, actual_main).sum(axis=1)
    ah = np.isin(aux, actual_aux).sum(axis=1)
    records = []
    for budget in budgets:
        ids = ranking.ids[:budget]
        tiers = np.bincount(mh[:budget]*(spec.star_k+1)+ah[:budget],
                            minlength=(spec.main_k+1)*(spec.star_k+1))
        common = {"draw_date": target_date, "budget": budget, "target_g": ranking.target,
                  "history_sha256": training_hash, "seed": ranking.seed,
                  "candidate_ids_sha256": hashlib.sha256(ids.astype("<u8").tobytes()).hexdigest(),
                  "tier_counts": json.dumps({f"{h}+{s}": int(tiers[h*(spec.star_k+1)+s])
                                             for h in range(spec.main_k+1)
                                             for s in range(spec.star_k+1)}, sort_keys=True)}
        for metric, threshold in [("full_winner", None), ("main_ge3", 3),
                                  ("main_ge4", 4), ("all_main", spec.main_k)]:
            count = int(tiers[-1]) if threshold is None else int((mh[:budget] >= threshold).sum())
            favourable = favourable_tickets(spec, threshold)
            records.append({**common, "metric": metric, "matching_tickets": count,
                            "draw_hit": int(count > 0),
                            "random_expected_tickets": budget*favourable/spec.n_tickets(),
                            "random_draw_hit_probability": null_any(spec.n_tickets(), favourable, budget)})
    return records


def aggregate_replay(frame: pd.DataFrame, spec: GameSpec) -> list[dict]:
    rows = []
    for (budget, metric), group in frame.groupby(["budget", "metric"], sort=True):
        draws = len(group)
        hits = int(group.draw_hit.sum())
        expected = float(group.random_draw_hit_probability.sum())
        interval = wilson(hits, draws)
        rows.append({"budget": int(budget), "metric": metric, "draws": draws,
                     "universe_fraction": int(budget)/spec.n_tickets(), "hits": hits,
                     "random_expected_hits": expected,
                     "lift": hits/expected if expected else None,
                     "hit_rate_ci95": interval,
                     "lift_ci95": [v*draws/expected for v in interval] if expected else None,
                     "matching_tickets": int(group.matching_tickets.sum()),
                     "random_expected_tickets": float(group.random_expected_tickets.sum())})
    return rows


def validate_poi_benchmark(directory: str | Path) -> dict:
    """Fail closed on missing/corrupt artifacts, illegal/duplicate tickets or budget drift."""
    root = Path(directory)
    summary = json.loads((root / FILES[1]).read_text())
    if set(summary["artifacts"]) != {FILES[0], FILES[2], FILES[3], FILES[4]}:
        raise ValueError("Incomplete PoI artifact manifest")
    for name, expected in summary["artifacts"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"PoI artifact hash mismatch: {name}")
    manifest = json.loads((root / FILES[4]).read_text())
    if digest({k: v for k, v in manifest.items() if k != "commitment_sha256"}) != manifest["commitment_sha256"]:
        raise ValueError("Invalid PoI commitment")
    if (manifest["version"] != summary["version"] or manifest["budgets"] != summary["budgets"]
            or manifest["history_cutoff"] != summary["history_cutoff"]):
        raise ValueError("PoI manifest and replay cohort differ")
    spec = GameSpec(**manifest["game"])
    expected_size = max(manifest["budgets"])
    if manifest["candidate_size"] != expected_size or manifest["universe_size"] != spec.n_tickets():
        raise ValueError("PoI candidate size differs from budget")
    codes = []
    count = 0
    for chunk in pd.read_csv(root / FILES[3], chunksize=50_000):
        if not np.array_equal(chunk["rank"], np.arange(count+1, count+len(chunk)+1)):
            raise ValueError("PoI ranks must be contiguous")
        code = np.zeros(len(chunk), dtype=np.uint64)
        for prefix, k, n in (("ball", spec.main_k, spec.main_n), ("star", spec.star_k, spec.star_n)):
            if not k:
                continue
            values = chunk[[f"{prefix}_{i+1}" for i in range(k)]].to_numpy()
            if (not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all()
                    or (values < 1).any() or (values > n).any() or (np.diff(values, axis=1) <= 0).any()):
                raise ValueError("Illegal PoI ticket")
            for column in values.T:
                code = code*np.uint64(n+1) + column.astype(np.uint64)
        codes.append(code)
        count += len(chunk)
    if count != expected_size or len(np.unique(np.concatenate(codes))) != count:
        raise ValueError("PoI ticket count or uniqueness failed")
    preview = pd.read_csv(root / FILES[2])
    expected = pd.read_csv(root / FILES[3], nrows=min(1000, count))
    if not preview.equals(expected):
        raise ValueError("PoI preview is not an exact ranked prefix")
    return summary


def build_poi_benchmark(history: pd.DataFrame, spec: GameSpec, *, output_directory: str | Path,
                        target_draw_date: str, snapshot_sha256: str, holdout: int = HOLDOUT,
                        budgets: tuple[int, ...] = BUDGETS, window: int = WINDOW,
                        seed: int = SEED, created_utc: str | None = None) -> dict:
    """Write auditable replay, exact next set, preview and hashes. No payout imputation."""
    history = normalized_history(history, spec)
    budgets = tuple(sorted(set(budgets)))
    if not budgets or budgets[0] < 1 or budgets[-1] > spec.n_tickets() or holdout < 1:
        raise ValueError("Invalid budgets or holdout")
    if len(history) <= window:
        raise ValueError("Not enough history for a held-out evaluation")
    target_draw_date = pd.Timestamp(target_draw_date).date().isoformat()
    if target_draw_date <= str(history.draw_date.iloc[-1]):
        raise ValueError("Target must follow all training data")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    source_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                     for name in ("poi_benchmark.py", "likely_set_generator.py", "protocol.py")}
    config = {"version": VERSION, "source_code_sha256": source_hashes,
              "game": asdict(spec), "budgets": list(budgets),
              "window": window, "seed": seed, "pairing": "main",
              "target_rule": "trailing mean rescored under training-cutoff matrix",
              "tie_rule": "distance ascending, G descending, SplitMix64(full_ticket_id+seed)"}
    config_hash = digest(config)
    replay_path = output / FILES[0]
    cached = pd.read_csv(replay_path, float_precision="round_trip") if replay_path.exists() else pd.DataFrame()
    records = []
    start = max(window, len(history)-holdout)
    for index in range(start, len(history)):
        train = history.iloc[:index]
        draw_date = str(history.draw_date.iloc[index])
        train_hash = digest(train.to_dict("records"))
        # Reuse only complete, byte-verified rows with matching outcome/config/training.
        outcome_hash = digest(history.iloc[index].to_dict())
        key = digest([config_hash, train_hash, draw_date, outcome_hash])
        prior = cached[cached.cache_key == key] if "cache_key" in cached else pd.DataFrame()
        expected_pairs = {(b, m) for b in budgets for m in
                          ("full_winner", "main_ge3", "main_ge4", "all_main")}
        if (len(prior) == len(expected_pairs)
                and set(zip(prior.budget, prior.metric)) == expected_pairs
                and all(digest({k: v for k, v in r.items() if k != "row_sha256"}) == r["row_sha256"]
                        for r in prior.to_dict("records"))):
            records.extend(prior.to_dict("records"))
            continue
        draw_seed = int(digest([seed, draw_date])[:16], 16)
        ranking = rank_poi(train, spec, budgets[-1], seed=draw_seed, window=window)
        rows = evaluate_draw(ranking, history.iloc[index], budgets, train_hash, draw_date)
        for row in rows:
            row["cache_key"] = key
            row["row_sha256"] = digest(row)
        records.extend(rows)
        print(f"PoI {spec.name}: {index-start+1}/{len(history)-start} {draw_date}", flush=True)
    replay = pd.DataFrame(records)
    replay.to_csv(replay_path, index=False, lineterminator="\n")
    next_seed = int(digest([seed, target_draw_date])[:16], 16)
    ranking = rank_poi(history, spec, budgets[-1], seed=next_seed, window=window)
    columns = ["rank", *[f"ball_{i+1}" for i in range(spec.main_k)],
               *[f"star_{i+1}" for i in range(spec.star_k)], "g_score"]
    aux_count = len(auxiliary_combinations(spec))
    preview = None
    # Fixed gzip mtime makes download hashes reproducible across identical rebuilds.
    with (output / FILES[3]).open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
        zipped.write((",".join(columns)+"\n").encode())
        for offset in range(0, len(ranking.ids), 50_000):
            ids = ranking.ids[offset:offset+50_000]
            mains, aux = ranking.decode(ids)
            values = np.column_stack((np.arange(offset+1, offset+1+len(ids)), mains, aux,
                                      ranking.main_scores[ids//aux_count]))
            chunk = pd.DataFrame(values, columns=columns)
            zipped.write(chunk.to_csv(index=False, header=False, lineterminator="\n").encode())
            if preview is None:
                preview = chunk.iloc[:min(1000, len(chunk))]
    preview.to_csv(output / FILES[2], index=False, lineterminator="\n")
    created = created_utc or datetime.now(timezone.utc).isoformat()
    stamp = pd.Timestamp(created)
    if stamp.tzinfo is None:
        raise ValueError("created_utc must include timezone")
    # Conservative cutoff: never claim a same-day or past reconstruction as pre-draw.
    timely = stamp.tz_convert("UTC") < pd.Timestamp(target_draw_date, tz="UTC")
    manifest = {**config, "config_sha256": config_hash, "history_sha256": digest(history.to_dict("records")),
                "source_snapshot_sha256": snapshot_sha256, "history_rows": len(history),
                "history_cutoff": str(history.draw_date.iloc[-1]), "target_draw_date": target_draw_date,
                "created_utc": created, "target_g": ranking.target, "draw_seed": next_seed,
                "candidate_size": len(ranking.ids), "universe_size": spec.n_tickets(),
                "publication_status": "AWAITING_PUBLIC_TIMESTAMP_VERIFICATION" if timely else "LATE_RECONSTRUCTION",
                "realized_roi": None, "roi_status": "UNASSESSED_NO_VERIFIED_ALL_PRIZE_PAYOUTS",
                "artifacts": {name: hashlib.sha256((output/name).read_bytes()).hexdigest()
                              for name in (FILES[2], FILES[3])}}
    manifest["commitment_sha256"] = digest(manifest)
    (output / FILES[4]).write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    summary = {**config, "evidence_kind": "historical_forward_replay_not_predraw_publication",
               "source_snapshot_sha256": snapshot_sha256, "history_cutoff": str(history.draw_date.iloc[-1]),
               "history_rows": len(history), "first_evaluated_draw": str(replay.draw_date.min()),
               "last_evaluated_draw": str(replay.draw_date.max()), "holdout": len(history)-start,
               "results": aggregate_replay(replay, spec), "realized_roi": None,
               "roi_status": manifest["roi_status"], "predictive_edge_status": "UNESTABLISHED",
               "uncertainty": "95% Wilson intervals across draws; descriptive, unadjusted for multiple comparisons",
               "null": "Exact full-universe uniformly sampled unique sets at the identical ticket count",
               "main_match_boundary": "Main-only match counts are diagnostics, not prize tiers or jackpot wins",
               "next_manifest": FILES[4],
               "artifacts": {name: hashlib.sha256((output/name).read_bytes()).hexdigest()
                             for name in (FILES[0], FILES[2], FILES[3], FILES[4])}}
    (output / FILES[1]).write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    validate_poi_benchmark(output)
    return summary
