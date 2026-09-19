"""Exact full-ticket containment in frozen, inclusive forecast score ranges.

Range coverage and cash ROI have different denominators. Every legal ticket in a
range is counted, including every tie; there is no million-ticket truncation.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from itertools import combinations
from math import isfinite, prod, sqrt
from pathlib import Path
from statistics import NormalDist
from urllib.request import urlopen

import pandas as pd

from .pair_raster import _history_matrices
from .protocol import GameSpec

METHOD = "garch_normal_ranges_v1"
PENDING_FILE = "forecast_range_pending.json"
REPLAY_FILE = "forecast_range_replay.csv"
LEDGER_FILE = "forecast_range_ledger.json"
SUMMARY_FILE = "forecast_range_summary.json"
SPACE = "https://huggingface.co/spaces/kugguk/lottobench-community-leaderboard"
API = "https://huggingface.co/api/spaces/kugguk/lottobench-community-leaderboard"


def digest(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def forecast_ranges(first: float, second: float, diagnostics: dict) -> list[dict]:
    """Declared normal-approximation bands, not an assertion of calibrated coverage.

The hybrid uses the UNION of the two bands with the GARCH scale. Its nominal
level describes each component band, never the probability of their union.
"""
    variance = diagnostics.get("variance_next")
    if variance is None:
        return []
    if not all(isfinite(float(v)) for v in (first, second, variance)) or variance < 0:
        raise ValueError("forecast range parameters must be finite with nonnegative variance")
    ranges = []
    for level in (0.80, 0.95):
        width = NormalDist().inv_cdf((1 + level) / 2) * sqrt(variance)
        for model, centers in (("garch", [first]), ("hybrid_union", [first, second])):
            ranges.append({
                "range_id": f"{model}_{int(level * 100)}",
                "model": model,
                "nominal_component_level": level,
                "intervals": [[center - width, center + width] for center in centers],
            })
    return ranges


def contains(intervals: list, score: int) -> bool:
    return any(lower <= score <= upper for lower, upper in intervals)


def measure_ranges(ranges: list[dict], histogram: dict, universe: int,
                   ticket_price: float, actual_score: int | None = None) -> list[dict]:
    if sum(histogram.values()) != universe:
        raise ValueError("range histogram must count the complete legal ticket universe")
    if not isfinite(ticket_price) or ticket_price <= 0:
        raise ValueError("ticket price must be finite and positive")
    rows = []
    for band in ranges:
        count = sum(int(n) for score, n in histogram.items()
                    if contains(band["intervals"], int(score)))
        rows.append({
            **band,
            "candidate_count": count,
            "universe_size": universe,
            "fair_containment_probability": count / universe,
            "full_purchase_stake_eur": count * ticket_price,
            "actual_g_score": actual_score,
            "full_winner_in_range": (contains(band["intervals"], actual_score)
                                     if actual_score is not None else None),
            "cash_roi": None,
            "cash_roi_status": "UNASSESSED_NO_SETTLED_ALL_PRIZE_PAYOUTS",
        })
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    """Expected random hits use the actual set size on each eligible draw."""
    results = []
    for range_id in sorted({row["range_id"] for row in rows}):
        group = [row for row in rows if row["range_id"] == range_id]
        hits = sum(bool(row["full_winner_in_range"]) for row in group)
        expected = sum(row["fair_containment_probability"] for row in group)
        second = [row for row in group if row.get("second_or_better_hit") is not None]
        results.append({
            "range_id": range_id, "draws": len(group), "full_winner_hits": hits,
            "hit_rate": hits / len(group), "fair_expected_hits": expected,
            "mean_fair_containment_probability": expected / len(group),
            "containment_lift": hits / expected if expected else None,
            "excess_hits_over_fair": hits - expected,
            "total_full_purchase_stake_eur": sum(row["full_purchase_stake_eur"] for row in group),
            "cash_roi": None,
            "second_or_better_draws": len(second),
            "second_or_better_hits": sum(row["second_or_better_hit"] for row in second),
            "fair_second_or_better_expected_hits": sum(row["fair_second_or_better_probability"] for row in second),
        })
    return results


def score_ticket(matrices, main: tuple, stars: tuple) -> int:
    mains, auxiliary, cross = matrices
    return int(sum(mains[a][b] for a, b in combinations(main, 2))
               + sum(auxiliary[a][b] for a, b in combinations(stars, 2))
               + sum(cross[a][b] for a in main for b in stars))


def prize_measures(rows: list[dict], spec: GameSpec, actual: tuple, matrices,
                   reserve=None) -> list[dict]:
    """Enumerate all second-tier winners, never infer them from separate ball hits."""
    main, stars = actual
    winners = None
    if (spec.main_n, spec.main_k, spec.star_n, spec.star_k) == (50, 5, 12, 2):
        winners = [(main, tuple(sorted((hit, other)))) for hit in stars
                   for other in range(1, 13) if other not in stars]
    elif (spec.main_n, spec.main_k, spec.star_k) == (45, 6, 0):
        if reserve is not None and not pd.isna(reserve):
            if int(reserve) != reserve or not 1 <= int(reserve) <= 45 or int(reserve) in main:
                raise ValueError("invalid official NL Lotto reserve number")
            winners = [(tuple(sorted((*five, int(reserve)))), ())
                       for five in combinations(main, 5)]
    for row in rows:
        row.update({"second_prize_winning_tickets_in_range": None,
                    "second_or_better_hit": None, "fair_second_or_better_probability": None,
                    "second_prize_status": "UNAVAILABLE_GAME_RULES_OR_RESERVE_NUMBER"})
        if winners is None:
            continue
        second_count = sum(contains(row["intervals"], score_ticket(matrices, *ticket))
                           for ticket in winners)
        n, k, w = spec.n_tickets(), row["candidate_count"], len(winners) + 1
        probability = 1.0 if n - k < w else 1.0 - prod((n - k - i) / (n - i) for i in range(w))
        row.update({"second_prize_winning_tickets_in_range": second_count,
                    "second_or_better_hit": bool(row["full_winner_in_range"] or second_count),
                    "fair_second_or_better_probability": probability,
                    "second_prize_status": "EXACT_MATCH_CLASS_CONTAINMENT"})
    return rows


def pending_record(history: pd.DataFrame, spec: GameSpec, measures: list[dict], *,
                   history_cutoff: str, target_draw_date: str, snapshot_sha256: str) -> dict:
    dates = pd.to_datetime(history["draw_date"], errors="raise")
    if (dates.duplicated().any() or not dates.is_monotonic_increasing
            or str(dates.max().date()) != history_cutoff
            or pd.Timestamp(history_cutoff) >= pd.Timestamp(target_draw_date)):
        raise ValueError("range history must be ordered, unique and strictly before the target")
    record = {
        "method_version": METHOD,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "history_cutoff": history_cutoff, "target_draw_date": target_draw_date,
        "snapshot_sha256": snapshot_sha256,
        "game": {"name": spec.name, "main_n": spec.main_n, "main_k": spec.main_k,
                 "star_n": spec.star_n, "star_k": spec.star_k},
        "score_definition": "typed_pair_counts_frozen_at_history_cutoff",
        "score_matrices": [matrix.tolist() for matrix in _history_matrices(history, spec)],
        "ranges": measures,
    }
    return {"commitment_sha256": digest(record), "forecast": record}


def verify_commitment(pending: dict) -> None:
    if digest(pending["forecast"]) != pending["commitment_sha256"]:
        raise ValueError("forecast range commitment does not match its frozen contents")


def _read_url(url: str) -> bytes:
    with urlopen(url, timeout=15) as response:
        return response.read()


def publication_receipt(pending: dict, key: str, *, read_url=None) -> dict:
    """Prove the exact forecast existed in a dated public commit, before settlement."""
    if key not in ("euromillions", "nl-lotto"):
        raise ValueError("public range receipts require an observed profile")
    verify_commitment(pending)
    read_url = read_url or _read_url
    path = f"data/profiles/{key}/{PENDING_FILE}"
    tree = json.loads(read_url(f"{API}/tree/main/data/profiles/{key}?expand=true"))
    entry = next(item for item in tree if item.get("path") == path)
    commit = entry["lastCommit"]
    url = f"{SPACE}/resolve/{commit['id']}/{path}"
    if json.loads(read_url(url)) != pending:
        raise ValueError("public forecast differs from the frozen local commitment")
    timestamp = datetime.fromisoformat(commit["date"].replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("public commit timestamp needs a timezone")
    # Conservative eligibility: same-day uploads await exact official draw-time evidence.
    eligible = timestamp.astimezone(timezone.utc).date().isoformat() < pending["forecast"]["target_draw_date"]
    return {"revision": commit["id"], "published_utc": commit["date"],
            "url": url, "verified_before_draw_date": eligible}


def settle(pending: dict, history: pd.DataFrame, spec: GameSpec) -> dict | None:
    verify_commitment(pending)
    record = pending["forecast"]
    if record["game"] != {"name": spec.name, "main_n": spec.main_n, "main_k": spec.main_k,
                          "star_n": spec.star_n, "star_k": spec.star_k}:
        raise ValueError("frozen forecast game differs from the settlement game")
    target = history.loc[history["draw_date"].astype(str) == record["target_draw_date"]]
    if target.empty:
        return None
    if len(target) != 1:
        raise ValueError("settlement requires one unique official target draw")
    row = target.iloc[0]
    main = tuple(sorted(int(row[f"ball_{i}"]) for i in range(1, spec.main_k + 1)))
    stars = tuple(sorted(int(row[f"star_{i}"]) for i in range(1, spec.star_k + 1)))
    spec.validate_ticket((main, stars))
    score = score_ticket(record["score_matrices"], main, stars)
    measures = [{**band, "actual_g_score": score,
                 "full_winner_in_range": contains(band["intervals"], score)}
                for band in record["ranges"]]
    return {"draw_date": record["target_draw_date"], "actual_main": list(main),
            "actual_auxiliary": list(stars), "actual_g_score": score,
            "ranges": prize_measures(measures, spec, (main, stars), record["score_matrices"],
                                     row.get("reserve_number"))}


def write_benchmark(directory: Path, pending: dict, replay_rows: list[dict],
                    history: pd.DataFrame, spec: GameSpec, *, profile_key: str | None = None,
                    read_url=None) -> dict:
    """Append frozen forecasts; never convert a retrospective fit into live evidence."""
    pending_path, ledger_path = directory / PENDING_FILE, directory / LEDGER_FILE
    ledger = json.loads(ledger_path.read_text()) if ledger_path.is_file() else {"entries": []}
    if pending_path.is_file():
        previous = json.loads(pending_path.read_text())
        verify_commitment(previous)
        entry = next((item for item in ledger["entries"]
                      if item["pending"]["commitment_sha256"] == previous["commitment_sha256"]), None)
        if entry is None:
            entry = {"pending": previous, "publication": None, "settlement": None}
            ledger["entries"].append(entry)
        if profile_key and entry["publication"] is None:
            try:
                entry["publication"] = publication_receipt(previous, profile_key, read_url=read_url)
                entry.pop("publication_error", None)
            except Exception as exc:
                entry["publication_error"] = f"{type(exc).__name__}: {exc}"
        # Keep the first sealed forecast for the same target and method.
        old, new = previous["forecast"], pending["forecast"]
        if (old["target_draw_date"], old["method_version"]) == (new["target_draw_date"], new["method_version"]):
            pending = previous
    if not any(item["pending"]["commitment_sha256"] == pending["commitment_sha256"]
               for item in ledger["entries"]):
        ledger["entries"].append({"pending": pending, "publication": None, "settlement": None})
    live_rows = []
    for entry in ledger["entries"]:
        verified = bool((entry.get("publication") or {}).get("verified_before_draw_date"))
        latest_settlement = settle(entry["pending"], history, spec)
        # Operator feeds can expose a rolling window. Preserve an already settled
        # draw when it ages out; absence from today's feed is not a retraction.
        if latest_settlement is not None:
            entry["settlement"] = latest_settlement
        entry["evidence_scope"] = "VERIFIED_PRE_DRAW" if verified else "PUBLICATION_UNVERIFIED_OR_LATE"
        if verified and entry["settlement"]:
            live_rows.extend(entry["settlement"]["ranges"])
    summary = {
        "method_version": METHOD,
        "definition": "A hit means the entire official winning ticket belongs to the full inclusive score-range set, without truncation.",
        "interval_assumption": "Normal approximation using current GARCH variance; nominal levels are uncalibrated. Hybrid takes a union with the same scale around both targets.",
        "replay_scope": "RETROSPECTIVE_FORWARD_ONLY_NOT_PRE_DRAW_PUBLICATION",
        "replay": aggregate(replay_rows), "prospective": aggregate(live_rows),
        "pending": pending["forecast"]["ranges"],
        "target_draw_date": pending["forecast"]["target_draw_date"],
        "cash_roi_status": "UNASSESSED_NO_SETTLED_ALL_PRIZE_PAYOUTS",
        "cash_roi_formula": "(sum(all actual prize payouts) - sum(all ticket stakes)) / sum(all ticket stakes)",
    }
    directory.mkdir(parents=True, exist_ok=True)
    for name, payload in ((PENDING_FILE, pending), (LEDGER_FILE, ledger), (SUMMARY_FILE, summary)):
        (directory / name).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    frame = pd.DataFrame(replay_rows, columns=list(replay_rows[0]) if replay_rows else
                         ["range_id", "draw_date", "full_winner_in_range"])
    if "intervals" in frame:
        frame["intervals"] = frame["intervals"].map(json.dumps)
    frame.to_csv(directory / REPLAY_FILE, index=False, lineterminator="\n")
    return summary
