"""Exact transformer + GARCH branch candidate sets for the next unseen draw.

The two existing temporal providers forecast the next value of the historical pair-co-occurrence
(``G``) sequence.  This module turns their two forecasts into a transparent branch ranking over
the *entire* legal ticket universe: a ticket is preferred when its historical ``G`` score is close
to either forecast branch. Ties use a seeded, number-neutral permutation, not a
claim that lower-numbered tickets are more probable.

This is a research search-space reducer, not a probability model.  A candidate set covering
``m`` of ``N`` legal tickets has fair-draw jackpot coverage ``m / N`` regardless of its ranking.
The compressed million-row artifact is therefore kept separate from the fixed-budget benchmark
and is never presented as a purchased portfolio or as realized ROI.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from itertools import combinations, islice
from math import comb, exp, gcd, lgamma, log
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .pair_raster import _history_matrices
from .protocol import GameSpec, Ticket
from .providers.temporal import (
    GarchMarkovBranchProvider,
    SequenceTransformerProvider,
    _score_series,
)
from .roi import JackpotModel

ARCHIVE_FILE = "temporal_hybrid_candidates_1m.csv.gz"
PREVIEW_FILE = "temporal_hybrid_preview.csv"
BACKTEST_FILE = "temporal_hybrid_backtest.csv"
SUMMARY_FILE = "temporal_hybrid_summary.json"
METHOD_VERSION = "v2_loo_modern_era_neutral_ties"


def _tie_keys(indexes: np.ndarray, seed: int) -> np.ndarray:
    """Bijective SplitMix64 keys: stable ties without lexical number preference."""
    values = np.asarray(indexes, dtype=np.uint64) ^ np.uint64(seed)
    with np.errstate(over="ignore"):
        values = values + np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> 27)) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> 31)


def _model_history(history: pd.DataFrame, spec: GameSpec) -> pd.DataFrame:
    """Use the declared current 12-star rules, never infer a cutoff from outcomes."""
    if (spec.main_n, spec.main_k, spec.star_n, spec.star_k) == (50, 5, 12, 2):
        if "draw_date" not in history:
            raise ValueError("EuroMillions temporal history requires dated rule-era evidence")
        dates = pd.to_datetime(history["draw_date"], errors="raise")
        history = history.loc[dates >= pd.Timestamp("2016-09-27")].copy()
        if history.empty:
            raise ValueError("no history in the current EuroMillions 12-star era")
    return history


@dataclass(frozen=True)
class TemporalCandidateResult:
    """Compact in-memory result; the complete candidate set lives in ``archive_path``."""

    preview: pd.DataFrame
    backtest: pd.DataFrame
    summary: dict
    archive_path: Path


def _score_batches(history: pd.DataFrame, spec: GameSpec, batch_size: int):
    """Yield ``(global_indexes, main rows, auxiliary rows, G scores)`` in legal-ticket order."""
    main_pairs, auxiliary_pairs, cross_pairs = _history_matrices(history, spec)
    auxiliary = (
        list(combinations(range(1, spec.star_n + 1), spec.star_k))
        if spec.star_k
        else [()]
    )
    auxiliary_array = (
        np.asarray(auxiliary, dtype=np.int16)
        if spec.star_k
        else np.empty((1, 0), dtype=np.int16)
    )
    auxiliary_scores = np.zeros(len(auxiliary), dtype=np.int32)
    for first, second in combinations(range(spec.star_k), 2):
        auxiliary_scores += auxiliary_pairs[
            auxiliary_array[:, first], auxiliary_array[:, second]
        ]

    main_iterator = combinations(range(1, spec.main_n + 1), spec.main_k)
    main_offset = 0
    while True:
        chunk = list(islice(main_iterator, batch_size))
        if not chunk:
            return
        main_array = np.asarray(chunk, dtype=np.int16)
        main_scores = np.zeros(len(chunk), dtype=np.int32)
        for first, second in combinations(range(spec.main_k), 2):
            main_scores += main_pairs[main_array[:, first], main_array[:, second]]

        if spec.star_k:
            affinity = cross_pairs[main_array].sum(axis=1)
            cross_scores = np.zeros((len(chunk), len(auxiliary)), dtype=np.int32)
            for position in range(spec.star_k):
                cross_scores += affinity[:, auxiliary_array[:, position]]
            scores = main_scores[:, None] + auxiliary_scores[None, :] + cross_scores
        else:
            scores = main_scores[:, None]

        count = scores.size
        indexes = main_offset * len(auxiliary) + np.arange(count, dtype=np.int64)
        yield indexes, main_array, auxiliary_array, scores.reshape(-1)
        main_offset += len(chunk)


def _forecasts(
    history: pd.DataFrame, spec: GameSpec, transformer_epochs: int
) -> tuple[float, float, dict, dict]:
    _matrices, poi = _score_series(history, spec)
    garch_target, garch_diagnostics = GarchMarkovBranchProvider._forecast(poi, 52)
    transformer = SequenceTransformerProvider(epochs=transformer_epochs)
    transformer_target, transformer_diagnostics = transformer._forecast(poi)
    return garch_target, transformer_target, garch_diagnostics, transformer_diagnostics


def _score_order(histogram: Counter[int], first: float, second: float) -> list[int]:
    return sorted(
        histogram,
        key=lambda value: (
            min(abs(value - first), abs(value - second)),
            (abs(value - first) + abs(value - second)) / 2.0,
            -value,
        ),
    )


def _take_by_score(
    histogram: Counter[int], ordered_scores: list[int], candidate_size: int
) -> dict[int, int]:
    remaining = candidate_size
    selected: dict[int, int] = {}
    for score in ordered_scores:
        take = min(remaining, histogram[score])
        if take:
            selected[score] = take
            remaining -= take
        if remaining == 0:
            return selected
    raise RuntimeError("score histogram did not cover the requested candidate set")


def _combination_rank(values: tuple[int, ...], n: int, k: int) -> int:
    """Zero-based lexicographic rank of a 1-based combination."""
    rank = 0
    previous = 0
    for position, value in enumerate(values):
        for skipped in range(previous + 1, value):
            rank += comb(n - skipped, k - position - 1)
        previous = value
    return rank


def ticket_global_index(ticket: Ticket, spec: GameSpec) -> int:
    main, auxiliary = ticket
    spec.validate_ticket(ticket)
    main_rank = _combination_rank(main, spec.main_n, spec.main_k)
    auxiliary_rank = (
        _combination_rank(auxiliary, spec.star_n, spec.star_k) if spec.star_k else 0
    )
    return main_rank * spec.n_star_combinations() + auxiliary_rank


def _ticket_score(history: pd.DataFrame, spec: GameSpec, ticket: Ticket) -> int:
    main_pairs, auxiliary_pairs, cross_pairs = _history_matrices(history, spec)
    main, auxiliary = ticket
    score = sum(main_pairs[first, second] for first, second in combinations(main, 2))
    score += sum(
        auxiliary_pairs[first, second] for first, second in combinations(auxiliary, 2)
    )
    score += sum(cross_pairs[first, second] for first in main for second in auxiliary)
    return int(score)


def _rank_actual_ticket(
    history: pd.DataFrame,
    spec: GameSpec,
    actual: Ticket,
    first_target: float,
    second_target: float,
    batch_size: int,
    seed: int = 20260829,
) -> tuple[int, int, int, int]:
    histogram: Counter[int] = Counter()
    actual_index = ticket_global_index(actual, spec)
    actual_score = _ticket_score(history, spec, actual)
    actual_key = _tie_keys(np.array([actual_index]), seed)[0]
    same_score_through_actual = 0
    for indexes, _main, _auxiliary, scores in _score_batches(history, spec, batch_size):
        counts = np.bincount(scores)
        histogram.update({index: int(count) for index, count in enumerate(counts) if count})
        before = _tie_keys(indexes, seed) <= actual_key
        same_score_through_actual += int(np.count_nonzero((scores == actual_score) & before))
    ordered = _score_order(histogram, first_target, second_target)
    # Stop at the actual score band; later bands are not better.
    better = sum(histogram[score] for score in ordered[: ordered.index(actual_score)])
    return (
        int(better + same_score_through_actual), actual_score,
        int(better + 1), int(better + histogram[actual_score]),
    )


def _control_contains(index: int, universe: int, size: int, seed: int) -> bool:
    """Membership in an exact-size deterministic uniform control via an affine permutation."""
    multiplier = (2 * seed + 1) % universe
    multiplier = multiplier or 1
    while gcd(multiplier, universe) != 1:
        multiplier += 2
        if multiplier >= universe:
            multiplier = 1
    offset = int.from_bytes(hashlib.sha256(str(seed).encode()).digest()[:8], "big") % universe
    return ((multiplier * index + offset) % universe) < size


def _binomial_upper_tail(successes: int, trials: int, probability: float) -> float:
    if successes <= 0:
        return 1.0
    if probability >= 1.0:
        return 1.0
    if probability <= 0.0:
        return 0.0
    terms = []
    for value in range(successes, trials + 1):
        log_term = (
            lgamma(trials + 1)
            - lgamma(value + 1)
            - lgamma(trials - value + 1)
            + value * log(probability)
            + (trials - value) * log(1.0 - probability)
        )
        terms.append(exp(log_term))
    return float(min(1.0, sum(terms)))


def _walk_forward_backtest(
    history: pd.DataFrame,
    spec: GameSpec,
    candidate_size: int,
    *,
    holdout: int,
    transformer_epochs: int,
    batch_size: int,
    seed: int,
) -> pd.DataFrame:
    if holdout < 1 or len(history) <= holdout:
        return pd.DataFrame()
    rows = []
    main_columns = [f"ball_{index}" for index in range(1, spec.main_k + 1)]
    auxiliary_columns = [f"star_{index}" for index in range(1, spec.star_k + 1)]
    for fold, target_index in enumerate(range(len(history) - holdout, len(history)), start=1):
        training = history.iloc[:target_index].copy()
        target_row = history.iloc[target_index]
        actual: Ticket = (
            tuple(sorted(int(target_row[column]) for column in main_columns)),
            tuple(sorted(int(target_row[column]) for column in auxiliary_columns)),
        )
        first, second, _garch, _transformer = _forecasts(
            training, spec, transformer_epochs
        )
        rank, actual_score, band_first, band_last = _rank_actual_ticket(
            training, spec, actual, first, second, batch_size, seed
        )
        global_index = ticket_global_index(actual, spec)
        rows.append(
            {
                "fold": fold,
                "draw_date": str(target_row.get("draw_date", target_index)),
                "training_rows": len(training),
                "garch_target_g": first,
                "transformer_target_g": second,
                "actual_g_score": actual_score,
                "actual_rank": rank,
                "score_band_rank_first": band_first,
                "score_band_rank_last": band_last,
                "method_version": METHOD_VERSION,
                "hybrid_contains_winner": rank <= candidate_size,
                "matched_uniform_contains_winner": _control_contains(
                    global_index, spec.n_tickets(), candidate_size, seed + fold
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_archive(
    path: Path,
    *,
    numbers: np.ndarray,
    scores: np.ndarray,
    order: np.ndarray,
    spec: GameSpec,
    first_target: float,
    second_target: float,
    history_cutoff: str,
    target_draw_date: str,
    score_bands: dict[int, tuple[int, int]],
) -> tuple[str, list[dict]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    preview: list[dict] = []
    with gzip.open(path, "wt", newline="", encoding="utf-8", compresslevel=6) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "rank",
                "main_draw",
                "auxiliary_draw",
                "g_score",
                "nearest_branch",
                "distance_to_garch",
                "distance_to_transformer",
                "history_cutoff",
                "target_draw_date",
                "score_status",
                "score_band_rank_first",
                "score_band_rank_last",
            ]
        )
        for rank, selected_index in enumerate(order, start=1):
            selected_index = int(selected_index)
            values = numbers[selected_index]
            main = " ".join(str(int(value)) for value in values[: spec.main_k])
            auxiliary = " ".join(
                str(int(value))
                for value in values[spec.main_k : spec.main_k + spec.star_k]
            )
            score = int(scores[selected_index])
            garch_distance = abs(score - first_target)
            transformer_distance = abs(score - second_target)
            branch = "garch" if garch_distance <= transformer_distance else "transformer"
            row = {
                "rank": rank,
                "main_draw": main,
                "auxiliary_draw": auxiliary,
                "g_score": score,
                "nearest_branch": branch,
                "distance_to_garch": garch_distance,
                "distance_to_transformer": transformer_distance,
                "history_cutoff": history_cutoff,
                "target_draw_date": target_draw_date,
                "score_status": "PENDING",
                "score_band_rank_first": score_bands[score][0],
                "score_band_rank_last": score_bands[score][1],
            }
            writer.writerow(row.values())
            if rank <= 250:
                preview.append(row)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest, preview


def build_temporal_candidate_set(
    history: pd.DataFrame,
    spec: GameSpec,
    *,
    output_directory: str | Path,
    history_cutoff: str,
    target_draw_date: str,
    snapshot_sha256: str,
    jackpot: JackpotModel,
    candidate_size: int = 1_000_000,
    holdout: int = 12,
    transformer_epochs: int = 8,
    batch_size: int = 50_000,
    seed: int = 20260829,
) -> TemporalCandidateResult:
    """Build, validate, and write an exact pre-draw hybrid candidate artifact."""
    if history.empty:
        raise ValueError("temporal candidate set requires known history")
    source_history_rows = len(history)
    history = _model_history(history, spec)
    universe = spec.n_tickets()
    candidate_size = min(int(candidate_size), universe)
    if candidate_size < 1 or batch_size < 1:
        raise ValueError("candidate_size and batch_size must be positive")

    first, second, garch_diagnostics, transformer_diagnostics = _forecasts(
        history, spec, transformer_epochs
    )
    histogram: Counter[int] = Counter()
    for _indexes, _main, _auxiliary, scores in _score_batches(history, spec, batch_size):
        counts = np.bincount(scores)
        histogram.update({index: int(count) for index, count in enumerate(counts) if count})
    if sum(histogram.values()) != universe:
        raise RuntimeError("temporal score raster did not cover the full legal universe")
    ordered_scores = _score_order(histogram, first, second)
    take_by_score = _take_by_score(histogram, ordered_scores, candidate_size)
    score_level = {score: level for level, score in enumerate(ordered_scores)}
    score_bands = {}
    band_offset = 0
    for score in ordered_scores:
        score_bands[score] = (band_offset + 1, band_offset + histogram[score])
        band_offset += histogram[score]

    # Streaming top-k per score band: memory is bounded by the requested set plus
    # one batch. Selection and actual-ticket evaluation use the same tie keys.
    pools: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    auxiliary_count = spec.n_star_combinations()
    for indexes, main_array, auxiliary_array, scores in _score_batches(
        history, spec, batch_size
    ):
        for score, take in take_by_score.items():
            local = np.flatnonzero(scores == score)
            if not len(local):
                continue
            values = np.concatenate(
                (main_array[local // auxiliary_count], auxiliary_array[local % auxiliary_count]),
                axis=1,
            )
            keys = _tie_keys(indexes[local], seed)
            if score in pools:
                old_keys, old_values = pools[score]
                keys = np.concatenate((old_keys, keys))
                values = np.concatenate((old_values, values))
            if len(keys) > take:
                keep = np.argpartition(keys, take - 1)[:take]
                keys, values = keys[keep], values[keep]
            pools[score] = keys, values
    selected_keys = np.concatenate([pools[score][0] for score in take_by_score])
    numbers = np.concatenate([pools[score][1] for score in take_by_score])
    selected_scores = np.concatenate([
        np.full(len(pools[score][0]), score, dtype=np.int32) for score in take_by_score
    ])
    selected_levels = np.array([score_level[int(score)] for score in selected_scores])
    if len(numbers) != candidate_size:
        raise RuntimeError(f"selected {len(numbers)} tickets; expected {candidate_size}")
    order = np.lexsort((selected_keys, selected_levels))
    output_directory = Path(output_directory)
    archive_path = output_directory / ARCHIVE_FILE
    archive_sha256, preview_rows = _write_archive(
        archive_path,
        numbers=numbers,
        scores=selected_scores,
        order=order,
        spec=spec,
        first_target=first,
        second_target=second,
        history_cutoff=history_cutoff,
        target_draw_date=target_draw_date,
        score_bands=score_bands,
    )
    preview = pd.DataFrame(preview_rows)

    backtest = _walk_forward_backtest(
        history,
        spec,
        candidate_size,
        holdout=holdout,
        transformer_epochs=transformer_epochs,
        batch_size=batch_size,
        seed=seed,
    )
    model_hits = int(backtest["hybrid_contains_winner"].sum()) if not backtest.empty else 0
    control_hits = (
        int(backtest["matched_uniform_contains_winner"].sum()) if not backtest.empty else 0
    )
    fair_fraction = candidate_size / universe
    p_value = _binomial_upper_tail(model_hits, len(backtest), fair_fraction) if len(backtest) else 1.0
    repeatable_gate = model_hits >= 2 and p_value <= 0.01 and model_hits > control_hits
    stake = candidate_size * jackpot.ticket_price
    summary: dict[str, Any] = {
        "schema_version": "2.0.0",
        "method": "exact_transformer_garch_branch_raster",
        "method_version": METHOD_VERSION,
        "source_history_rows": source_history_rows,
        "training_score_definition": "leave_one_out_typed_pair_count_at_training_cutoff",
        "history_rule_era": "12_star_since_2016_09_27" if spec.star_n == 12 else "supplied_game_rules",
        "research_status": "EVIDENCE_GATE_PASSED" if repeatable_gate else "RESEARCH_ONLY",
        "profitability_status": "UNASSESSED_NO_PURCHASE_OR_SETTLED_PAYOUT",
        "interpretation": (
            "Exact candidate ranking by distance to either temporal G forecast. It does not "
            "change fair-draw ticket probability and is not betting advice."
        ),
        "history_cutoff": history_cutoff,
        "target_draw_date": target_draw_date,
        "snapshot_sha256": snapshot_sha256,
        "history_rows": int(len(history)),
        "score_history_first_draw": str(history.iloc[0].get("draw_date", "unknown")),
        "universe_size": universe,
        "candidate_size": candidate_size,
        "mechanical_jackpot_coverage_pct": fair_fraction * 100.0,
        "mechanical_jackpot_odds_one_in": universe / candidate_size,
        "full_purchase_stake": stake,
        "currency": "EUR",
        "break_even_gross_payout": stake,
        "jackpot_containment_alone_proves_profit": False,
        "garch_target_g": first,
        "transformer_target_g": second,
        "garch_diagnostics": garch_diagnostics,
        "transformer_diagnostics": transformer_diagnostics,
        "ranking_tie_break": "nearest branch, mean branch distance, higher G, seeded SplitMix64 key",
        "tie_seed": seed,
        "rank_interpretation": "1 is nearest score band; order within a band is not predictive confidence",
        "selected_score_bands": [
            {"g_score": score, "universe_count": histogram[score], "selected_count": take,
             "rank_first": score_bands[score][0], "rank_last": score_bands[score][1]}
            for score, take in take_by_score.items()
        ],
        "forward_only_gate": {
            "holdout_draws": int(len(backtest)),
            "hybrid_jackpot_containment_hits": model_hits,
            "matched_uniform_containment_hits": control_hits,
            "fair_expected_hits": len(backtest) * fair_fraction,
            "one_sided_binomial_p_value": p_value,
            "requires_at_least_two_hits": True,
            "passed": repeatable_gate,
        },
        "artifacts": {
            ARCHIVE_FILE: archive_sha256,
            PREVIEW_FILE: None,
            BACKTEST_FILE: None,
        },
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    preview_path = output_directory / PREVIEW_FILE
    backtest_path = output_directory / BACKTEST_FILE
    preview.to_csv(preview_path, index=False, lineterminator="\n")
    backtest.to_csv(backtest_path, index=False, lineterminator="\n")
    summary["artifacts"][PREVIEW_FILE] = hashlib.sha256(preview_path.read_bytes()).hexdigest()
    summary["artifacts"][BACKTEST_FILE] = hashlib.sha256(backtest_path.read_bytes()).hexdigest()
    (output_directory / SUMMARY_FILE).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return TemporalCandidateResult(preview, backtest, summary, archive_path)
