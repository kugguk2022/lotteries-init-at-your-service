"""Exhaustive, forward-only historical pair-density raster for legal lottery tickets.

The raster translates the repository owner's original R idea into a bounded-memory build:
every legal ticket is scored by the historical co-occurrence counts of its main-number pairs,
auxiliary pairs, and main/auxiliary pairs.  Only the top rows and the exact score distribution
are published, so a 139-million-ticket EuroMillions universe does not become a giant artifact.

Pair density is a descriptive ordering of past co-occurrence.  It is not a probability model and
does not make any fair-draw ticket mechanically more likely to win.
"""

from __future__ import annotations

import hashlib
import heapq
from collections import Counter
from dataclasses import dataclass
from itertools import combinations, islice
from math import comb

import numpy as np
import pandas as pd

from lotteries_core.protocol import GameSpec


@dataclass(frozen=True)
class PairRasterResult:
    """Compact publication result for one exhaustive ticket-universe scan."""

    top: pd.DataFrame
    distribution: pd.DataFrame
    summary: dict


def _history_matrices(
    history: pd.DataFrame, spec: GameSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    main_columns = [f"ball_{index}" for index in range(1, spec.main_k + 1)]
    auxiliary_columns = [f"star_{index}" for index in range(1, spec.star_k + 1)]
    missing = [
        column for column in [*main_columns, *auxiliary_columns] if column not in history.columns
    ]
    if missing:
        raise ValueError(f"history is missing required draw columns: {missing}")

    main_pairs = np.zeros((spec.main_n + 1, spec.main_n + 1), dtype=np.int32)
    auxiliary_pairs = np.zeros((spec.star_n + 1, spec.star_n + 1), dtype=np.int32)
    cross_pairs = np.zeros((spec.main_n + 1, spec.star_n + 1), dtype=np.int32)

    main_values = history[main_columns].to_numpy(dtype=np.int16)
    auxiliary_values = (
        history[auxiliary_columns].to_numpy(dtype=np.int16)
        if auxiliary_columns
        else np.empty((len(history), 0), dtype=np.int16)
    )
    if main_values.size and (main_values.min() < 1 or main_values.max() > spec.main_n):
        raise ValueError("history contains a main number outside the game specification")
    if auxiliary_values.size and (
        auxiliary_values.min() < 1 or auxiliary_values.max() > spec.star_n
    ):
        raise ValueError("history contains an auxiliary number outside the game specification")

    for main, auxiliary in zip(main_values, auxiliary_values, strict=True):
        for first, second in combinations((int(value) for value in main), 2):
            main_pairs[first, second] += 1
            main_pairs[second, first] += 1
        for first, second in combinations((int(value) for value in auxiliary), 2):
            auxiliary_pairs[first, second] += 1
            auxiliary_pairs[second, first] += 1
        for main_number in main:
            for auxiliary_number in auxiliary:
                cross_pairs[int(main_number), int(auxiliary_number)] += 1
    return main_pairs, auxiliary_pairs, cross_pairs


def _weighted_quantile(histogram: Counter[int], quantile: float) -> int:
    total = sum(histogram.values())
    target = max(1, int(np.ceil(total * quantile)))
    seen = 0
    for score, count in sorted(histogram.items()):
        seen += count
        if seen >= target:
            return int(score)
    raise ValueError("cannot calculate a quantile from an empty histogram")


def build_pair_raster(
    history: pd.DataFrame,
    spec: GameSpec,
    *,
    snapshot_sha256: str,
    history_cutoff: str,
    target_draw_date: str,
    top_n: int = 250,
    batch_size: int = 50_000,
) -> PairRasterResult:
    """Score every legal ticket and return a compact, auditable publication bundle.

    Enumeration is lexicographic and the deterministic tie-break is the lower legal-ticket index.
    The input ``history`` is the sole evidence source, so callers control the forward-only cutoff.
    """

    if history.empty:
        raise ValueError("pair raster requires at least one historical draw")
    if top_n < 1 or batch_size < 1:
        raise ValueError("top_n and batch_size must be positive")

    main_pair_counts, auxiliary_pair_counts, cross_pair_counts = _history_matrices(history, spec)
    auxiliary_combinations = (
        list(combinations(range(1, spec.star_n + 1), spec.star_k))
        if spec.star_k
        else [()]
    )
    auxiliary_array = (
        np.asarray(auxiliary_combinations, dtype=np.int16)
        if spec.star_k
        else np.empty((1, 0), dtype=np.int16)
    )
    auxiliary_scores = np.zeros(len(auxiliary_combinations), dtype=np.int32)
    for first, second in combinations(range(spec.star_k), 2):
        auxiliary_scores += auxiliary_pair_counts[
            auxiliary_array[:, first], auxiliary_array[:, second]
        ]

    main_pair_positions = list(combinations(range(spec.main_k), 2))
    total_main_combinations = comb(spec.main_n, spec.main_k)
    total_ticket_combinations = total_main_combinations * len(auxiliary_combinations)
    ranking_scale = total_ticket_combinations + 1
    histogram: Counter[int] = Counter()
    candidate_heap: list[tuple[int, int, tuple[int, ...], tuple[int, ...], int, int, int]] = []

    main_iterator = combinations(range(1, spec.main_n + 1), spec.main_k)
    main_offset = 0
    while True:
        chunk = list(islice(main_iterator, batch_size))
        if not chunk:
            break
        main_array = np.asarray(chunk, dtype=np.int16)
        main_scores = np.zeros(len(chunk), dtype=np.int32)
        for first, second in main_pair_positions:
            main_scores += main_pair_counts[main_array[:, first], main_array[:, second]]

        if spec.star_k:
            affinity = cross_pair_counts[main_array].sum(axis=1)
            cross_scores = np.zeros((len(chunk), len(auxiliary_combinations)), dtype=np.int32)
            for position in range(spec.star_k):
                cross_scores += affinity[:, auxiliary_array[:, position]]
            full_scores = (
                main_scores[:, None] + auxiliary_scores[None, :] + cross_scores
            )
        else:
            cross_scores = np.zeros((len(chunk), 1), dtype=np.int32)
            full_scores = main_scores[:, None]

        flat_scores = full_scores.reshape(-1)
        score_counts = np.bincount(flat_scores)
        histogram.update(
            {int(score): int(count) for score, count in enumerate(score_counts) if count}
        )

        local_ticket_count = len(flat_scores)
        local_top_n = min(top_n, local_ticket_count)
        local_indexes = np.arange(local_ticket_count, dtype=np.int64)
        global_indexes = (
            main_offset * len(auxiliary_combinations) + local_indexes
        )
        ranking_keys = flat_scores.astype(np.int64) * ranking_scale - global_indexes
        selected = np.argpartition(ranking_keys, -local_top_n)[-local_top_n:]
        for local_index in selected:
            local_index = int(local_index)
            main_index, auxiliary_index = divmod(
                local_index, len(auxiliary_combinations)
            )
            main = tuple(int(value) for value in main_array[main_index])
            auxiliary = tuple(int(value) for value in auxiliary_combinations[auxiliary_index])
            score = int(full_scores[main_index, auxiliary_index])
            global_index = (
                (main_offset + main_index) * len(auxiliary_combinations) + auxiliary_index
            )
            entry = (
                score * ranking_scale - global_index,
                global_index,
                main,
                auxiliary,
                int(main_scores[main_index]),
                int(auxiliary_scores[auxiliary_index]),
                int(cross_scores[main_index, auxiliary_index]),
            )
            if len(candidate_heap) < top_n:
                heapq.heappush(candidate_heap, entry)
            elif entry[0] > candidate_heap[0][0]:
                heapq.heapreplace(candidate_heap, entry)
        main_offset += len(chunk)

    if main_offset != total_main_combinations:
        raise RuntimeError(
            f"enumerated {main_offset} main combinations; expected {total_main_combinations}"
        )
    if sum(histogram.values()) != total_ticket_combinations:
        raise RuntimeError("pair-raster score distribution does not cover the full ticket universe")

    cumulative_by_score: dict[int, int] = {}
    cumulative = 0
    for score, count in sorted(histogram.items()):
        cumulative += count
        cumulative_by_score[score] = cumulative

    ranked_entries = sorted(candidate_heap, key=lambda entry: entry[0], reverse=True)
    top_rows = []
    for rank, entry in enumerate(ranked_entries, start=1):
        _, _, main, auxiliary, main_score, auxiliary_score, cross_score = entry
        score = main_score + auxiliary_score + cross_score
        commitment_source = (
            f"{snapshot_sha256}|after:{history_cutoff}|target:{target_draw_date}|"
            f"pair-density|{rank}|{main}|{auxiliary}|{score}"
        )
        top_rows.append(
            {
                "rank": rank,
                "main_draw": " ".join(str(value) for value in main),
                "auxiliary_draw": " ".join(str(value) for value in auxiliary),
                "historical_pair_density_score": score,
                "main_pair_score": main_score,
                "auxiliary_pair_score": auxiliary_score,
                "main_auxiliary_cross_score": cross_score,
                "universe_percentile_pct": (
                    cumulative_by_score[score] / total_ticket_combinations * 100
                ),
                "history_cutoff": history_cutoff,
                "target_draw_date": target_draw_date,
                "score_status": "PENDING",
                "commitment_sha256": hashlib.sha256(
                    commitment_source.encode("utf-8")
                ).hexdigest(),
            }
        )

    distribution_rows = []
    cumulative = 0
    for score, count in sorted(histogram.items()):
        cumulative += count
        distribution_rows.append(
            {
                "historical_pair_density_score": score,
                "ticket_combinations": count,
                "share_pct": count / total_ticket_combinations * 100,
                "cumulative_pct": cumulative / total_ticket_combinations * 100,
            }
        )

    score_min = min(histogram)
    score_max = max(histogram)
    summary = {
        "schema_version": "1.0.0",
        "method": "exhaustive_historical_pair_density",
        "interpretation": (
            "Descriptive past co-occurrence ranking; not jackpot probability, a prediction, "
            "or betting advice."
        ),
        "history_cutoff": history_cutoff,
        "target_draw_date": target_draw_date,
        "snapshot_sha256": snapshot_sha256,
        "historical_draws": int(len(history)),
        "main_combinations_scored": total_main_combinations,
        "auxiliary_combinations_per_main": len(auxiliary_combinations),
        "ticket_combinations_scored": total_ticket_combinations,
        "published_top_rows": len(top_rows),
        "score_distribution": {
            "minimum": score_min,
            "median": _weighted_quantile(histogram, 0.50),
            "p90": _weighted_quantile(histogram, 0.90),
            "p99": _weighted_quantile(histogram, 0.99),
            "maximum": score_max,
        },
    }
    return PairRasterResult(
        top=pd.DataFrame(top_rows),
        distribution=pd.DataFrame(distribution_rows),
        summary=summary,
    )
