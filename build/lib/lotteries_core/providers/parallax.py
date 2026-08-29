"""Parallax Guard: replicated residual evidence plus a coverage-first portfolio optimizer.

The name describes the admission rule.  A putative draw signal is viewed through two disjoint,
interleaved halves of the available history.  A number or pair is admitted only when both views
show the same signed deviation from the exact fair-draw expectation *and* the weaker view survives
a family-wise (Bonferroni) normal threshold.  A one-window hot streak therefore contributes exactly
zero.  Closed co-occurrence cliques are visible here because the statistic is pair excess, not graph
centrality, but ordinary sampling noise is normally rejected.

The learned residual is deliberately not the main source of value.  Candidate tickets are selected
as one portfolio with a greedy marginal objective:

* new main-number pairs (combinatorial reach),
* balanced main and star usage (no pinned-star concentration),
* new star pairs,
* low estimated crowd popularity (the jackpot-sharing lever), and
* the replicated residual, only when the guard admits it.

``mode="ablation"`` runs the identical candidate generator and portfolio objective with the learned
residual set to zero.  Thus any difference between ``parallax_guard`` and
``parallax_guard_ablation`` can be attributed to the historical signal; performance shared by both
belongs to the portfolio construction.  Neither mode changes a fair ticket's draw probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import ceil
from statistics import NormalDist

import numpy as np
import pandas as pd

from ..coverage import coverage_report
from ..likely_set_generator import GameConfig, build_comatrices, detect_columns
from ..popularity import PopularityModel
from ..protocol import GameSpec, InferenceProvider, ProviderResult, Ticket

MODES = ("guarded", "ablation")

# First draw under the current 5-of-50 + 2-of-12 EuroMillions matrix.  Older rows used smaller
# Lucky Star pools and are valid historical records, but they are not observations of today's null.
_CURRENT_RULE_START = {("euromillions", 50, 5, 12, 2): pd.Timestamp("2016-09-27")}


@dataclass(frozen=True)
class ReplicatedEvidence:
    """Guarded standardized residuals for every supported ticket component."""

    main_number: np.ndarray
    main_pair: np.ndarray
    cross_pair: np.ndarray
    star_number: np.ndarray
    star_pair: np.ndarray
    thresholds: dict[str, float]
    fold_rows: tuple[int, int]

    @property
    def nonzero(self) -> int:
        # Pair matrices are symmetric, so count only one triangle for those two families.
        return int(
            np.count_nonzero(self.main_number)
            + np.count_nonzero(np.triu(self.main_pair, 1))
            + np.count_nonzero(self.cross_pair)
            + np.count_nonzero(self.star_number)
            + np.count_nonzero(np.triu(self.star_pair, 1))
        )

    @property
    def max_abs(self) -> float:
        arrays = (
            self.main_number,
            self.main_pair,
            self.cross_pair,
            self.star_number,
            self.star_pair,
        )
        return max((float(np.max(np.abs(a))) for a in arrays if a.size), default=0.0)


def _number_counts(df: pd.DataFrame, columns: list[str], population: int) -> np.ndarray:
    counts = np.zeros(population, dtype=float)
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce").dropna().astype(int)
        valid = values[(values >= 1) & (values <= population)]
        if len(valid):
            counts += np.bincount(valid.to_numpy() - 1, minlength=population)
    return counts


def _current_rules_history(history: pd.DataFrame, spec: GameSpec) -> pd.DataFrame:
    """Return only rows generated under ``spec`` when a game has documented matrix changes."""
    start = _CURRENT_RULE_START.get(
        (spec.name.lower(), spec.main_n, spec.main_k, spec.star_n, spec.star_k)
    )
    if start is None:
        return history
    date_column = next(
        (column for column in ("draw_date", "date") if column in history.columns), None
    )
    if date_column is None:
        raise ValueError(
            f"{spec.name} history needs a draw_date/date column to isolate the current rules regime"
        )
    dates = pd.to_datetime(history[date_column], errors="coerce")
    current = history.loc[dates >= start]
    if current.empty:
        raise ValueError(f"history has no {spec.name} rows under the current rules since {start.date()}")
    return current


def _replicated_excess(
    count_a: np.ndarray,
    count_b: np.ndarray,
    *,
    rows_a: int,
    rows_b: int,
    fair_probability: float,
    n_tests: int,
    alpha: float,
) -> tuple[np.ndarray, float]:
    """Return soft-thresholded residuals replicated in both disjoint folds.

    Counts across draws follow a binomial marginal under the fair null.  The returned magnitude is
    the weaker absolute z-score minus the family-wise threshold.  Opposite signs, threshold misses,
    empty folds, and deterministic probabilities all return zero.
    """
    threshold = NormalDist().inv_cdf(1.0 - alpha / (2.0 * max(1, n_tests)))
    if rows_a <= 0 or rows_b <= 0 or not (0.0 < fair_probability < 1.0):
        return np.zeros_like(count_a, dtype=float), float(threshold)

    scale_a = np.sqrt(rows_a * fair_probability * (1.0 - fair_probability))
    scale_b = np.sqrt(rows_b * fair_probability * (1.0 - fair_probability))
    z_a = (np.asarray(count_a, dtype=float) - rows_a * fair_probability) / scale_a
    z_b = (np.asarray(count_b, dtype=float) - rows_b * fair_probability) / scale_b
    same_direction = np.signbit(z_a) == np.signbit(z_b)
    weaker = np.minimum(np.abs(z_a), np.abs(z_b))
    magnitude = np.where(same_direction, np.maximum(weaker - threshold, 0.0), 0.0)
    return np.sign(z_a) * magnitude, float(threshold)


def replicated_evidence(
    history: pd.DataFrame,
    spec: GameSpec,
    *,
    alpha: float = 0.05,
) -> ReplicatedEvidence:
    """Compute pair and marginal evidence that replicates in alternating history folds."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie strictly inside (0, 1), got {alpha}")

    history = _current_rules_history(history, spec)
    cfg = GameConfig(spec.name, spec.main_n, spec.main_k, spec.star_n, spec.star_k)
    main_cols, star_cols = detect_columns(history, cfg)
    if len(main_cols) != spec.main_k or len(star_cols) != spec.star_k:
        raise ValueError(
            f"history columns do not match {spec.name}: mains={main_cols}, stars={star_cols}"
        )

    fold_a = history.iloc[::2]
    fold_b = history.iloc[1::2]
    rows_a, rows_b = len(fold_a), len(fold_b)
    matrix_a = build_comatrices(fold_a, cfg, main_cols, star_cols)
    matrix_b = build_comatrices(fold_b, cfg, main_cols, star_cols)

    thresholds: dict[str, float] = {}
    main_number, thresholds["main_number"] = _replicated_excess(
        _number_counts(fold_a, main_cols, spec.main_n),
        _number_counts(fold_b, main_cols, spec.main_n),
        rows_a=rows_a,
        rows_b=rows_b,
        fair_probability=spec.main_k / spec.main_n,
        n_tests=spec.main_n,
        alpha=alpha,
    )
    main_pair, thresholds["main_pair"] = _replicated_excess(
        matrix_a.Wmain,
        matrix_b.Wmain,
        rows_a=rows_a,
        rows_b=rows_b,
        fair_probability=spec.main_k * (spec.main_k - 1) / (spec.main_n * (spec.main_n - 1)),
        n_tests=spec.main_n * (spec.main_n - 1) // 2,
        alpha=alpha,
    )

    if spec.star_k > 0:
        star_number, thresholds["star_number"] = _replicated_excess(
            _number_counts(fold_a, star_cols, spec.star_n),
            _number_counts(fold_b, star_cols, spec.star_n),
            rows_a=rows_a,
            rows_b=rows_b,
            fair_probability=spec.star_k / spec.star_n,
            n_tests=spec.star_n,
            alpha=alpha,
        )
        cross_pair, thresholds["cross_pair"] = _replicated_excess(
            matrix_a.Wcross[:, : spec.star_n],
            matrix_b.Wcross[:, : spec.star_n],
            rows_a=rows_a,
            rows_b=rows_b,
            fair_probability=(spec.main_k / spec.main_n) * (spec.star_k / spec.star_n),
            n_tests=spec.main_n * spec.star_n,
            alpha=alpha,
        )
        if spec.star_k >= 2:
            star_pair, thresholds["star_pair"] = _replicated_excess(
                matrix_a.Wstar[: spec.star_n, : spec.star_n],
                matrix_b.Wstar[: spec.star_n, : spec.star_n],
                rows_a=rows_a,
                rows_b=rows_b,
                fair_probability=(spec.star_k * (spec.star_k - 1))
                / (spec.star_n * (spec.star_n - 1)),
                n_tests=spec.star_n * (spec.star_n - 1) // 2,
                alpha=alpha,
            )
        else:
            star_pair = np.zeros((spec.star_n, spec.star_n), dtype=float)
            thresholds["star_pair"] = 0.0
    else:
        star_number = np.zeros(0, dtype=float)
        cross_pair = np.zeros((spec.main_n, 0), dtype=float)
        star_pair = np.zeros((0, 0), dtype=float)
        thresholds.update(star_number=0.0, cross_pair=0.0, star_pair=0.0)

    np.fill_diagonal(main_pair, 0.0)
    if star_pair.size:
        np.fill_diagonal(star_pair, 0.0)
    return ReplicatedEvidence(
        main_number=main_number,
        main_pair=main_pair,
        cross_pair=cross_pair,
        star_number=star_number,
        star_pair=star_pair,
        thresholds=thresholds,
        fold_rows=(rows_a, rows_b),
    )


def _ticket_signal(ticket: Ticket, evidence: ReplicatedEvidence) -> float:
    main, star = ticket
    main0 = tuple(v - 1 for v in main)
    star0 = tuple(v - 1 for v in star)
    score = float(evidence.main_number[list(main0)].sum())
    score += sum(float(evidence.main_pair[a, b]) for a, b in combinations(main0, 2))
    if star0:
        score += float(evidence.star_number[list(star0)].sum())
        score += sum(float(evidence.star_pair[a, b]) for a, b in combinations(star0, 2))
        score += sum(float(evidence.cross_pair[m, s]) for m in main0 for s in star0)
    return score


def _rank01(values: np.ndarray) -> np.ndarray:
    """Stable percentile ranks, returning zero when a component carries no information."""
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.allclose(values, values[0]):
        return np.zeros(values.shape, dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, values.size)
    return ranks


def _candidate_pool(
    spec: GameSpec,
    size: int,
    rng: np.random.Generator,
    popularity: PopularityModel,
) -> list[Ticket]:
    """Build a mixed uniform/crowd-contrarian pool without conditioning on history."""
    size = min(max(1, int(size)), spec.n_tickets())
    main_population = np.arange(1, spec.main_n + 1)
    star_population = np.arange(1, spec.star_n + 1)
    inverse_popularity = 1.0 / (popularity.number_weights(spec) + 1e-12)
    inverse_popularity /= inverse_popularity.sum()
    seen: set[Ticket] = set()
    out: list[Ticket] = []
    attempts = 0
    while len(out) < size and attempts < size * 100:
        # Alternating proposals keep the signal search broad while also supplying low-crowd options.
        probabilities = inverse_popularity if attempts % 2 else None
        main = tuple(
            sorted(
                int(v)
                for v in rng.choice(
                    main_population, spec.main_k, replace=False, p=probabilities
                )
            )
        )
        star = (
            tuple(
                sorted(
                    int(v)
                    for v in rng.choice(star_population, spec.star_k, replace=False)
                )
            )
            if spec.star_k > 0
            else ()
        )
        ticket = (main, star)
        attempts += 1
        if ticket not in seen:
            seen.add(ticket)
            out.append(ticket)
    return out


def _select_portfolio(
    spec: GameSpec,
    candidates: list[Ticket],
    signal: np.ndarray,
    crowd: np.ndarray,
    budget: int,
    *,
    max_shared_main: int,
) -> tuple[list[Ticket], np.ndarray]:
    """Greedily maximize marginal reach, balance, crowd avoidance, and guarded evidence."""
    budget = min(int(budget), len(candidates), spec.n_tickets())
    signal_rank = _rank01(signal)
    crowd_rank = _rank01(crowd)
    main_sets = [frozenset(t[0]) for t in candidates]
    main_pairs = [frozenset(combinations(t[0], 2)) for t in candidates]
    star_pairs = [frozenset(combinations(t[1], 2)) for t in candidates]

    selected: list[int] = []
    chosen: set[int] = set()
    used_main: set[int] = set()
    used_main_pairs: set[tuple[int, int]] = set()
    used_star_pairs: set[tuple[int, int]] = set()
    main_usage = np.zeros(spec.main_n + 1, dtype=int)
    star_usage = np.zeros(spec.star_n + 1, dtype=int)
    marginal_scores: list[float] = []
    star_cap = ceil(budget * spec.star_k / spec.star_n) if spec.star_k > 0 else 0
    star_balance_slack = 1

    while len(selected) < budget:
        best_index: int | None = None
        best_utility = -float("inf")
        relax_overlap = len(selected) + 1 == budget
        for index, (main, star) in enumerate(candidates):
            if index in chosen:
                continue
            if not relax_overlap and any(
                len(main_sets[index] & main_sets[prior]) > max_shared_main for prior in selected
            ):
                continue
            if star and any(star_usage[value] >= star_cap for value in star):
                continue
            if star:
                next_star_usage = star_usage.copy()
                next_star_usage[list(star)] += 1
                if int(np.ptp(next_star_usage[1:])) > star_balance_slack:
                    continue

            new_pair_fraction = len(main_pairs[index] - used_main_pairs) / max(
                1, len(main_pairs[index])
            )
            new_number_fraction = len(main_sets[index] - used_main) / spec.main_k
            main_balance = 1.0 - float(np.mean(main_usage[list(main)])) / max(1, len(selected))
            if star:
                star_balance = 1.0 - float(np.mean(star_usage[list(star)])) / max(
                    1, len(selected)
                )
                new_star_pair = len(star_pairs[index] - used_star_pairs) / max(
                    1, len(star_pairs[index])
                )
            else:
                star_balance = 1.0
                new_star_pair = 0.0

            # Pair reach dominates by design.  Signal cannot overpower gross portfolio duplication.
            utility = (
                4.0 * new_pair_fraction
                + 2.0 * new_number_fraction
                + 0.35 * main_balance
                + 0.75 * star_balance
                + 0.30 * new_star_pair
                + 0.85 * crowd_rank[index]
                + 0.90 * signal_rank[index]
            )
            if utility > best_utility:
                best_index = index
                best_utility = utility

        if best_index is None:
            # A strict overlap cap can exhaust a small candidate pool; relax only to meet equal budget.
            max_shared_main += 1
            if max_shared_main >= spec.main_k:
                star_cap += 1
                star_balance_slack += 1
            continue

        selected.append(best_index)
        chosen.add(best_index)
        marginal_scores.append(best_utility)
        main, star = candidates[best_index]
        used_main.update(main)
        used_main_pairs.update(main_pairs[best_index])
        used_star_pairs.update(star_pairs[best_index])
        main_usage[list(main)] += 1
        if star:
            star_usage[list(star)] += 1

    return [candidates[i] for i in selected], np.asarray(marginal_scores, dtype=float)


class ParallaxGuardProvider(InferenceProvider):
    """Replicated-residual likely-set generator with an explicit signal-off ablation."""

    name = "parallax_guard"
    description = (
        "Admits only fair-null residuals replicated in two disjoint history views, then optimizes "
        "the whole ticket set for pair reach, balanced stars, and low crowd popularity. Includes "
        "an identical signal-off ablation; claims no change to fair-draw odds."
    )

    def __init__(
        self,
        *,
        mode: str = "guarded",
        alpha: float = 0.05,
        oversample: int = 250,
        max_shared_main: int | None = None,
        popularity: PopularityModel | None = None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if oversample < 1:
            raise ValueError("oversample must be positive")
        self.mode = mode
        self.name = "parallax_guard" if mode == "guarded" else "parallax_guard_ablation"
        self.alpha = float(alpha)
        self.oversample = int(oversample)
        self.max_shared_main = max_shared_main
        self.popularity = popularity or PopularityModel()
        self._evidence: ReplicatedEvidence | None = None
        self._source_history_rows = 0
        self._history_rows = 0

    def fit(self, history: pd.DataFrame, spec: GameSpec | None = None) -> ParallaxGuardProvider:
        if spec is None:
            raise ValueError("ParallaxGuardProvider.fit requires a GameSpec")
        self._source_history_rows = len(history)
        current_history = _current_rules_history(history, spec)
        self._history_rows = len(current_history)
        self._evidence = replicated_evidence(current_history, spec, alpha=self.alpha)
        return self

    def propose(self, spec: GameSpec, budget: int, rng: np.random.Generator) -> ProviderResult:
        if self._evidence is None:
            raise RuntimeError("ParallaxGuardProvider.propose called before fit()")
        pool_size = min(max(budget * self.oversample, budget), spec.n_tickets())
        candidates = _candidate_pool(spec, pool_size, rng, self.popularity)
        raw_signal = np.asarray(
            [_ticket_signal(ticket, self._evidence) for ticket in candidates], dtype=float
        )
        applied_signal = raw_signal if self.mode == "guarded" else np.zeros_like(raw_signal)
        crowd = -np.log(
            np.asarray(
                [self.popularity.ticket_popularity(spec, ticket) for ticket in candidates],
                dtype=float,
            )
            + 1e-12
        )
        max_shared_main = (
            self.max_shared_main
            if self.max_shared_main is not None
            else max(1, spec.main_k - 3)
        )
        tickets, scores = _select_portfolio(
            spec,
            candidates,
            applied_signal,
            crowd,
            budget,
            max_shared_main=max_shared_main,
        )
        report = coverage_report(spec, tickets)
        star_counts = np.zeros(spec.star_n, dtype=int)
        for _main, star in tickets:
            if star:
                star_counts[np.asarray(star) - 1] += 1
        return ProviderResult(
            tickets=tickets,
            scores=scores,
            diagnostics={
                "mode": self.mode,
                "signal_applied": self.mode == "guarded",
                "alpha_family_wise": self.alpha,
                "source_history_rows": self._source_history_rows,
                "history_rows": self._history_rows,
                "fold_rows": list(self._evidence.fold_rows),
                "evidence_nonzero": self._evidence.nonzero,
                "evidence_max_abs": self._evidence.max_abs,
                "thresholds": self._evidence.thresholds,
                "candidate_pool": len(candidates),
                "pair_coverage": report["pair_coverage"],
                "number_coverage": report["number_coverage"],
                "star_usage_min": int(star_counts.min()) if star_counts.size else 0,
                "star_usage_max": int(star_counts.max()) if star_counts.size else 0,
            },
        )
