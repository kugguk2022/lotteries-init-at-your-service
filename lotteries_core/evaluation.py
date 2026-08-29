"""Forward-only, equal-budget evaluation -- the framework's single source of truth for "is it better?".

Rules enforced here (see ``repurpose.md`` for why they are non-negotiable):

* **Forward-only.** At holdout draw ``t`` every provider is fit on draws ``< t`` only, then scored
  against draw ``t``. No holdout draw ever informs the state that scores it.
* **Equal budget.** Every provider and the aggregated portfolio spend the *same* number of tickets
  ``B`` at every step, so comparisons are fair by construction.
* **Coverage + ROI, not just hits.** Because hitting a fair draw is astronomically unlikely in any
  realistic window, the primary reported metrics are combinatorial coverage and expected
  unpopularity-adjusted ROI; realised hit-recall is reported too but treated as high-variance noise.

The headline comparison is *best single provider* versus *coordinated aggregation* at identical
budget -- the operational form of the research question.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .aggregation import AggregationWeights, aggregate
from .coverage import coverage_report
from .envelope import InferenceEnvelope
from .popularity import PopularityModel
from .protocol import GameSpec, InferenceProvider, Ticket
from .roi import JackpotModel, default_jackpot_model, portfolio_expected_roi


@dataclass
class StepResult:
    t: int
    draw_date: str = ""
    actual_main: tuple[int, ...] = ()
    actual_star: tuple[int, ...] = ()
    per_provider: dict[str, dict] = field(default_factory=dict)
    per_provider_tickets: dict[str, list[Ticket]] = field(default_factory=dict)
    aggregated: dict = field(default_factory=dict)
    aggregated_tickets: list[Ticket] = field(default_factory=list)


def _actual_main(row: pd.Series, spec: GameSpec, main_cols: list[str]) -> set[int]:
    return {int(row[c]) for c in main_cols}


def _hit_recall(tickets: list[Ticket], actual_main: set[int], spec: GameSpec) -> float:
    """Mean fraction of a ticket's main numbers that appear in the actual draw (coverage of truth)."""
    if not tickets:
        return 0.0
    recalls = [len(set(main) & actual_main) / spec.main_k for main, _ in tickets]
    return float(np.mean(recalls))


def evaluate_forward(
    history: pd.DataFrame,
    spec: GameSpec,
    providers: list[InferenceProvider],
    *,
    budget: int = 25,
    holdout: int = 20,
    seed: int = 1234,
    jackpot: JackpotModel | None = None,
    popularity: PopularityModel | None = None,
    weights: AggregationWeights | None = None,
    main_cols: list[str] | None = None,
    include_steps: bool = False,
) -> dict:
    """Run the forward-only, equal-budget benchmark and return a JSON-serialisable summary.

    Set ``include_steps`` to expose the precommitted tickets and observed metrics for each holdout
    contest. The default summary contract stays compact for existing CLI and API consumers.
    """
    jackpot = jackpot or default_jackpot_model(spec)
    popularity = popularity or PopularityModel()

    if main_cols is None:
        main_cols = [c for c in history.columns if str(c).lower().startswith(("ball_", "n"))][
            : spec.main_k
        ]
    star_cols = [
        c
        for c in history.columns
        if str(c).lower().startswith(("star_", "dream", "lucky"))
    ][: spec.star_k]
    n = len(history)
    if n <= holdout + 5:
        raise ValueError("history too short for the requested holdout window")
    start = n - holdout

    steps: list[StepResult] = []
    for t in range(start, n):
        train = history.iloc[:t]
        actual = history.iloc[t]
        actual_main = _actual_main(actual, spec, main_cols)
        actual_star = tuple(sorted(int(actual[c]) for c in star_cols))
        raw_draw_date = actual.get("draw_date", t)
        draw_date = (
            raw_draw_date.date().isoformat()
            if isinstance(raw_draw_date, pd.Timestamp)
            else str(raw_draw_date)
        )

        envelopes: list[InferenceEnvelope] = []
        step = StepResult(
            t=t,
            draw_date=draw_date,
            actual_main=tuple(sorted(actual_main)),
            actual_star=actual_star,
        )
        for prov in providers:
            # Fit forward-only; providers that accept a spec get it.
            try:
                prov.fit(train, spec)  # type: ignore[call-arg]
            except TypeError:
                prov.fit(train)
            result = prov.propose(spec, budget, np.random.default_rng(seed + t))
            env = InferenceEnvelope.build(
                provider=prov.name, game=spec, result=result, seed=seed + t,
                training_data=train, created_utc="",
            )
            envelopes.append(env)
            step.per_provider_tickets[prov.name] = list(result.tickets)
            cov = coverage_report(spec, result.tickets)
            roi = portfolio_expected_roi(spec, result.tickets, jackpot, popularity)
            step.per_provider[prov.name] = {
                "hit_recall": _hit_recall(result.tickets, actual_main, spec),
                **cov,
                **roi,
            }

        agg_tickets = aggregate(
            envelopes, spec, budget, weights=weights, jackpot=jackpot, popularity=popularity
        )
        agg_cov = coverage_report(spec, agg_tickets)
        agg_roi = portfolio_expected_roi(spec, agg_tickets, jackpot, popularity)
        step.aggregated = {
            "hit_recall": _hit_recall(agg_tickets, actual_main, spec),
            **agg_cov,
            **agg_roi,
        }
        step.aggregated_tickets = list(agg_tickets)
        steps.append(step)

    summary = _summarise(steps, providers, budget, holdout, seed, spec)
    if include_steps:
        summary["steps"] = [_serialise_step(step) for step in steps]
    return summary


def _serialise_ticket(ticket: Ticket) -> dict[str, list[int]]:
    main, star = ticket
    return {"main": list(main), "auxiliary": list(star)}


def _serialise_step(step: StepResult) -> dict:
    agents = {
        provider: {
            "metrics": metrics,
            "tickets": [_serialise_ticket(ticket) for ticket in step.per_provider_tickets[provider]],
        }
        for provider, metrics in step.per_provider.items()
    }
    agents["coordinated_aggregation"] = {
        "metrics": step.aggregated,
        "tickets": [_serialise_ticket(ticket) for ticket in step.aggregated_tickets],
    }
    return {
        "t": step.t,
        "draw_date": step.draw_date,
        "actual": {
            "main": list(step.actual_main),
            "auxiliary": list(step.actual_star),
        },
        "agents": agents,
    }


def _mean_metric(steps: list[StepResult], who: str, key: str) -> float:
    if who == "aggregated":
        vals = [s.aggregated.get(key, np.nan) for s in steps]
    else:
        vals = [s.per_provider.get(who, {}).get(key, np.nan) for s in steps]
    return float(np.nanmean(vals)) if vals else float("nan")


def _summarise(
    steps: list[StepResult],
    providers: list[InferenceProvider],
    budget: int,
    holdout: int,
    seed: int,
    spec: GameSpec,
) -> dict:
    metric_keys = ["hit_recall", "pair_coverage", "number_coverage", "mean_jaccard_diversity",
                   "expected_roi_per_ticket", "unpopularity_lift"]
    provider_summary = {
        prov.name: {k: _mean_metric(steps, prov.name, k) for k in metric_keys}
        for prov in providers
    }
    aggregated_summary = {k: _mean_metric(steps, "aggregated", k) for k in metric_keys}

    # Headline: does aggregation beat the best single provider on the two levers we control?
    best_cov_provider = max(provider_summary, key=lambda p: provider_summary[p]["pair_coverage"])
    best_roi_provider = max(
        provider_summary, key=lambda p: provider_summary[p]["unpopularity_lift"]
    )
    return {
        "game": spec.name,
        "budget": budget,
        "holdout": holdout,
        "seed": seed,
        "providers": provider_summary,
        "aggregated": aggregated_summary,
        "headline": {
            "best_single_pair_coverage": provider_summary[best_cov_provider]["pair_coverage"],
            "best_single_pair_coverage_provider": best_cov_provider,
            "aggregated_pair_coverage": aggregated_summary["pair_coverage"],
            "coverage_improvement": aggregated_summary["pair_coverage"]
            - provider_summary[best_cov_provider]["pair_coverage"],
            "best_single_unpopularity_lift": provider_summary[best_roi_provider]["unpopularity_lift"],
            "aggregated_unpopularity_lift": aggregated_summary["unpopularity_lift"],
        },
    }
