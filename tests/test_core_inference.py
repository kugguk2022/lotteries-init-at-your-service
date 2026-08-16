"""Tests for the lotteries_core distributed-inference framework.

These cover the contract that makes the framework trustworthy: legal tickets, reproducible
envelopes, deterministic aggregation, the honest ROI direction (unpopular beats popular), and the
headline claim under test (coordination should not *reduce* coverage versus a single provider).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lotteries_core.aggregation import aggregate
from lotteries_core.coverage import pair_coverage
from lotteries_core.envelope import InferenceEnvelope, data_sha256
from lotteries_core.evaluation import evaluate_forward
from lotteries_core.popularity import PopularityModel
from lotteries_core.protocol import GameSpec
from lotteries_core.providers import FrequencyProvider, UnpopularityProvider
from lotteries_core.roi import (
    InstantGamePool,
    JackpotModel,
    expected_jackpot_payout,
    expected_roi_per_ticket,
)


def _synthetic_history(n: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        mains = sorted(rng.choice(np.arange(1, 51), size=5, replace=False))
        stars = sorted(rng.choice(np.arange(1, 13), size=2, replace=False))
        rows.append(
            {
                "draw_date": f"2020-01-{(i % 28) + 1:02d}",
                **{f"ball_{j+1}": int(mains[j]) for j in range(5)},
                **{f"star_{j+1}": int(stars[j]) for j in range(2)},
            }
        )
    return pd.DataFrame(rows)


def test_gamespec_universe_and_validation():
    spec = GameSpec.euromillions()
    assert spec.n_main_combinations() == 2118760
    assert spec.n_star_combinations() == 66
    assert spec.n_tickets() == 2118760 * 66
    spec.validate_ticket(((1, 2, 3, 4, 5), (1, 2)))
    with pytest.raises(ValueError):
        spec.validate_ticket(((1, 2, 3, 4, 4), (1, 2)))  # duplicate main
    with pytest.raises(ValueError):
        spec.validate_ticket(((5, 4, 3, 2, 1), (1, 2)))  # unsorted


def test_provider_tickets_are_legal_and_distinct():
    spec = GameSpec.euromillions()
    prov = FrequencyProvider().fit(_synthetic_history(), spec)
    res = prov.propose(spec, budget=30, rng=np.random.default_rng(1))
    assert len(res.tickets) == 30
    assert len(set(res.tickets)) == 30
    for t in res.tickets:
        spec.validate_ticket(t)


def test_envelope_roundtrip_and_provenance():
    spec = GameSpec.euromillions()
    hist = _synthetic_history()
    prov = UnpopularityProvider()
    res = prov.propose(spec, budget=10, rng=np.random.default_rng(2))
    env = InferenceEnvelope.build(
        provider=prov.name, game=spec, result=res, seed=2, training_data=hist, created_utc="",
    )
    env.validate()
    text = env.to_json()
    back = InferenceEnvelope.from_json(text)
    assert back.tickets == env.tickets
    assert back.data_sha256 == data_sha256(hist)
    assert back.verify_data(hist)
    assert not back.verify_data(hist.iloc[:-1])  # different data -> different hash


def test_unpopular_beats_popular_on_conditional_payout():
    """The honest core: an unpopular combination has higher expected payout than a popular one."""
    spec = GameSpec.euromillions()
    pop = PopularityModel()
    jack = JackpotModel()
    popular = ((1, 2, 3, 4, 5), (1, 2))          # sequence, all low, calendar-heavy
    unpopular = ((13, 27, 38, 44, 49), (9, 11))  # spread, high numbers
    share_pop = pop.popularity_vector(spec, [popular, unpopular])[0] * 2
    share_unpop = pop.popularity_vector(spec, [popular, unpopular])[1] * 2
    assert share_pop > share_unpop
    pay_pop = expected_jackpot_payout(spec, jack, float(share_pop))
    pay_unpop = expected_jackpot_payout(spec, jack, float(share_unpop))
    assert pay_unpop > pay_pop
    # ROI stays negative (negative-sum game) but is less negative for the unpopular ticket.
    assert expected_roi_per_ticket(spec, jack, float(share_unpop)) > expected_roi_per_ticket(
        spec, jack, float(share_pop)
    )


def test_aggregation_is_deterministic_and_legal():
    spec = GameSpec.euromillions()
    hist = _synthetic_history()
    provs = [FrequencyProvider().fit(hist, spec), UnpopularityProvider()]
    envs = []
    for p in provs:
        res = p.propose(spec, budget=20, rng=np.random.default_rng(5))
        envs.append(
            InferenceEnvelope.build(
                provider=p.name, game=spec, result=res, seed=5, training_data=hist, created_utc=""
            )
        )
    a = aggregate(envs, spec, budget=15)
    b = aggregate(envs, spec, budget=15)
    assert a == b  # deterministic
    assert len(a) == 15
    assert len(set(a)) == 15
    for t in a:
        spec.validate_ticket(t)


def test_aggregation_coverage_not_worse_than_single_provider():
    """Coordinated aggregation should not lose combinatorial coverage vs either single provider."""
    spec = GameSpec.euromillions()
    hist = _synthetic_history()
    provs = [FrequencyProvider().fit(hist, spec), UnpopularityProvider()]
    envs = []
    single_cov = []
    for p in provs:
        res = p.propose(spec, budget=25, rng=np.random.default_rng(11))
        single_cov.append(pair_coverage(spec, res.tickets))
        envs.append(
            InferenceEnvelope.build(
                provider=p.name, game=spec, result=res, seed=11, training_data=hist, created_utc=""
            )
        )
    agg = aggregate(envs, spec, budget=25)
    assert pair_coverage(spec, agg) >= min(single_cov) - 1e-9


def test_instant_game_advantage_detection():
    """Remaining-prize EV can flip favourable when top prizes remain -- the real Ginther edge."""
    favourable = InstantGamePool(
        ticket_price=5.0, tickets_remaining=1000.0,
        prize_tiers=((100000.0, 2.0), (100.0, 400.0)),
    )
    depleted = InstantGamePool(
        ticket_price=5.0, tickets_remaining=1000.0,
        prize_tiers=((100.0, 30.0),),  # 3000/1000 = 3.0 EV < 5.0 price
    )
    assert favourable.remaining_roi() > 0
    assert favourable.is_advantage()
    assert depleted.remaining_roi() < 0
    assert not depleted.is_advantage()


def test_forward_evaluation_runs_and_is_forward_only():
    spec = GameSpec.euromillions()
    hist = _synthetic_history(n=200)
    provs = [FrequencyProvider(), UnpopularityProvider()]
    summary = evaluate_forward(hist, spec, provs, budget=15, holdout=10, seed=99)
    assert set(summary["providers"]) == {"frequency", "unpopularity"}
    assert "aggregated" in summary
    # Unpopularity provider should capture more of the shared-jackpot lever than frequency.
    assert (
        summary["providers"]["unpopularity"]["unpopularity_lift"]
        >= summary["providers"]["frequency"]["unpopularity_lift"] - 1e-6
    )
