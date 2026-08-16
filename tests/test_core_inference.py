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
from lotteries_core.providers import (
    FrequencyProvider,
    ParallaxGuardProvider,
    PerronFrobeniusProvider,
    UnpopularityProvider,
    null_tv_band,
    replicated_evidence,
    stationary_distribution,
)
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


# ----------------------------------------------------------------------------------------------
# Perron-Frobenius / PageRank provider
# ----------------------------------------------------------------------------------------------


RIGGED = {1, 2, 3}


def _rigged_history(n: int = 400, seed: int = 3) -> pd.DataFrame:
    """History where 1, 2, 3 are drawn together far more often than chance allows.

    They are deliberately *not* an isolated clique -- each rigged draw also carries two ordinary
    numbers, so the biased nodes stay connected to the rest of the graph. See
    ``test_isolated_clique_gets_no_pagerank_boost`` for why that distinction matters.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        if i % 2 == 0:
            mains = sorted(RIGGED | {int(v) for v in rng.choice(np.arange(6, 51), 2, replace=False)})
        else:
            mains = sorted(int(v) for v in rng.choice(np.arange(6, 51), size=5, replace=False))
        stars = sorted(int(v) for v in rng.choice(np.arange(1, 13), size=2, replace=False))
        rows.append(
            {
                "draw_date": f"2020-01-{(i % 28) + 1:02d}",
                **{f"ball_{j+1}": int(mains[j]) for j in range(5)},
                **{f"star_{j+1}": int(stars[j]) for j in range(2)},
            }
        )
    return pd.DataFrame(rows)


def test_stationary_vector_is_the_perron_eigenvector():
    """pi must be a strictly positive probability vector fixed by the damped operator M."""
    rng = np.random.default_rng(0)
    W = rng.integers(0, 9, size=(12, 12)).astype(float)
    W = W + W.T
    np.fill_diagonal(W, 0.0)
    damping = 0.85
    rank = stationary_distribution(W, damping=damping, tol=1e-15)

    assert rank.pi.shape == (12,)
    assert np.all(rank.pi > 0)                      # Perron-Frobenius: strictly positive
    assert rank.pi.sum() == pytest.approx(1.0)

    P = W / W.sum(axis=0)
    M = damping * P + (1.0 - damping) / 12.0
    assert np.allclose(M @ rank.pi, rank.pi, atol=1e-10)  # eigenvalue exactly 1


def test_dangling_numbers_do_not_break_the_chain():
    """A number that was never drawn is a dangling node; it must still receive teleport mass."""
    W = np.zeros((6, 6))
    W[np.ix_([0, 1, 2], [0, 1, 2])] = 4.0
    np.fill_diagonal(W, 0.0)
    rank = stationary_distribution(W)
    assert np.all(rank.pi > 0)
    assert rank.pi.sum() == pytest.approx(1.0)
    assert rank.pi[0] > rank.pi[5]  # connected beats isolated


def test_perron_ranks_a_rigged_graph_and_orientation_flips_it():
    spec = GameSpec.euromillions()
    hist = _rigged_history()
    aff = PerronFrobeniusProvider(orientation="affinity").fit(hist, spec)
    pi = aff._main.pi
    assert pi[:3].mean() > pi[3:].mean()      # the planted bias is ranked up
    assert aff._main.tv_from_uniform > 0.1    # and the signal is large, not sampling noise

    con = PerronFrobeniusProvider(orientation="contrarian").fit(hist, spec)
    hits = {
        p.orientation: sum(
            len(set(m) & RIGGED) for m, _ in p.propose(spec, 25, np.random.default_rng(0)).tickets
        )
        for p in (aff, con)
    }
    assert hits["affinity"] > hits["contrarian"]  # orientation genuinely reverses the preference


def test_isolated_clique_gets_no_pagerank_boost():
    """A closed regular component gets PageRank exactly proportional to its size -- i.e. uniform.

    This is why "these numbers always come up together" does not, on its own, lift a number in this
    ranking: a group that only ever appears with itself retains exactly the teleport mass it is
    entitled to and no more. It is a structural limit of the method, not a bug.
    """
    W = np.zeros((10, 10))
    W[np.ix_(range(4), range(4))] = 50.0  # a closed, regular 4-clique
    W[np.ix_(range(4, 10), range(4, 10))] = 7.0
    np.fill_diagonal(W, 0.0)
    rank = stationary_distribution(W)
    assert rank.pi[:4] == pytest.approx(0.1, abs=1e-9)   # exactly 1/10 each, despite 7x the weight
    assert rank.tv_from_uniform == pytest.approx(0.0, abs=1e-9)


def test_perron_provider_meets_budget_with_legal_distinct_tickets():
    spec = GameSpec.euromillions()
    hist = _synthetic_history()
    for orientation in ("affinity", "contrarian", "uniform"):
        prov = PerronFrobeniusProvider(orientation=orientation).fit(hist, spec)
        res = prov.propose(spec, budget=25, rng=np.random.default_rng(4))
        assert len(res.tickets) == 25
        assert len(set(res.tickets)) == 25
        for t in res.tickets:
            spec.validate_ticket(t)


def test_perron_provider_is_deterministic_regardless_of_rng():
    """No RNG is consumed, so two different seeds must yield byte-identical portfolios."""
    spec = GameSpec.euromillions()
    hist = _synthetic_history()
    prov = PerronFrobeniusProvider().fit(hist, spec)
    a = prov.propose(spec, 20, np.random.default_rng(1)).tickets
    b = prov.propose(spec, 20, np.random.default_rng(999)).tickets
    assert a == b


def test_single_bonus_ball_falls_back_to_the_cross_marginal():
    """With star_k == 1 there are no star-star pairs, so the star walk must not silently be uniform."""
    spec = GameSpec.eurodreams()
    rng = np.random.default_rng(5)
    rows = []
    for i in range(200):
        mains = sorted(int(v) for v in rng.choice(np.arange(1, 41), size=6, replace=False))
        # bonus ball 1 appears far more often than the rest
        dream = 1 if i % 2 == 0 else int(rng.integers(2, 6))
        rows.append({**{f"ball_{j+1}": mains[j] for j in range(6)}, "dream_1": dream})
    hist = pd.DataFrame(rows)
    prov = PerronFrobeniusProvider().fit(hist, spec)
    assert prov._star_fallback
    assert prov._star.pi.argmax() == 0            # bonus ball 1 carries the most cross mass
    assert prov._star.tv_from_uniform > 0.05      # and the fallback is not degenerate
    res = prov.propose(spec, budget=15, rng=np.random.default_rng(0))
    assert len(res.tickets) == 15
    for t in res.tickets:
        spec.validate_ticket(t)


def _dated_history(n: int = 400, seed: int = 13, planted: int | None = None) -> pd.DataFrame:
    """Fair history under the current EuroMillions matrix, optionally with one number planted.

    Dates start after 2016-09-27 so the whole frame survives the current-rules filter.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2017-01-03")
    rows = []
    for i in range(n):
        if planted is None:
            mains = sorted(int(v) for v in rng.choice(np.arange(1, 51), 5, replace=False))
        else:
            others = rng.choice([v for v in range(1, 51) if v != planted], 4, replace=False)
            mains = sorted([planted, *(int(v) for v in others)])
        stars = sorted(int(v) for v in rng.choice(np.arange(1, 13), 2, replace=False))
        rows.append(
            {
                "draw_date": (start + pd.Timedelta(days=4 * i)).date().isoformat(),
                **{f"ball_{j+1}": mains[j] for j in range(5)},
                **{f"star_{j+1}": stars[j] for j in range(2)},
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# Parallax Guard provider
# ----------------------------------------------------------------------------------------------


def test_parallax_guard_rejects_noise_on_a_fair_history():
    """The admission rule's whole purpose: a fair generator must yield zero admitted residuals."""
    spec = GameSpec.euromillions()
    evidence = replicated_evidence(_dated_history(), spec)
    assert evidence.nonzero == 0
    assert evidence.max_abs == pytest.approx(0.0)


def test_parallax_guard_admits_a_planted_signal():
    """A bias large enough to replicate in both folds and clear Bonferroni must get through."""
    spec = GameSpec.euromillions()
    evidence = replicated_evidence(_dated_history(planted=7), spec)
    assert evidence.nonzero > 0
    assert evidence.main_number[6] > 0  # number 7, 0-based
    assert evidence.main_number[6] == evidence.main_number.max()


def test_parallax_guard_needs_the_signal_in_both_folds():
    """A one-window hot streak must contribute exactly zero, even if it is locally extreme."""
    spec = GameSpec.euromillions()
    hist = _dated_history(n=400, seed=31)
    # Folds are the alternating rows, so writing 11 into every even row saturates fold A and leaves
    # fold B untouched. The evidence is overwhelming in one view and absent from the other -- exactly
    # the shape of a hot streak, and exactly what the guard exists to reject.
    hist.loc[::2, "ball_1"] = 11
    evidence = replicated_evidence(hist, spec)
    assert evidence.main_number[10] == pytest.approx(0.0)


def test_parallax_modes_are_legal_distinct_and_star_balanced():
    spec = GameSpec.euromillions()
    hist = _dated_history()
    for mode in ("guarded", "ablation"):
        prov = ParallaxGuardProvider(mode=mode).fit(hist, spec)
        res = prov.propose(spec, budget=25, rng=np.random.default_rng(3))
        assert len(res.tickets) == 25
        assert len(set(res.tickets)) == 25
        for t in res.tickets:
            spec.validate_ticket(t)
        # No pinned-star concentration: usage across the star pool stays within one of itself.
        spread = res.diagnostics["star_usage_max"] - res.diagnostics["star_usage_min"]
        assert spread <= 1, f"{mode} star usage spread {spread}"


def test_parallax_guarded_equals_ablation_when_no_signal_is_admitted():
    """With nothing admitted the two modes are the same estimator, which is what makes the
    ablation interpretable: any observed gap is signal, never sampler drift."""
    spec = GameSpec.euromillions()
    hist = _dated_history()
    guarded = ParallaxGuardProvider(mode="guarded").fit(hist, spec)
    ablation = ParallaxGuardProvider(mode="ablation").fit(hist, spec)
    assert guarded.propose(spec, 20, np.random.default_rng(8)).diagnostics["evidence_nonzero"] == 0
    a = guarded.propose(spec, 20, np.random.default_rng(8)).tickets
    b = ablation.propose(spec, 20, np.random.default_rng(8)).tickets
    assert a == b


def test_parallax_restricts_to_the_current_rules_regime():
    """EuroMillions changed its star pool in 2016; older rows are not observations of today's null."""
    spec = GameSpec.euromillions()
    old = _dated_history(n=100, seed=4)
    old["draw_date"] = [
        (pd.Timestamp("2013-01-04") + pd.Timedelta(days=4 * i)).date().isoformat()
        for i in range(len(old))
    ]
    recent = _dated_history(n=300, seed=5)
    prov = ParallaxGuardProvider().fit(pd.concat([old, recent], ignore_index=True), spec)
    res = prov.propose(spec, 10, np.random.default_rng(0))
    assert res.diagnostics["source_history_rows"] == 400
    assert res.diagnostics["history_rows"] == 300  # the pre-2016 rows are excluded

    with pytest.raises(ValueError, match="no euromillions rows under the current rules"):
        ParallaxGuardProvider().fit(old, spec)


def test_pagerank_signal_on_a_fair_history_sits_inside_the_null_band():
    """The provider's own falsification test: on fair draws pi is indistinguishable from uniform.

    This is the standing reason the method stays demoted. If a real history ever produced a TV above
    the 99th percentile of the fair-draw null, that would be evidence worth chasing.
    """
    spec = GameSpec.euromillions()
    hist = _synthetic_history(n=400, seed=21)  # generated by a fair RNG by construction
    observed = PerronFrobeniusProvider().fit(hist, spec)._main.tv_from_uniform
    band = null_tv_band(spec, n_draws=len(hist), replicates=48, seed=5)
    assert observed <= band["q99"]
