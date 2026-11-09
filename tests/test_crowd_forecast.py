"""Crowd-forecast track and all-tier payout: correctness, recovery, and no-false-edge controls."""

from __future__ import annotations

from itertools import combinations
from math import comb, prod
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lotteries_core.crowd_forecast import (
    CrowdModel,
    TierTable,
    WinnerHistory,
    esp,
    fit_crowd_model,
    forward_benchmark,
    hypergeometric_tier_probability,
    load_winner_history,
    uniform_model,
)
from lotteries_core.protocol import GameSpec
from lotteries_core.providers.calibrated_escape import CalibratedEscapeProvider
from lotteries_core.tier_payout import (
    TierEconomics,
    estimate_fund_per_line,
    portfolio_report,
    ticket_expected_return,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "ledger" / "crowd" / "euromillions_tier_winners.csv"

SMALL = GameSpec("small", main_n=20, main_k=4, star_n=6, star_k=2)
SMALL_TABLE = TierTable(((4, 2), (4, 1), (3, 2), (3, 1), (2, 2), (3, 0), (2, 1), (1, 2), (2, 0)))


def _simulate(model: CrowdModel, draws: int, seed: int, entries: float = 4e6) -> WinnerHistory:
    """Poisson winner counts from a known crowd: the planted-truth control."""
    rng = np.random.default_rng(seed)
    spec = model.spec
    main = np.sort(
        np.argsort(rng.random((draws, spec.main_n)), axis=1)[:, : spec.main_k] + 1, axis=1
    )
    star = np.sort(
        np.argsort(rng.random((draws, spec.star_n)), axis=1)[:, : spec.star_k] + 1, axis=1
    )
    n = np.full(draws, entries) * rng.uniform(0.7, 1.4, size=draws)
    expected = model.expected_winners(main, star, n)
    a = np.array([t[0] for t in SMALL_TABLE.tiers])
    b = np.array([t[1] for t in SMALL_TABLE.tiers])
    winners = rng.poisson(expected[:, a, b]).astype(float)
    dates = np.array([f"2030-01-{i + 1:02d}" if i < 28 else f"2030-02-{i - 27:02d}"
                      for i in range(draws)])
    return WinnerHistory(spec, SMALL_TABLE, dates, main, star, n, winners)


def _planted() -> CrowdModel:
    rng = np.random.default_rng(5)
    theta = np.linspace(0.35, -0.35, SMALL.main_n) + rng.normal(0, 0.05, SMALL.main_n)
    phi = np.array([0.2, 0.1, 0.0, 0.0, -0.1, -0.2])
    return CrowdModel(SMALL, theta - theta.mean(), phi - phi.mean(), 0.2, name="planted")


def test_esp_matches_brute_force():
    x = np.array([[0.5, 1.5, 2.0, 0.25, 3.0]])
    got = esp(x, 3)[0]
    want = [sum(prod(c) for c in combinations(x[0], j)) for j in range(4)]
    assert np.allclose(got, want)


def test_tier_probabilities_are_a_distribution_and_uniform_is_hypergeometric():
    spec = GameSpec.euromillions()
    main, star = np.array([[3, 17, 25, 40, 50]]), np.array([[2, 11]])
    uniform = uniform_model(spec).tier_probabilities(main, star)[0]
    assert uniform.sum() == pytest.approx(1.0)
    for a in range(6):
        for b in range(3):
            assert uniform[a, b] == pytest.approx(hypergeometric_tier_probability(spec, a, b))
    skewed = _planted().tier_probabilities(np.array([[1, 2, 3, 4]]), np.array([[1, 2]]))[0]
    assert skewed.sum() == pytest.approx(1.0)
    # Popular numbers drawn -> more crowd tickets in the top tier than under a uniform crowd.
    assert skewed[4, 2] > hypergeometric_tier_probability(SMALL, 4, 2)


def test_exact_match_probability_is_product_form():
    model = CrowdModel(SMALL, _planted().theta_main, np.zeros(6), 0.0)
    drawn = np.array([[2, 5, 9, 14]])
    weights = np.exp(model.theta_main)
    total = sum(prod(c) for c in combinations(weights, 4))
    want = prod(weights[drawn[0] - 1]) / total / comb(6, 2)
    assert model.tier_probabilities(drawn, np.array([[1, 2]]))[0, 4, 2] == pytest.approx(want)


def test_planted_crowd_is_recovered_and_beats_the_null_forward_only():
    truth = _planted()
    history = _simulate(truth, draws=40, seed=11)
    fitted = fit_crowd_model(history, ridge_residual=5.0)
    assert np.corrcoef(fitted.theta_main, truth.theta_main)[0, 1] > 0.9
    forecasters = {
        "uniform_crowd": lambda _train: uniform_model(SMALL),
        "fitted_crowd": lambda train: fit_crowd_model(train, ridge_residual=5.0),
    }
    result = forward_benchmark(history, forecasters, min_train=28)
    top = result["leaderboard"][0]
    assert top["forecaster"] == "fitted_crowd"
    assert top["skill_ci95_low"] > 0.3
    assert 0.7 < top["calibration_slope"] < 1.3
    assert result["first_evaluated_draw"] > str(history.dates[27])


def test_no_false_edge_when_the_crowd_really_is_uniform():
    history = _simulate(uniform_model(SMALL), draws=40, seed=3)
    forecasters = {
        "uniform_crowd": lambda _train: uniform_model(SMALL),
        "fitted_crowd": lambda train: fit_crowd_model(train),
    }
    row = forward_benchmark(history, forecasters, min_train=28)["leaderboard"]
    fitted = next(r for r in row if r["forecaster"] == "fitted_crowd")
    # The model must not manufacture skill from Poisson noise.
    assert fitted["skill_vs_uniform"] < 0.05


def test_loader_reads_fixture_and_rejects_implausible_rows(tmp_path):
    history = load_winner_history(FIXTURE)
    assert len(history) >= 40
    assert list(history.dates) == sorted(history.dates)
    frame = pd.read_csv(FIXTURE)
    frame.loc[0, "w_2_0"] = frame.loc[0, "w_2_0"] * 10  # a transcription slip
    bad = tmp_path / "bad.csv"
    frame.to_csv(bad, index=False)
    with pytest.raises(ValueError, match="implausible"):
        load_winner_history(bad)


def test_uniform_crowd_payout_matches_closed_form():
    spec = GameSpec.euromillions()
    entries, jackpot = 25e6, 100e6
    economics = TierEconomics.euromillions(entries=entries, jackpot=jackpot)
    result = ticket_expected_return(
        uniform_model(spec), economics, ((5, 12, 23, 34, 45), (3, 9)), n_samples=50
    )
    lower = 0.0
    for tier, share in economics.tier_shares.items():
        if tier == (5, 2):
            continue
        lam = entries * hypergeometric_tier_probability(spec, *tier)
        lower += share * economics.fund_per_line * (1 - np.exp(-lam))
    lam_j = entries / spec.n_tickets()
    top = jackpot / spec.n_tickets() * (1 - np.exp(-lam_j)) / lam_j
    assert result["expected_payout"] == pytest.approx(lower + top, rel=1e-6)
    assert sum(result["by_tier"].values()) == pytest.approx(result["expected_payout"])


def test_unpopular_ticket_is_paid_more_at_the_same_odds():
    spec = GameSpec.euromillions()
    theta = np.linspace(0.2, -0.2, 50)
    phi = np.linspace(0.1, -0.1, 12)
    crowd = CrowdModel(spec, theta, phi, 0.2)
    economics = TierEconomics.euromillions(entries=25e6, jackpot=100e6)
    popular = ticket_expected_return(crowd, economics, ((1, 2, 4, 7, 9), (1, 2)))
    unpopular = ticket_expected_return(crowd, economics, ((41, 44, 46, 48, 50), (11, 12)))
    assert unpopular["roi"] > popular["roi"]
    # The edge is not only the jackpot: lower tiers pay more too.
    lower = lambda r: r["expected_payout"] - r["by_tier"][(5, 2)]  # noqa: E731
    assert lower(unpopular) > lower(popular)
    report = portfolio_report(
        crowd, economics, [((41, 44, 46, 48, 50), (11, 12))],
        baseline={"mean_roi": popular["roi"]},
    )
    assert report["alpha_pp"] > 0
    assert report["mean_roi"] < 0  # still a negative-sum game


def test_fund_per_line_estimator_round_trips():
    assert estimate_fund_per_line(4.0, 1_000_000, 0.1659, 22_000_000) == pytest.approx(
        4.0 * 1_000_000 / 0.1659 / 22_000_000
    )


def test_calibrated_escape_returns_distinct_legal_low_crowd_tickets():
    spec = GameSpec.euromillions()
    theta = np.linspace(0.2, -0.2, 50)
    crowd = CrowdModel(spec, theta, np.linspace(0.1, -0.1, 12), 0.0, name="toy")
    provider = CalibratedEscapeProvider(crowd, oversample=60)
    first = provider.propose(spec, 12, np.random.default_rng(4))
    again = provider.propose(spec, 12, np.random.default_rng(4))
    assert first.tickets == again.tickets
    assert len(set(first.tickets)) == 12
    for ticket in first.tickets:
        spec.validate_ticket(ticket)
    for i, a in enumerate(first.tickets):
        for b in first.tickets[i + 1 :]:
            assert len(set(a[0]) & set(b[0])) <= 2
    mean_crowding = np.mean([crowd.ticket_log_crowding(*t) for t in first.tickets])
    assert mean_crowding < -0.3
