from __future__ import annotations

import re

import numpy as np
import pandas as pd

from lotteries_core import InferenceEnvelope, JackpotModel, portfolio_roi_trace, registry
from lotteries_core.protocol import GameSpec


def test_alpha_version_and_complete_public_registry():
    from lotteries_core import __version__

    assert re.fullmatch(r"0\.1\.0a\d+", __version__)
    assert len(registry.names()) == 12
    assert len(set(registry.names())) == 12
    assert {"gingerm", "claude_inference", "parallax", "ml_ensemble"} <= set(registry.names())


def test_roi_trace_exposes_assumptions_intermediates_and_integrity_stamp():
    spec = GameSpec.euromillions()
    tickets = [
        ((1, 8, 19, 34, 47), (3, 11)),
        ((7, 22, 37, 44, 50), (2, 9)),
    ]
    trace = portfolio_roi_trace(
        spec,
        tickets,
        JackpotModel(jackpot=75_000_000, ticket_price=2.5, n_other_tickets=40_000_000),
    )

    assert trace["schema_version"] == 1
    assert trace["game"]["combination_count"] == spec.n_tickets()
    assert trace["assumptions"]["scope"] == "jackpot-tier-only"
    assert trace["assumptions"]["cowinner_model"] == "poisson"
    assert len(trace["tickets"]) == 2
    assert all(row["expected_cowinners"] >= 0 for row in trace["tickets"])
    assert all(row["expected_roi"] < 0 for row in trace["tickets"])
    assert len(trace["trace_sha256"]) == 64


def test_envelope_and_roi_trace_bind_proposal_to_training_data():
    history = pd.DataFrame(
        {
            "ball_1": [1, 2, 3],
            "ball_2": [8, 9, 10],
            "ball_3": [19, 20, 21],
            "ball_4": [34, 35, 36],
            "ball_5": [47, 48, 49],
            "star_1": [2, 3, 4],
            "star_2": [9, 10, 11],
        }
    )
    spec = GameSpec.euromillions()
    provider = registry.create("frequency").fit(history, spec)
    result = provider.propose(spec, budget=3, rng=np.random.default_rng(7))
    envelope = InferenceEnvelope.build(
        provider=provider.name,
        game=spec,
        result=result,
        seed=7,
        training_data=history,
        created_utc="2026-08-22T00:00:00+00:00",
    )

    envelope.validate()
    assert envelope.verify_data(history)
    assert not envelope.verify_data(history.assign(ball_1=[2, 2, 3]))
    assert len(envelope.data_sha256) == 64
    assert len(portfolio_roi_trace(spec, result.tickets)["trace_sha256"]) == 64
