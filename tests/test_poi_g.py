from __future__ import annotations

import pandas as pd
import pytest

from lotteries_core.poi_g import generate_poi_g_subset
from lotteries_core.protocol import GameSpec
from lotteries_core.roi import JackpotModel


def _history(rows: int = 18) -> pd.DataFrame:
    values = []
    for i in range(rows):
        mains = sorted({(i + offset * 3) % 8 + 1 for offset in range(3)})
        if len(mains) < 3:
            mains = [1, 2, 3]
        values.append(
            {
                "ball_1": mains[0],
                "ball_2": mains[1],
                "ball_3": mains[2],
                "star_1": i % 3 + 1,
            }
        )
    return pd.DataFrame(values)


def test_poi_g_subset_has_requested_distinct_legal_tickets_and_provenance():
    spec = GameSpec("tiny", main_n=8, main_k=3, star_n=3, star_k=1)
    subset = generate_poi_g_subset(_history(), spec, 25, window=6)

    assert subset.size == 25
    assert len(set(subset.tickets)) == 25
    assert subset.universe_size == spec.n_tickets()
    assert subset.universe_fraction == pytest.approx(25 / spec.n_tickets())
    assert subset.reduction_factor == pytest.approx(spec.n_tickets() / 25)
    for ticket in subset.tickets:
        spec.validate_ticket(ticket)


def test_poi_g_is_deterministic_and_roi_is_only_for_selected_budget():
    spec = GameSpec("tiny", main_n=8, main_k=3, star_n=3, star_k=1)
    first = generate_poi_g_subset(_history(), spec, 30)
    second = generate_poi_g_subset(_history(), spec, 30)
    assert first.tickets == second.tickets
    assert first.scores == second.scores

    roi = first.modeled_portfolio_roi(
        spec,
        budget=5,
        jackpot=JackpotModel(jackpot=100.0, ticket_price=2.0, n_other_tickets=100.0),
    )
    assert roi["selection_budget"] == 5
    assert roi["expected_roi_per_ticket"] < 0


def test_poi_g_rejects_invalid_sizes():
    with pytest.raises(ValueError, match="positive"):
        generate_poi_g_subset(_history(), GameSpec("tiny", 8, 3, 3, 1), 0)
