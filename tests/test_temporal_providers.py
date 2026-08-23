from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lotteries_core.protocol import GameSpec
from lotteries_core.providers.temporal import GarchMarkovBranchProvider


def _history(rows: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    return pd.DataFrame(
        [
            {
                **{f"ball_{i + 1}": value for i, value in enumerate(sorted(rng.choice(12, 4, replace=False) + 1))},
                "star_1": int(rng.integers(1, 5)),
            }
            for _ in range(rows)
        ]
    )


def test_garch_markov_branch_returns_a_reproducible_legal_budget():
    spec = GameSpec("small", 12, 4, 4, 1)
    history = _history()
    first = GarchMarkovBranchProvider().fit(history, spec).propose(
        spec, 7, np.random.default_rng(9)
    )
    second = GarchMarkovBranchProvider().fit(history, spec).propose(
        spec, 7, np.random.default_rng(9)
    )
    assert first.tickets == second.tickets
    assert len(first.tickets) == len(set(first.tickets)) == 7
    assert first.diagnostics["variance_next"] > 0
    for ticket in first.tickets:
        spec.validate_ticket(ticket)


def test_sequence_transformer_is_a_real_optional_provider():
    torch = pytest.importorskip("torch")
    del torch
    from lotteries_core.providers.temporal import SequenceTransformerProvider

    spec = GameSpec("small", 12, 4, 4, 1)
    result = SequenceTransformerProvider(epochs=1).fit(_history(), spec).propose(
        spec, 3, np.random.default_rng(7)
    )
    assert len(result.tickets) == 3
    assert result.diagnostics["epochs"] == 1
    for ticket in result.tickets:
        spec.validate_ticket(ticket)
