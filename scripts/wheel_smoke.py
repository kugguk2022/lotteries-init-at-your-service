"""Smoke test executed from outside the repository after installing the built base wheel."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lotteries_core import InferenceEnvelope, portfolio_roi_trace, registry
from lotteries_core.protocol import GameSpec


def main() -> None:
    assert len(registry.names()) == 12
    assert "ml_ensemble" not in registry.available()
    for name, spec in registry.PROVIDERS.items():
        if not spec.optional:
            assert registry.create(name).name == name

    history = pd.DataFrame(
        {
            "ball_1": [1, 2],
            "ball_2": [8, 9],
            "ball_3": [19, 20],
            "ball_4": [34, 35],
            "ball_5": [47, 48],
            "star_1": [2, 3],
            "star_2": [9, 10],
        }
    )
    game = GameSpec.euromillions()
    provider = registry.create("frequency").fit(history, game)
    result = provider.propose(game, 2, np.random.default_rng(5))
    envelope = InferenceEnvelope.build(
        provider=provider.name,
        game=game,
        result=result,
        seed=5,
        training_data=history,
    )
    envelope.validate()
    assert envelope.verify_data(history)
    assert len(portfolio_roi_trace(game, result.tickets)["trace_sha256"]) == 64
    print("fresh-wheel smoke test passed: 12 providers registered; provenance and ROI trace verified")


if __name__ == "__main__":
    main()
