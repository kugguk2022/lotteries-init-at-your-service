from __future__ import annotations

import re

import numpy as np
import pandas as pd

import lottobench
from lotteries_core import InferenceEnvelope


def test_alpha_version_and_public_identity():
    from lotteries_core import __version__

    assert re.fullmatch(r"0\.1\.0a\d+", __version__)
    assert lottobench.game("euromillions").spec.name == "euromillions"
    assert "frequency" in lottobench.names()
    assert "gingerm" in lottobench.names()


def test_public_provider_builds_a_provenance_envelope():
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
    spec = lottobench.game("euromillions").spec
    provider = lottobench.create("frequency").fit(history, spec)
    result = provider.propose(spec, budget=3, rng=np.random.default_rng(7))
    envelope = InferenceEnvelope.build(
        provider=provider.name,
        game=spec,
        result=result,
        seed=7,
        training_data=history,
        created_utc="2026-08-23T00:00:00+00:00",
    )

    envelope.validate()
    assert envelope.verify_data(history)
    assert not envelope.verify_data(history.assign(ball_1=[2, 2, 3]))
