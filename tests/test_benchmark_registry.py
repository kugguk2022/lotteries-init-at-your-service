from __future__ import annotations

import pandas as pd

from lotteries_core import benchmark, registry


def test_all_providers_comes_from_the_distinct_entry_registry(monkeypatch, tmp_path):
    history = tmp_path / "history.csv"
    pd.DataFrame({"ball_1": [1]}).to_csv(history, index=False)
    seen: dict[str, object] = {}

    monkeypatch.setattr(registry, "available", registry.names)
    monkeypatch.setattr(registry, "create", lambda name: name)

    def fake_evaluate(frame, spec, providers, **kwargs):
        seen["providers"] = providers
        return {"providers": providers}

    monkeypatch.setattr(benchmark, "evaluate_forward", fake_evaluate)
    benchmark.main(["--history", str(history), "--all-providers"])

    assert seen["providers"] == registry.names()
    assert len(seen["providers"]) == 11
