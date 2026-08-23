from __future__ import annotations

import pandas as pd
import pytest

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
    assert len(seen["providers"]) == 12


def test_ml_ensemble_is_excluded_when_scikit_learn_is_missing(monkeypatch):
    """``available()`` must not promise a provider that will crash once the benchmark starts.

    Regression: scikit-learn was imported inside ``fit()`` rather than checked at construction, so
    a base ``pip install lottobench`` reported ml_ensemble as available and then died mid-run with a
    bare ModuleNotFoundError on ``--all-providers``.
    """
    import sys

    from lotteries_core.providers import load_ml_ensemble

    # A ``None`` entry in sys.modules makes ``import sklearn`` raise ImportError, which is how a
    # base install behaves without the ``ml`` extra.
    monkeypatch.setitem(sys.modules, "sklearn", None)

    with pytest.raises(ImportError, match=r"lottobench\[ml\]"):
        load_ml_ensemble()()

    assert "ml_ensemble" not in registry.available()
    assert "ml_ensemble" in registry.names()
