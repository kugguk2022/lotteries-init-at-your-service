import importlib.util
from pathlib import Path

import pandas as pd


def _module():
    path = Path(__file__).resolve().parents[1] / "publishing/huggingface-space/results_evidence.py"
    spec = importlib.util.spec_from_file_location("results_evidence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_containment_requires_mains_and_stars_on_same_ticket():
    frame = pd.DataFrame([
        {"agent": "a", "contest_number": 1, "main_hits": 5, "auxiliary_hits": 0},
        {"agent": "a", "contest_number": 1, "main_hits": 0, "auxiliary_hits": 2},
        {"agent": "a", "contest_number": 2, "main_hits": 5, "auxiliary_hits": 2},
        {"agent": "a", "contest_number": 2, "main_hits": 5, "auxiliary_hits": 2},
    ])
    row = _module().containment_table(frame, 5, 2).iloc[0]
    assert row["Exact full-ticket containment"] == "1/2"
    assert row["Exact main-set containment"] == "2/2"
    assert row["Settled cash ROI"].startswith("UNASSESSED")
