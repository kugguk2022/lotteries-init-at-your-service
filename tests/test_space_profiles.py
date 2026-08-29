from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "publishing" / "huggingface-space" / "data" / "profiles"
EXPECTED_AGENTS = {
    "uniform_random",
    "frequency",
    "unpopularity",
    "gingerm",
    "spectral_contrarian",
    "parallax",
    "coordinated_aggregation",
}


def _manifest(key: str) -> dict:
    return json.loads((PROFILES / key / "manifest.json").read_text(encoding="utf-8"))


def test_public_space_has_three_distinct_provenance_profiles():
    index = json.loads((PROFILES / "index.json").read_text(encoding="utf-8"))
    assert [profile["key"] for profile in index["profiles"]] == [
        "synthetic",
        "euromillions",
        "nl-lotto",
    ]
    assert _manifest("synthetic")["data_kind"] == "deterministic_synthetic"
    assert _manifest("euromillions")["data_kind"] == "observed_public_history"
    assert _manifest("nl-lotto")["data_kind"] == "observed_public_history"
    assert _manifest("euromillions")["game"] == {
        "name": "euromillions",
        "main_n": 50,
        "main_k": 5,
        "auxiliary_n": 12,
        "auxiliary_k": 2,
    }
    assert _manifest("synthetic")["game"] == {
        "name": "euromillions-synthetic-control",
        "main_n": 50,
        "main_k": 5,
        "auxiliary_n": 12,
        "auxiliary_k": 2,
    }
    assert _manifest("nl-lotto")["game"] == {
        "name": "nl-lotto",
        "main_n": 45,
        "main_k": 6,
        "auxiliary_n": 0,
        "auxiliary_k": 0,
    }


def test_every_profile_uses_the_same_complete_scoring_contract():
    contracts = []
    for key in ("synthetic", "euromillions", "nl-lotto"):
        manifest = _manifest(key)
        evaluation = manifest["evaluation"]
        contracts.append(
            (
                evaluation["protocol"],
                evaluation["budget_per_agent"],
                evaluation["holdout_contests"],
                tuple(evaluation["providers"]),
                evaluation["coordinator"],
                evaluation["house_reference"],
            )
        )
        contests = pd.read_csv(PROFILES / key / "contests.csv")
        tickets = pd.read_csv(PROFILES / key / "tickets.csv")
        prospective = pd.read_csv(PROFILES / key / "prospective.csv")
        assert set(contests["agent"]) == EXPECTED_AGENTS
        assert contests.groupby("contest_number")["agent"].nunique().eq(7).all()
        assert tickets.groupby(["contest_number", "agent"]).size().eq(12).all()
        assert prospective.groupby("agent").size().eq(12).all()
        assert prospective["score_status"].eq("PENDING").all()
        assert prospective["commitment_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert len(set(contracts)) == 1


def test_observed_profiles_publish_derived_outputs_not_raw_histories():
    for key in ("euromillions", "nl-lotto"):
        manifest = _manifest(key)
        assert manifest["source"]["raw_history_published"] is False
        assert not (PROFILES / key / "history.csv").exists()
        assert len(manifest["history"]["snapshot_sha256"]) == 64
