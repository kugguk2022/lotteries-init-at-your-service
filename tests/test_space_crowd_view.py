"""Screen C (crowd forecast) view module: artifact contract and table rendering."""

import ast
import importlib.util
import json
from pathlib import Path

import pytest

SPACE = Path(__file__).resolve().parents[1] / "publishing/huggingface-space"


def _module():
    spec = importlib.util.spec_from_file_location("crowd_view", SPACE / "crowd_view.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_published_artifact_loads_and_keeps_the_null_as_reference():
    view = _module()
    payload = view.load_evidence(SPACE, "euromillions")
    assert payload is not None
    board = view.leaderboard_table(payload)
    null = board[board["Forecaster"] == "Uniform null"].iloc[0]
    assert null["Skill vs null"] == "REFERENCE"
    assert list(board["Rank"]) == sorted(board["Rank"])
    head = view.headline(payload)
    assert head["ci_low"] <= head["skill"] <= head["ci_high"]
    assert head["interval_excludes_zero"] == (head["ci_low"] > 0)


def test_profiles_without_winner_counts_report_absence_not_an_error():
    view = _module()
    assert view.load_evidence(SPACE, "nl-lotto") is None
    assert view.load_evidence(SPACE, "synthetic") is None


def test_tables_have_the_expected_shape():
    view = _module()
    payload = view.load_evidence(SPACE, "euromillions")
    assert len(view.popularity_table(payload, "main")) == 50
    assert len(view.popularity_table(payload, "stars")) == 12
    replay = view.replay_table(payload)
    assert set(replay["Series"]) == {"Fitted crowd", "Uniform null"}
    assert "Static prior" in set(view.replay_table(payload, True)["Series"])
    payout = view.payout_table(payload)
    assert payout.iloc[0]["Agent"] == "Calibrated Escape"
    assert all(value.startswith("-") for value in payout["Modeled ROI"])  # never sold as profit
    assert len(view.ticket_table(payload)) == 12


def test_artifact_publishes_derived_figures_only():
    payload = json.loads((SPACE / "data/crowd/euromillions/crowd_forecast.json").read_text())
    assert payload["evidence_kind"] == "historical_forward_replay_not_predraw_publication"
    text = json.dumps(payload)
    for raw_field in ("actual_winners", "predicted_winners", '"entries_per_draw"'):
        assert raw_field not in text


def test_wrong_game_or_truncated_artifact_is_rejected(tmp_path):
    view = _module()
    source = json.loads((SPACE / "data/crowd/euromillions/crowd_forecast.json").read_text())
    target = tmp_path / "data/crowd/nl-lotto"
    target.mkdir(parents=True)
    (target / "crowd_forecast.json").write_text(json.dumps(source))
    with pytest.raises(ValueError, match="not 'nl-lotto'"):
        view.load_evidence(tmp_path, "nl-lotto")
    del source["leaderboard"]
    target = tmp_path / "data/crowd/euromillions"
    target.mkdir(parents=True)
    (target / "crowd_forecast.json").write_text(json.dumps(source))
    with pytest.raises(ValueError, match="missing"):
        view.load_evidence(tmp_path, "euromillions")


def test_app_mounts_the_third_screen():
    tree = ast.parse((SPACE / "app.py").read_text(encoding="utf-8"))
    labels = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "Tab"
        and node.args and isinstance(node.args[0], ast.Constant)
    }
    assert "SCREEN C / CROWD FORECAST LAB" in labels
    assert {"SCREEN A / AGENT ARENA", "SCREEN B / PRE-DRAW SET LAB"} <= labels
