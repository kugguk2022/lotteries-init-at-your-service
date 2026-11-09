"""Presentation and integrity checks for the fixed-budget PoI benchmark."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

LABELS = {
    "full_winner": "Full winning ticket (all mains + all stars)",
    "main_ge3": "At least 3 mains (stars ignored)",
    "main_ge4": "At least 4 mains (stars ignored)",
    "all_main": "All mains (stars ignored)",
}


def load_evidence(directory: Path) -> tuple[dict, dict, pd.DataFrame]:
    summary = json.loads((directory / "poi_benchmark_summary.json").read_text())
    for name, expected in summary["artifacts"].items():
        if Path(name).name != name:
            raise ValueError("Invalid PoI artifact path")
        path = directory / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"PoI artifact hash mismatch: {name}")
    manifest = json.loads((directory / "poi_next_manifest.json").read_text())
    replay = pd.read_csv(directory / "poi_benchmark_replay.csv")
    return summary, manifest, replay


def result_table(summary: dict, budget: int | None = None) -> pd.DataFrame:
    rows = []
    for result in summary["results"]:
        if budget is None and result["metric"] != "full_winner":
            continue
        if budget is not None and result["budget"] != int(budget):
            continue
        interval = result["lift_ci95"]
        row = ({"Tickets per draw": result["budget"],
                "Universe included (%)": round(100*result["universe_fraction"], 6)}
               if budget is None else {"Measurement": LABELS[result["metric"]]})
        row.update({
            "Draws with match / evaluated": f"{result['hits']} / {result['draws']}",
            "Random expected draws": round(result["random_expected_hits"], 6),
            "Lift / descriptive 95% interval": (
                f"{result['lift']:.3g} [{interval[0]:.3g}, {interval[1]:.3g}]"
                if interval is not None else "Unassessed"
            ),
        })
        if budget is not None:
            row.update({"Matching tickets": result["matching_tickets"],
                        "Random expected tickets": round(result["random_expected_tickets"], 4)})
        rows.append(row)
    return pd.DataFrame(rows)


def trend_table(replay: pd.DataFrame, budget: int, metric: str) -> pd.DataFrame:
    frame = replay[(replay.budget == int(budget)) & (replay.metric == metric)].sort_values("draw_date")
    return pd.DataFrame({
        "Draw": pd.to_datetime(frame.draw_date),
        "PoI cumulative draws with match": frame.draw_hit.cumsum(),
        "Random expected cumulative draws": frame.random_draw_hit_probability.cumsum(),
    }).melt(id_vars="Draw", var_name="Series", value_name="Draws with match")
