"""Presentation and integrity checks for the crowd-forecast screen.

The screen reads one derived artifact, ``data/crowd/<profile>/crowd_forecast.json``, built by
``scripts/build_crowd_forecast.py --space-out``. Nothing here fits a model or needs numpy/scipy:
the Space only displays evidence that the repository produced and tested.
"""
from __future__ import annotations

import json
from math import expm1
from pathlib import Path

import pandas as pd

ARTIFACT = "crowd_forecast.json"

FORECASTER_LABELS = {
    "fitted_crowd": "Fitted crowd",
    "uniform_crowd": "Uniform null",
    "static_prior": "Static prior",
}

AGENT_LABELS = {
    "uniform_random": "Random tickets",
    "unpopularity_static_prior": "Crowd Escape",
    "calibrated_escape": "Calibrated Escape",
}

_REQUIRED = (
    "schema_version", "game", "leaderboard", "replay", "popularity", "payout",
    "history_cutoff", "first_evaluated_draw", "last_evaluated_draw", "claims_boundary",
)


def artifact_path(root: Path, profile_key: str) -> Path:
    return Path(root) / "data" / "crowd" / profile_key / ARTIFACT


def load_evidence(root: Path, profile_key: str) -> dict | None:
    """Return the validated artifact, ``None`` if this profile has none, or raise ``ValueError``."""
    path = artifact_path(root, profile_key)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"crowd-forecast artifact is not valid JSON: {error}") from error
    missing = [key for key in _REQUIRED if key not in payload]
    if missing:
        raise ValueError(f"crowd-forecast artifact is missing: {', '.join(missing)}")
    if payload["game"] != profile_key:
        raise ValueError(
            f"crowd-forecast artifact is for {payload['game']!r}, not {profile_key!r}"
        )
    names = {row["forecaster"] for row in payload["leaderboard"]}
    if "uniform_crowd" not in names:
        raise ValueError("crowd-forecast artifact has no uniform null row")
    return payload


def _label(name: str, labels: dict) -> str:
    return labels.get(name, name.replace("_", " ").title())


def leaderboard_table(payload: dict) -> pd.DataFrame:
    rows = []
    for row in sorted(payload["leaderboard"], key=lambda r: r["rank"]):
        is_null = row["forecaster"] == "uniform_crowd"
        rows.append({
            "Rank": int(row["rank"]),
            "Forecaster": _label(row["forecaster"], FORECASTER_LABELS),
            "Skill vs null": "REFERENCE" if is_null else f"{row['skill_vs_uniform']:+.2f}",
            "95% interval": "-" if is_null else (
                f"{row['skill_ci95_low']:+.2f} to {row['skill_ci95_high']:+.2f}"
            ),
            "Draws won": "-" if is_null else (
                f"{int(row['draws_better_than_null'])} / {int(row['eval_draws'])}"
            ),
            "Typical miss": f"{100 * expm1(row['big_tier_log_rmse']):.0f}%",
            "Calibration slope": "-" if is_null else (
                f"{row['calibration_slope']:.2f}"
            ),
        })
    return pd.DataFrame(rows)


def replay_table(payload: dict, include_static: bool = False) -> pd.DataFrame:
    """Long-format per-draw typical miss for the line chart."""
    frame = pd.DataFrame(payload["replay"])
    if not include_static:
        frame = frame[frame["forecaster"] != "static_prior"]
    return pd.DataFrame({
        "Draw": pd.to_datetime(frame["draw_date"]),
        "Series": frame["forecaster"].map(lambda name: _label(name, FORECASTER_LABELS)),
        "Typical miss (%)": frame["typical_miss_pct"].astype(float),
    }).sort_values(["Draw", "Series"]).reset_index(drop=True)


def popularity_table(payload: dict, pool: str = "main") -> pd.DataFrame:
    key = "main_vs_average_pct" if pool == "main" else "stars_vs_average_pct"
    values = payload["popularity"][key]
    label = "Number" if pool == "main" else "Lucky Star"
    return pd.DataFrame({
        label: [f"{i:02d}" for i in range(1, len(values) + 1)],
        "Picked vs average (%)": [float(v) for v in values],
    })


def payout_table(payload: dict) -> pd.DataFrame:
    report = payload["payout"]
    rows = []
    for name, row in report["agents"].items():
        rows.append({
            "Agent": _label(name, AGENT_LABELS),
            "Modeled ROI": f"{100 * row['mean_roi']:.1f}%",
            "Edge vs random": f"{row['alpha_pp']:+.2f} pp",
            "Edge after haircut": f"{row['alpha_pp_calibration_haircut']:+.2f} pp",
            "Lower tiers, EUR/ticket": f"{row['lower_tier_payout_per_ticket']:.2f}",
            "Jackpot, EUR/ticket": f"{row['jackpot_payout_per_ticket']:.2f}",
            "_order": row["alpha_pp"],
        })
    frame = pd.DataFrame(rows).sort_values("_order", ascending=False)
    return frame.drop(columns="_order").reset_index(drop=True)


def ticket_table(payload: dict, agent: str = "calibrated_escape") -> pd.DataFrame:
    tickets = payload["payout"]["agents"].get(agent, {}).get("tickets_committed", [])
    rows = []
    for index, text in enumerate(tickets, start=1):
        main, _, stars = text.partition(" + ")
        rows.append({"Ticket": index, "Main numbers": main, "Lucky stars": stars})
    return pd.DataFrame(rows)


def headline(payload: dict) -> dict:
    """The handful of figures the screen's summary strip shows."""
    board = {row["forecaster"]: row for row in payload["leaderboard"]}
    fitted = board.get("fitted_crowd") or min(
        (r for r in payload["leaderboard"] if r["forecaster"] != "uniform_crowd"),
        key=lambda r: r["rank"],
    )
    economics = payload["payout"]["economics"]
    return {
        "skill": float(fitted["skill_vs_uniform"]),
        "ci_low": float(fitted["skill_ci95_low"]),
        "ci_high": float(fitted["skill_ci95_high"]),
        "interval_excludes_zero": bool(fitted["skill_ci95_low"] > 0),
        "eval_draws": int(fitted["eval_draws"]),
        "history_draws": int(payload.get("history_draws", 0)),
        "first": payload["first_evaluated_draw"],
        "last": payload["last_evaluated_draw"],
        "cutoff": payload["history_cutoff"],
        "jackpot_eur": float(economics["jackpot_eur"]),
        "entries": float(economics["entries"]),
        "slope": float(payload["payout"]["out_of_sample_calibration_slope"]),
    }
