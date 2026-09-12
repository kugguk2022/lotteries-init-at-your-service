"""Publish the Crowd Escape provider's next-draw ticket slate.

Crowd Escape models how people choose tickets, not how a fair lottery machine draws balls.  The
artifact produced here therefore exposes the provider's already-committed prospective tickets as
a transparent pre-draw research slate, together with the modeled jackpot-sharing benefit of
choosing combinations that the crowd prior considers unpopular.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .popularity import PopularityModel
from .protocol import GameSpec, Ticket
from .roi import (
    JackpotModel,
    expected_cowinners,
    expected_jackpot_payout,
    expected_roi_per_ticket,
)

CROWD_ESCAPE_FILE = "crowd_escape_forecasted_draws.csv"
CROWD_ESCAPE_SUMMARY_FILE = "crowd_escape_summary.json"


@dataclass(frozen=True)
class CrowdEscapeForecast:
    """A dedicated view of the sealed Crowd Escape prospective submissions."""

    draws: pd.DataFrame
    summary: dict


def _ticket(row: object, spec: GameSpec) -> Ticket:
    main = tuple(int(value) for value in str(getattr(row, "main_draw")).split())
    auxiliary_text = str(getattr(row, "auxiliary_draw", "")).strip()
    auxiliary = (
        ()
        if not auxiliary_text or auxiliary_text.lower() == "nan"
        else tuple(int(value) for value in auxiliary_text.split())
    )
    ticket = (main, auxiliary)
    spec.validate_ticket(ticket)
    return ticket


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_crowd_escape_forecast(
    prospective: pd.DataFrame,
    spec: GameSpec,
    *,
    output_directory: Path,
    history_cutoff: str,
    target_draw_date: str,
    snapshot_sha256: str,
    jackpot: JackpotModel,
    popularity: PopularityModel | None = None,
) -> CrowdEscapeForecast:
    """Write the provider's committed next-draw slate and an interpretation manifest."""

    required = {
        "agent",
        "rank_within_agent",
        "main_draw",
        "auxiliary_draw",
        "score_status",
        "commitment_sha256",
        "backtest_consistency_pct",
        "backtest_mean_roi_alpha_pp",
    }
    missing = required - set(prospective.columns)
    if missing:
        raise ValueError(f"prospective submissions are missing columns: {sorted(missing)}")

    selected = prospective[prospective["agent"] == "unpopularity"].copy()
    selected = selected.sort_values("rank_within_agent").reset_index(drop=True)
    if selected.empty:
        raise ValueError("prospective submissions contain no Crowd Escape tickets")
    if selected["rank_within_agent"].duplicated().any():
        raise ValueError("Crowd Escape ranks must be unique")
    if not selected["score_status"].eq("PENDING").all():
        raise ValueError("Crowd Escape forecast must be published before settlement")
    if not selected["commitment_sha256"].str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("Crowd Escape commitments must be SHA-256 digests")

    tickets = [_ticket(row, spec) for row in selected.itertuples(index=False)]
    if len(set(tickets)) != len(tickets):
        raise ValueError("Crowd Escape forecast contains duplicate tickets")

    popularity = popularity or PopularityModel()
    shares = popularity.absolute_shares(spec, tickets)
    expected_splits = [
        expected_cowinners(spec, float(share), jackpot.n_other_tickets) for share in shares
    ]
    conditional_payouts = [
        expected_jackpot_payout(spec, jackpot, float(share)) for share in shares
    ]
    jackpot_rois = [
        expected_roi_per_ticket(spec, jackpot, float(share)) * 100 for share in shares
    ]

    draws = selected[
        [
            "rank_within_agent",
            "main_draw",
            "auxiliary_draw",
            "backtest_consistency_pct",
            "backtest_mean_roi_alpha_pp",
            "score_status",
            "commitment_sha256",
        ]
    ].copy()
    draws = draws.rename(columns={"rank_within_agent": "rank"})
    draws.insert(0, "target_draw_date", target_draw_date)
    draws.insert(1, "history_cutoff", history_cutoff)
    draws["crowd_popularity_share_vs_average"] = shares
    draws["crowd_escape_lift_vs_average"] = 1.0 / shares
    draws["modeled_expected_other_jackpot_winners"] = expected_splits
    draws["modeled_payout_if_jackpot_match"] = conditional_payouts
    draws["modeled_jackpot_tier_roi_pct"] = jackpot_rois
    draws["mechanical_jackpot_odds_one_in"] = spec.n_tickets()
    draws["interpretation"] = (
        "crowd-avoidance ticket; identical fair-draw probability to every legal ticket"
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    artifact_path = output_directory / CROWD_ESCAPE_FILE
    draws.to_csv(artifact_path, index=False, lineterminator="\n")

    ticket_count = len(draws)
    summary = {
        "schema_version": "1.0.0",
        "provider": "unpopularity",
        "display_name": "Crowd Escape",
        "history_cutoff": history_cutoff,
        "target_draw_date": target_draw_date,
        "snapshot_sha256": snapshot_sha256,
        "status": "PENDING",
        "ticket_count": ticket_count,
        "universe_size": spec.n_tickets(),
        "mechanical_jackpot_coverage_pct": ticket_count / spec.n_tickets() * 100,
        "mechanical_jackpot_odds_per_ticket_one_in": spec.n_tickets(),
        "full_purchase_stake": ticket_count * jackpot.ticket_price,
        "mean_crowd_popularity_share_vs_average": float(shares.mean()),
        "best_crowd_escape_lift_vs_average": float((1.0 / shares).max()),
        "mean_modeled_payout_if_jackpot_match": float(sum(conditional_payouts) / ticket_count),
        "backtest_consistency_pct": float(selected["backtest_consistency_pct"].iloc[0]),
        "backtest_mean_roi_alpha_pp": float(selected["backtest_mean_roi_alpha_pp"].iloc[0]),
        "model_scope": (
            "Static human ticket-choice prior: calendar, low-number, lucky-number, and simple "
            "pattern biases. It forecasts crowding and possible jackpot sharing, never draw odds."
        ),
        "claims_boundary": (
            "Every legal ticket remains equally likely in a fair draw. Modeled conditional payout "
            "and jackpot-tier ROI are research estimates, not realized profit or betting advice."
        ),
        "artifacts": {CROWD_ESCAPE_FILE: _sha256(artifact_path)},
    }
    (output_directory / CROWD_ESCAPE_SUMMARY_FILE).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return CrowdEscapeForecast(draws=draws, summary=summary)
