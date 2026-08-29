"""Flatten detailed forward evaluations into contest and ticket ledgers."""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from lotteries_core.popularity import PopularityModel
from lotteries_core.protocol import GameSpec


def _as_ticket(ticket: dict) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return tuple(ticket["main"]), tuple(ticket["auxiliary"])


def build_agent_ledgers(
    history: pd.DataFrame,
    summary: dict,
    spec: GameSpec,
    snapshot_sha256: str,
    *,
    evaluation_scope: str = "synthetic_forward_replay",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create public derived ledgers without redistributing the source history."""
    popularity = PopularityModel()
    main_columns = [column for column in history if column.startswith("ball_")][: spec.main_k]
    wins: dict[str, int] = {}
    alpha_totals: dict[str, float] = {}
    contest_rows: list[dict] = []
    ticket_rows: list[dict] = []

    for contest_number, step in enumerate(summary["steps"], start=1):
        agents = step["agents"]
        house_roi = float(agents["uniform_random"]["metrics"]["expected_roi_per_ticket"])
        agent_names = list(agents)
        ranked = sorted(
            agent_names,
            key=lambda name: float(agents[name]["metrics"]["expected_roi_per_ticket"]),
            reverse=True,
        )
        contest_rank = {name: rank for rank, name in enumerate(ranked, start=1)}

        metric_frame = pd.DataFrame(
            {
                name: {
                    "roi": float(agents[name]["metrics"]["expected_roi_per_ticket"]),
                    "hit": float(agents[name]["metrics"]["hit_recall"]),
                    "uncrowded": float(agents[name]["metrics"]["unpopularity_lift"]),
                }
                for name in agent_names
            }
        ).T
        percentiles = metric_frame.rank(pct=True)
        anomaly_index = (
            0.55 * percentiles["roi"]
            + 0.25 * percentiles["hit"]
            + 0.20 * percentiles["uncrowded"]
        ) * 100

        train = history.iloc[: int(step["t"])]
        counts = train[main_columns].stack().value_counts().to_dict()
        max_count = max(counts.values()) if counts else 1
        all_tickets = [
            _as_ticket(ticket)
            for agent in agents.values()
            for ticket in agent["tickets"]
        ]
        popularity_scores = [popularity.ticket_popularity(spec, ticket) for ticket in all_tickets]
        min_popularity = min(popularity_scores)
        popularity_span = max(popularity_scores) - min_popularity or 1.0
        popularity_cursor = 0

        actual_main = set(step["actual"]["main"])
        actual_auxiliary = set(step["actual"]["auxiliary"])
        for name in agent_names:
            metrics = agents[name]["metrics"]
            roi_alpha_pp = (float(metrics["expected_roi_per_ticket"]) - house_roi) * 100
            beat_house = roi_alpha_pp > 1e-12
            wins[name] = wins.get(name, 0) + int(beat_house)
            alpha_totals[name] = alpha_totals.get(name, 0.0) + roi_alpha_pp
            tickets = agents[name]["tickets"]
            best_main_hits = max(
                (len(set(ticket["main"]) & actual_main) for ticket in tickets), default=0
            )
            best_auxiliary_hits = max(
                (len(set(ticket["auxiliary"]) & actual_auxiliary) for ticket in tickets), default=0
            )

            contest_rows.append(
                {
                    "contest_number": contest_number,
                    "draw_date": step["draw_date"],
                    "agent": name,
                    "contest_rank": contest_rank[name],
                    "roi_alpha_vs_house_pp": roi_alpha_pp,
                    "beats_house": beat_house,
                    "win_rate_vs_house_pct": wins[name] / contest_number * 100,
                    "mean_roi_alpha_vs_house_pp": alpha_totals[name] / contest_number,
                    "expected_roi_pct": float(metrics["expected_roi_per_ticket"]) * 100,
                    "pair_coverage_pct": float(metrics["pair_coverage"]) * 100,
                    "number_coverage_pct": float(metrics["number_coverage"]) * 100,
                    "hit_recall_pct": float(metrics["hit_recall"]) * 100,
                    "unpopularity_lift_x": float(metrics["unpopularity_lift"]),
                    "anomaly_index": float(anomaly_index[name]),
                    "best_main_hits": best_main_hits,
                    "best_auxiliary_hits": best_auxiliary_hits,
                    "actual_main": " ".join(str(value) for value in step["actual"]["main"]),
                    "actual_auxiliary": " ".join(
                        str(value) for value in step["actual"]["auxiliary"]
                    ),
                    "ticket_set_json": json.dumps(
                        tickets, separators=(",", ":"), sort_keys=True
                    ),
                    "evaluation_scope": evaluation_scope,
                }
            )

            for ticket_index, ticket_payload in enumerate(tickets, start=1):
                ticket = _as_ticket(ticket_payload)
                popularity_score = popularity_scores[popularity_cursor]
                popularity_cursor += 1
                crowd_escape = 1 - (
                    (popularity_score - min_popularity) / popularity_span
                )
                momentum = sum(counts.get(number, 0) / max_count for number in ticket[0]) / len(
                    ticket[0]
                )
                commitment_source = (
                    f"{snapshot_sha256}|{step['draw_date']}|{name}|{ticket_index}|{ticket}"
                )
                ticket_rows.append(
                    {
                        "contest_number": contest_number,
                        "draw_date": step["draw_date"],
                        "agent": name,
                        "contest_rank": contest_rank[name],
                        "ticket_index": ticket_index,
                        "main_draw": " ".join(str(value) for value in ticket[0]),
                        "auxiliary_draw": " ".join(str(value) for value in ticket[1]),
                        "main_hits": len(set(ticket[0]) & actual_main),
                        "auxiliary_hits": len(set(ticket[1]) & actual_auxiliary),
                        "roi_alpha_vs_house_pp": roi_alpha_pp,
                        "win_rate_vs_house_pct": wins[name] / contest_number * 100,
                        "crowd_escape_score": crowd_escape * 100,
                        "momentum_score": momentum * 100,
                        "commitment_sha256": hashlib.sha256(
                            commitment_source.encode("utf-8")
                        ).hexdigest(),
                        "score_status": "observed",
                        "evaluation_scope": evaluation_scope,
                    }
                )

    return pd.DataFrame(contest_rows), pd.DataFrame(ticket_rows)
