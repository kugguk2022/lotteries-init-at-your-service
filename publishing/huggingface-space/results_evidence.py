"""Display-only metric contracts; never infer cash returns from jackpot-only EV."""

from __future__ import annotations

import pandas as pd


def containment_table(tickets: pd.DataFrame, main_k: int, auxiliary_k: int) -> pd.DataFrame:
    """Count contests, not tickets; both parts of a hit must occur on ONE ticket."""
    rows = []
    for agent, group in tickets.groupby("agent", sort=False):
        main = group["main_hits"].astype(int)
        auxiliary = group["auxiliary_hits"].astype(int)
        count = group["contest_number"].nunique()
        full = group.loc[(main == main_k) & (auxiliary == auxiliary_k)]
        main_only = group.loc[main == main_k]
        three = group.loc[main >= min(3, main_k)]
        rows.append({
            "Agent": agent,
            "Replay draws": count,
            "Tickets / draw": f"{group.groupby('contest_number').size().min()}–"
                              f"{group.groupby('contest_number').size().max()}",
            "Exact full-ticket containment": f"{full['contest_number'].nunique()}/{count}",
            "Exact main-set containment": f"{main_only['contest_number'].nunique()}/{count}",
            "At least 3 mains on one ticket": f"{three['contest_number'].nunique()}/{count}",
            "Settled cash ROI": "UNASSESSED — payout/stake ledger required",
        })
    return pd.DataFrame(rows)


def hybrid_replay_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Do not reinterpret exact artifact position as confidence or inverse rank."""
    if frame.empty:
        return pd.DataFrame()
    result = pd.DataFrame({
        "Draw": frame["draw_date"],
        "Actual G": frame["actual_g_score"],
        "GARCH target": frame["garch_target_g"].round(3),
        "Transformer target": frame["transformer_target_g"].round(3),
        "Artifact position (1 first)": frame["actual_rank"].astype(int),
        "Full winner in candidate set": frame["hybrid_contains_winner"],
        "In equal-size uniform control": frame["matched_uniform_contains_winner"],
    })
    if {"score_band_rank_first", "score_band_rank_last"}.issubset(frame.columns):
        result["Equal-score band (not confidence)"] = [
            f"{int(first):,}–{int(last):,}"
            for first, last in zip(frame["score_band_rank_first"], frame["score_band_rank_last"])
        ]
    else:
        result["Equal-score band (not confidence)"] = "Legacy artifact — bounds not recorded"
    return result
