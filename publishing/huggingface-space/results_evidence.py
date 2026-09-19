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


def range_summary_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Range": row["range_id"].replace("_", " "),
        "Full winner": f"{row['full_winner_hits']}/{row['draws']}",
        "Hit rate": f"{row['hit_rate']:.1%}",
        "Equal-size random expected hits": round(row["fair_expected_hits"], 2),
        "Containment lift": (f"{row['containment_lift']:.3f}×"
                             if row["containment_lift"] is not None else "N/A"),
        "Second prize or better": f"{row['second_or_better_hits']}/{row['second_or_better_draws']}",
        "Random expected second or better": round(row["fair_second_or_better_expected_hits"], 2),
        "Cash ROI": "Not yet measured",
    } for row in rows])


def range_pending_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Range": row["range_id"].replace("_", " "),
        "Inclusive G ranges": " ∪ ".join(f"[{a:.3f}, {b:.3f}]" for a, b in row["intervals"]),
        "All tickets in range": row["candidate_count"],
        "Legal universe included": f"{row['fair_containment_probability']:.2%}",
        "Cost to buy entire range": f"€{row['full_purchase_stake_eur']:,.2f}",
    } for row in rows])


def range_replay_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "Draw": frame["draw_date"], "Range": frame["range_id"],
        "Winning mains": frame["actual_main"], "Winning stars": frame["actual_auxiliary"].fillna("—"),
        "Winner's G": frame["actual_g_score"], "Full winner inside": frame["full_winner_in_range"],
        "Second-prize tickets inside": frame["second_prize_winning_tickets_in_range"],
        "Second or better": frame["second_or_better_hit"],
        "Tickets in range": frame["candidate_count"],
        "Equal-size random full-winner chance": frame["fair_containment_probability"].map(lambda v: f"{v:.2%}"),
        "Full-range stake EUR": frame["full_purchase_stake_eur"],
    })
