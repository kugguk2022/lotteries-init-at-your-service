from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from bs4 import BeautifulSoup

from euromillions import garchx as base_garchx
from euromillions import garchx_alternative_volatility as alt_garchx
from euromillions.arithmetic_branch import build_branch_frame
from euromillions.diagnostics3 import (
    annotate_match_statistics,
    build_pair_z_diagnostics,
)
from euromillions.garchx_residuals import fit_residual_distribution
from euromillions_agent.phase2_sobol import generate_sobol_tickets

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY = REPO_ROOT / "data" / "euromillions.csv"
DEFAULT_POI = REPO_ROOT / "outputs" / "euromillions" / "features" / "poi.csv"
DEFAULT_OUTPUTS_ROOT = REPO_ROOT / "outputs" / "euromillions"
DEFAULT_OUT_DIR = DEFAULT_OUTPUTS_ROOT / "garch_return_benchmark"
DEFAULT_PRIZE_CACHE = DEFAULT_OUT_DIR / "prize_cache_by_date.json"
DEFAULT_TOP_N = 25
DEFAULT_HOLDOUT = 24
DEFAULT_POOL_SIZE = 10000
DEFAULT_TICKET_COST = 2.5
DEFAULT_REQUEST_SLEEP = 0.15
DEFAULT_TIMEOUT_SECONDS = 20.0
LOTTERY_RESULTS_URL = "https://www.lottery.co.uk/euromillions/results-{date_slug}"
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/136.0.0.0 Safari/537.36"
    )
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    summary_json: str


TOP_GARCH_MODELS = (
    ModelSpec("garchx", "base", "garchx/garchx_summary.json"),
    ModelSpec(
        "garchx_alternative_volatility",
        "alternative",
        "garchx_alternative_volatility/garchx_summary.json",
    ),
    ModelSpec(
        "garchx_alternative_volatility_v2",
        "alternative",
        "garchx_alternative_volatility_v2/garchx_summary.json",
    ),
)


@dataclass(frozen=True)
class ResolvedModelSpec:
    name: str
    kind: str
    start_date: str | None
    fourier_order: int
    max_garch_order: int
    floor: float
    residual_model: str


@dataclass
class MethodSummary:
    steps: int
    total_ticket_cost: float
    total_return: float
    net_profit: float
    roi: float
    mean_return_per_draw: float
    mean_profit_per_draw: float
    mean_best_ball_hits: float
    mean_best_star_hits: float
    mean_best_ticket_payout: float
    any_payout_draw_rate: float
    break_even_draw_rate: float
    exact_main5_accuracy: float
    exact_5plus2_accuracy: float
    mean_target_gap: float
    mean_abs_poi_error: float
    mean_shortlist_size: float


@dataclass
class PrizeTableRecord:
    draw_date: str
    source_url: str
    prizes: dict[tuple[int, int], float]


@dataclass
class WalkForwardContext:
    spec: ResolvedModelSpec
    frame: pd.DataFrame
    weekday_flag: str


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward benchmark of top GARCH-family EuroMillions approaches on realized shortlist returns."
    )
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--poi", type=Path, default=DEFAULT_POI)
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prize-cache", type=Path, default=DEFAULT_PRIZE_CACHE)
    parser.add_argument("--holdout", type=int, default=DEFAULT_HOLDOUT)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--ticket-cost", type=float, default=DEFAULT_TICKET_COST)
    parser.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE)
    parser.add_argument(
        "--pool-path",
        type=Path,
        default=None,
        help="Optional CSV of candidate tickets. If missing, a Sobol pool is generated.",
    )
    parser.add_argument(
        "--request-sleep",
        type=float,
        default=DEFAULT_REQUEST_SLEEP,
        help="Pause between remote prize-page requests to avoid hammering the site.",
    )
    return parser.parse_args()


def parse_money(value: str) -> float:
    cleaned = re.sub(r"[^0-9.]+", "", str(value))
    if not cleaned:
        return 0.0
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def parse_match_category(text: str) -> tuple[int, int] | None:
    match = re.search(
        r"Match\s+(\d)(?:\s+and\s+(\d)\s+Star[s]?)?$",
        str(text).strip(),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return int(match.group(1)), int(match.group(2) or 0)


def format_results_url(draw_date: str) -> str:
    stamp = pd.Timestamp(draw_date)
    return LOTTERY_RESULTS_URL.format(date_slug=stamp.strftime("%d-%m-%Y"))


def scrape_prize_table_for_date(draw_date: str, *, timeout_seconds: float) -> PrizeTableRecord:
    url = format_results_url(draw_date)
    response = requests.get(url, headers=HTTP_HEADERS, timeout=timeout_seconds)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    prize_table: dict[tuple[int, int], float] = {}
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if not rows:
            continue
        header = [cell.get_text(" ", strip=True) for cell in rows[0].find_all(["th", "td"])]
        if "Category" not in header or "Prize Per Winner" not in header:
            continue
        category_idx = header.index("Category")
        prize_idx = header.index("Prize Per Winner")
        for tr in rows[1:]:
            cols = [cell.get_text(" ", strip=True) for cell in tr.find_all(["th", "td"])]
            if len(cols) <= max(category_idx, prize_idx):
                continue
            hits = parse_match_category(cols[category_idx])
            if hits is None:
                continue
            prize_table[hits] = float(parse_money(cols[prize_idx]))
        if prize_table:
            break

    required_keys = [
        (5, 2),
        (5, 1),
        (5, 0),
        (4, 2),
        (4, 1),
        (3, 2),
        (4, 0),
        (2, 2),
        (3, 1),
        (3, 0),
        (1, 2),
        (2, 1),
        (2, 0),
    ]
    missing = [key for key in required_keys if key not in prize_table]
    if missing:
        raise RuntimeError(f"Prize table incomplete for {draw_date}: missing {missing}")

    return PrizeTableRecord(draw_date=draw_date, source_url=url, prizes=prize_table)


def load_cached_prize_records(cache_path: Path) -> dict[str, PrizeTableRecord]:
    if not cache_path.exists():
        return {}
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    steps = raw.get("steps", [])
    out: dict[str, PrizeTableRecord] = {}
    for entry in steps:
        draw_date = str(entry["draw_date"])
        prizes = {
            tuple(map(int, str(key).split("_"))): float(value)
            for key, value in entry.get("prizes", {}).items()
        }
        out[draw_date] = PrizeTableRecord(
            draw_date=draw_date,
            source_url=str(entry.get("source_url", "")),
            prizes=prizes,
        )
    return out


def write_cached_prize_records(cache_path: Path, records: dict[str, PrizeTableRecord]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    steps = []
    for draw_date in sorted(records):
        record = records[draw_date]
        steps.append(
            {
                "draw_date": record.draw_date,
                "source_url": record.source_url,
                "prizes": {f"{k[0]}_{k[1]}": v for k, v in sorted(record.prizes.items())},
            }
        )
    cache_path.write_text(json.dumps({"steps": steps}, indent=2), encoding="utf-8")


def load_or_fetch_prize_tables(
    draw_dates: list[str],
    *,
    cache_path: Path,
    request_sleep: float,
) -> dict[str, PrizeTableRecord]:
    cached = load_cached_prize_records(cache_path)
    missing = [draw_date for draw_date in draw_dates if draw_date not in cached]
    for index, draw_date in enumerate(missing):
        cached[draw_date] = scrape_prize_table_for_date(
            draw_date,
            timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
        )
        write_cached_prize_records(cache_path, cached)
        if index < len(missing) - 1 and request_sleep > 0.0:
            time.sleep(request_sleep)
    return {draw_date: cached[draw_date] for draw_date in draw_dates}


def load_resolved_model_specs(outputs_root: Path) -> tuple[ResolvedModelSpec, ...]:
    resolved: list[ResolvedModelSpec] = []
    for spec in TOP_GARCH_MODELS:
        summary_path = outputs_root / spec.summary_json
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        start_date = summary.get("cutoff_start_date") or summary.get("start_date")
        resolved.append(
            ResolvedModelSpec(
                name=spec.name,
                kind=spec.kind,
                start_date=str(start_date) if start_date else None,
                fourier_order=int(summary.get("fourier_order", 2)),
                max_garch_order=int(summary.get("max_garch_order", 2)),
                floor=float(summary.get("floor", 1e-2)),
                residual_model=str(summary.get("residual_model", "student_t_mixture")),
            )
        )
    return tuple(resolved)


def build_walk_forward_contexts(
    *,
    history_path: Path,
    poi_path: Path,
    outputs_root: Path,
) -> dict[str, WalkForwardContext]:
    contexts: dict[str, WalkForwardContext] = {}
    for spec in load_resolved_model_specs(outputs_root):
        if spec.kind == "base":
            frame, weekday_flag = base_garchx.load_model_frame(
                history_path,
                poi_path,
                start_date=spec.start_date,
            )
        else:
            frame, weekday_flag = alt_garchx.load_model_frame(
                history_path,
                poi_path,
                start_date=spec.start_date,
            )
        contexts[spec.name] = WalkForwardContext(spec=spec, frame=frame, weekday_flag=weekday_flag)
    return contexts


def common_evaluation_dates(contexts: dict[str, WalkForwardContext], holdout: int) -> list[str]:
    date_sets = [
        set(context.frame["date"].dt.strftime("%Y-%m-%d"))
        for context in contexts.values()
    ]
    common = sorted(set.intersection(*date_sets))
    if not common:
        raise ValueError("No common evaluation dates across the selected GARCH models.")
    return common[-min(len(common), int(holdout)) :]


def generate_or_load_candidate_pool(
    *,
    pool_path: Path,
    pool_size: int,
) -> pd.DataFrame:
    if pool_path.exists():
        df = pd.read_csv(pool_path, header=None)
    else:
        tickets = generate_sobol_tickets(
            n_tickets=int(pool_size),
            main_n=50,
            main_k=5,
            star_n=12,
            star_k=2,
            oversample=2,
            seed=42,
            max_shared_main=5,
            max_shared_star=2,
        )
        rows: list[list[int]] = []
        for main, stars in tickets:
            if stars is None:
                raise ValueError("Expected full 5+2 tickets from the Sobol generator.")
            rows.append([*main, *stars])
        df = pd.DataFrame(rows)
        pool_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(pool_path, index=False, header=False)

    if df.shape[1] != 7:
        raise ValueError(f"Candidate pool must have 7 integer columns, got shape={df.shape}.")

    df = df.astype(int)
    df.columns = [f"ball_{idx}" for idx in range(1, 6)] + ["star_1", "star_2"]
    df.insert(0, "combination_index", np.arange(1, len(df) + 1, dtype=int))
    return df


def score_batch(pair_counts: np.ndarray, combos: np.ndarray) -> np.ndarray:
    return (
        pair_counts[combos[:, 0], combos[:, 1]]
        + pair_counts[combos[:, 0], combos[:, 2]]
        + pair_counts[combos[:, 0], combos[:, 3]]
        + pair_counts[combos[:, 0], combos[:, 4]]
        + pair_counts[combos[:, 1], combos[:, 2]]
        + pair_counts[combos[:, 1], combos[:, 3]]
        + pair_counts[combos[:, 1], combos[:, 4]]
        + pair_counts[combos[:, 2], combos[:, 3]]
        + pair_counts[combos[:, 2], combos[:, 4]]
        + pair_counts[combos[:, 3], combos[:, 4]]
    ).astype(int)


def score_ticket_pool(
    candidate_pool: pd.DataFrame,
    *,
    pair_counts: np.ndarray,
    main_n: int,
) -> np.ndarray:
    mains = candidate_pool[[f"ball_{idx}" for idx in range(1, 6)]].to_numpy(dtype=int)
    stars = candidate_pool[["star_1", "star_2"]].to_numpy(dtype=int)
    main_scores = score_batch(pair_counts, mains)
    star1_enc = main_n + stars[:, 0]
    star2_enc = main_n + stars[:, 1]
    star_scores = pair_counts[star1_enc, star2_enc].astype(int)
    main_star_sum_1 = pair_counts[mains, star1_enc[:, None]].sum(axis=1).astype(int)
    main_star_sum_2 = pair_counts[mains, star2_enc[:, None]].sum(axis=1).astype(int)
    return (main_scores + star_scores + main_star_sum_1 + main_star_sum_2).astype(int)


def build_garch_shortlist(
    candidate_pool: pd.DataFrame,
    *,
    pair_counts: np.ndarray,
    main_n: int,
    pair_diag,
    target_score: int,
    top_n: int,
) -> tuple[pd.DataFrame, int]:
    scored = candidate_pool.copy()
    scored["score"] = score_ticket_pool(scored, pair_counts=pair_counts, main_n=main_n)
    scored["score_gap"] = (scored["score"] - int(target_score)).abs().astype(int)
    best_gap = int(scored["score_gap"].min())
    scored = annotate_match_statistics(scored, pair_diag)
    shortlist = (
        scored.sort_values(
            ["score_gap", "ticket_pair_sum_zscore", "ticket_pair_mean_z", "min_ticket_pair_z"],
            ascending=[True, False, False, False],
        )
        .head(int(top_n))
        .reset_index(drop=True)
    )
    shortlist.insert(0, "rank", np.arange(1, len(shortlist) + 1, dtype=int))
    return shortlist, best_gap


def build_base_next_design_row(next_row: pd.Series, weekday_flag: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "t": [int(next_row["t"])],
            weekday_flag: [float(next_row[weekday_flag])],
        }
    )


def next_sigma2_base(
    params_hat: np.ndarray,
    *,
    eps: np.ndarray,
    sigma2: np.ndarray,
    next_fourier_terms: np.ndarray,
    floor: float,
    best_order: int,
) -> float:
    alpha = float(params_hat[1])
    beta_end = 2 + best_order
    betas = np.asarray(params_hat[2:beta_end], dtype=float)
    gamma_hat = np.asarray(params_hat[beta_end : beta_end + next_fourier_terms.shape[1]], dtype=float)
    seasonal_var = float(np.exp(float(params_hat[0]) + float(next_fourier_terms[0] @ gamma_hat)) + floor)
    arch_term = alpha * float(eps[-1] ** 2)
    garch_term = 0.0
    for lag, beta in enumerate(betas, start=1):
        garch_term += float(beta) * float(sigma2[-lag])
    return max(seasonal_var + arch_term + garch_term, 1e-6)


def next_sigma2_alternative(
    params_hat: np.ndarray,
    *,
    eps: np.ndarray,
    sigma2: np.ndarray,
    floor: float,
    best_order: int,
) -> float:
    alpha = float(params_hat[1])
    betas = np.asarray(params_hat[2 : 2 + best_order], dtype=float)
    omega = float(np.exp(float(params_hat[0])) + floor)
    arch_term = alpha * float(eps[-1] ** 2)
    garch_term = 0.0
    for lag, beta in enumerate(betas, start=1):
        garch_term += float(beta) * float(sigma2[-lag])
    return max(omega + arch_term + garch_term, floor)


def walk_forward_forecast_base(
    train_df: pd.DataFrame,
    next_row: pd.Series,
    *,
    weekday_flag: str,
    spec: ResolvedModelSpec,
) -> float:
    X_mean = sm.add_constant(train_df[["t", weekday_flag]], has_constant="add")
    ols = sm.OLS(train_df["poi"], X_mean).fit()
    eps = np.asarray(train_df["poi"] - ols.fittedvalues, dtype=float)
    fourier_terms, _ = base_garchx.build_fourier_terms(train_df["woy"], spec.fourier_order)
    best_order, best_result = base_garchx.fit_garch_orders(
        eps,
        fourier_terms,
        max_garch_order=spec.max_garch_order,
        floor=spec.floor,
    )
    _, sigma2 = base_garchx.garchx_filter(
        np.asarray(best_result.x, dtype=float),
        eps,
        fourier_terms,
        garch_order=best_order,
        floor=spec.floor,
    )
    std_resid = eps / np.sqrt(sigma2)
    innovation_dist = fit_residual_distribution(
        std_resid[1:],
        residual_model=spec.residual_model,
    )
    next_x = sm.add_constant(
        build_base_next_design_row(next_row, weekday_flag),
        has_constant="add",
    ).reindex(columns=X_mean.columns, fill_value=0.0)
    mean_next = float(ols.predict(next_x).iloc[0])
    next_fourier_terms, _ = base_garchx.build_fourier_terms(
        pd.Series([int(next_row["woy"])]),
        spec.fourier_order,
    )
    sigma2_next = next_sigma2_base(
        np.asarray(best_result.x, dtype=float),
        eps=eps,
        sigma2=sigma2,
        next_fourier_terms=next_fourier_terms,
        floor=spec.floor,
        best_order=best_order,
    )
    return mean_next + (float(np.sqrt(sigma2_next)) * float(innovation_dist.loc))


def walk_forward_forecast_alternative(
    train_df: pd.DataFrame,
    next_row: pd.Series,
    *,
    weekday_flag: str,
    spec: ResolvedModelSpec,
) -> float:
    fourier_terms, fourier_names = alt_garchx.build_fourier_terms(train_df["woy"], spec.fourier_order)
    fourier_df = pd.DataFrame(fourier_terms, columns=fourier_names, index=train_df.index)
    X_mean = sm.add_constant(
        pd.concat([train_df[["t", weekday_flag]], fourier_df], axis=1),
        has_constant="add",
    )
    ols = sm.OLS(train_df["poi"], X_mean).fit()
    eps = np.asarray(train_df["poi"] - ols.fittedvalues, dtype=float)
    best_order, best_result = alt_garchx.fit_garch_orders(
        eps,
        max_garch_order=spec.max_garch_order,
        floor=spec.floor,
    )
    _, sigma2 = alt_garchx.garchx_filter(
        np.asarray(best_result.x, dtype=float),
        eps,
        garch_order=best_order,
        floor=spec.floor,
    )
    std_resid = eps / np.sqrt(sigma2)
    innovation_dist = fit_residual_distribution(
        std_resid[1:],
        residual_model=spec.residual_model,
    )
    next_fourier_terms, _ = alt_garchx.build_fourier_terms(
        pd.Series([int(next_row["woy"])]),
        spec.fourier_order,
    )
    next_fourier_df = pd.DataFrame(next_fourier_terms, columns=fourier_names)
    next_x = sm.add_constant(
        pd.concat(
            [
                build_base_next_design_row(next_row, weekday_flag),
                next_fourier_df,
            ],
            axis=1,
        ),
        has_constant="add",
    ).reindex(columns=X_mean.columns, fill_value=0.0)
    mean_next = float(ols.predict(next_x).iloc[0])
    sigma2_next = next_sigma2_alternative(
        np.asarray(best_result.x, dtype=float),
        eps=eps,
        sigma2=sigma2,
        floor=spec.floor,
        best_order=best_order,
    )
    return mean_next + (float(np.sqrt(sigma2_next)) * float(innovation_dist.loc))


def walk_forward_forecast(
    context: WalkForwardContext,
    *,
    draw_date: str,
) -> tuple[float, float]:
    frame = context.frame
    train_df = frame[frame["date"] < pd.Timestamp(draw_date)].copy()
    if len(train_df) < 30:
        raise ValueError(f"Not enough training rows for {context.spec.name} on {draw_date}.")
    next_rows = frame[frame["date"] == pd.Timestamp(draw_date)]
    if next_rows.empty:
        raise KeyError(f"{context.spec.name} has no row for draw_date={draw_date}.")
    next_row = next_rows.iloc[0]
    actual_poi = float(next_row["poi"])
    if context.spec.kind == "base":
        forecast = walk_forward_forecast_base(
            train_df,
            next_row,
            weekday_flag=context.weekday_flag,
            spec=context.spec,
        )
    else:
        forecast = walk_forward_forecast_alternative(
            train_df,
            next_row,
            weekday_flag=context.weekday_flag,
            spec=context.spec,
        )
    return float(forecast), actual_poi


def ticket_hit_tuple(ticket_row: pd.Series, actual_draw: pd.Series) -> tuple[int, int]:
    actual_balls = {int(actual_draw[f"ball_{idx}"]) for idx in range(1, 6)}
    actual_stars = {int(actual_draw[f"star_{idx}"]) for idx in range(1, 3)}
    balls = {int(ticket_row[f"ball_{idx}"]) for idx in range(1, 6)}
    stars = {int(ticket_row["star_1"]), int(ticket_row["star_2"])}
    return len(actual_balls.intersection(balls)), len(actual_stars.intersection(stars))


def score_shortlist_returns(
    shortlist: pd.DataFrame,
    *,
    actual_draw: pd.Series,
    prize_table: dict[tuple[int, int], float],
    ticket_cost: float,
) -> dict[str, float | int]:
    if shortlist.empty:
        return {
            "shortlist_size": 0,
            "best_ball_hits": 0,
            "best_star_hits": 0,
            "exact_main5": 0,
            "exact_5plus2": 0,
            "ticket_cost": 0.0,
            "draw_return": 0.0,
            "draw_profit": 0.0,
            "draw_roi": float("nan"),
            "winning_ticket_count": 0,
            "best_ticket_payout": 0.0,
            "any_payout_draw": 0,
            "break_even_draw": 0,
        }

    best_ball_hits = -1
    best_star_hits = -1
    best_ticket_payout = 0.0
    draw_return = 0.0
    winning_ticket_count = 0

    for row in shortlist.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        ball_hits, star_hits = ticket_hit_tuple(row_series, actual_draw)
        payout = float(prize_table.get((ball_hits, star_hits), 0.0))
        draw_return += payout
        if payout > 0.0:
            winning_ticket_count += 1
        if (
            ball_hits > best_ball_hits
            or (ball_hits == best_ball_hits and star_hits > best_star_hits)
            or (
                ball_hits == best_ball_hits
                and star_hits == best_star_hits
                and payout > best_ticket_payout
            )
        ):
            best_ball_hits = int(ball_hits)
            best_star_hits = int(star_hits)
            best_ticket_payout = float(payout)

    cost = float(len(shortlist) * ticket_cost)
    profit = float(draw_return - cost)
    roi = float(profit / cost) if cost > 0 else float("nan")
    return {
        "shortlist_size": int(len(shortlist)),
        "best_ball_hits": int(best_ball_hits),
        "best_star_hits": int(best_star_hits),
        "exact_main5": int(best_ball_hits == 5),
        "exact_5plus2": int(best_ball_hits == 5 and best_star_hits == 2),
        "ticket_cost": cost,
        "draw_return": float(draw_return),
        "draw_profit": profit,
        "draw_roi": roi,
        "winning_ticket_count": int(winning_ticket_count),
        "best_ticket_payout": float(best_ticket_payout),
        "any_payout_draw": int(draw_return > 0.0),
        "break_even_draw": int(profit >= 0.0),
    }


def summarize_method(step_frame: pd.DataFrame) -> MethodSummary:
    total_cost = float(step_frame["ticket_cost"].sum())
    total_return = float(step_frame["draw_return"].sum())
    net_profit = float(total_return - total_cost)
    roi = float(net_profit / total_cost) if total_cost > 0 else float("nan")
    return MethodSummary(
        steps=int(len(step_frame)),
        total_ticket_cost=total_cost,
        total_return=total_return,
        net_profit=net_profit,
        roi=roi,
        mean_return_per_draw=float(step_frame["draw_return"].mean()),
        mean_profit_per_draw=float(step_frame["draw_profit"].mean()),
        mean_best_ball_hits=float(step_frame["best_ball_hits"].mean()),
        mean_best_star_hits=float(step_frame["best_star_hits"].mean()),
        mean_best_ticket_payout=float(step_frame["best_ticket_payout"].mean()),
        any_payout_draw_rate=float(step_frame["any_payout_draw"].mean()),
        break_even_draw_rate=float(step_frame["break_even_draw"].mean()),
        exact_main5_accuracy=float(step_frame["exact_main5"].mean()),
        exact_5plus2_accuracy=float(step_frame["exact_5plus2"].mean()),
        mean_target_gap=float(step_frame["target_gap"].mean()),
        mean_abs_poi_error=float(step_frame["abs_poi_error"].mean()),
        mean_shortlist_size=float(step_frame["shortlist_size"].mean()),
    )


def build_report(manifest: dict) -> str:
    lines = [
        "# GARCH Return Benchmark",
        "",
        f"- Benchmark mode: {manifest['benchmark_mode']}",
        f"- Evaluation window: {manifest['evaluation_start_date']} to {manifest['evaluation_end_date']}",
        f"- Holdout draws: {manifest['holdout_steps']}",
        f"- Ticket budget per draw: common top-{manifest['requested_top_n']} clipped to each step's shared shortlist size",
        f"- Candidate pool size: {manifest['candidate_pool_size']}",
        f"- Ticket cost: {manifest['ticket_cost']}",
        f"- Prize alignment: {manifest['prize_alignment_note']}",
        "",
        "## Ranking by ROI",
    ]
    for index, item in enumerate(manifest["ranking_by_roi"], start=1):
        summary = manifest["methods"][item]
        lines.append(
            f"- {index}. `{item}` | roi={summary['roi']:.4f} | net_profit={summary['net_profit']:.2f} | "
            f"total_return={summary['total_return']:.2f} | total_cost={summary['total_ticket_cost']:.2f} | "
            f"mean_best_ball_hits={summary['mean_best_ball_hits']:.4f} | "
            f"any_payout_draw_rate={summary['any_payout_draw_rate']:.4f}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    history_path = resolve_repo_path(args.history)
    poi_path = resolve_repo_path(args.poi)
    outputs_root = resolve_repo_path(args.outputs_root)
    out_dir = resolve_repo_path(args.out_dir)
    prize_cache_path = resolve_repo_path(args.prize_cache)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = pd.read_csv(history_path, parse_dates=["draw_date"])
    history = history.sort_values("draw_date").drop_duplicates(subset=["draw_date"]).reset_index(drop=True)
    contexts = build_walk_forward_contexts(
        history_path=history_path,
        poi_path=poi_path,
        outputs_root=outputs_root,
    )
    evaluation_dates = common_evaluation_dates(contexts, int(args.holdout))
    effective_holdout = len(evaluation_dates)
    prize_records = load_or_fetch_prize_tables(
        evaluation_dates,
        cache_path=prize_cache_path,
        request_sleep=float(args.request_sleep),
    )

    pool_path = (
        resolve_repo_path(args.pool_path)
        if args.pool_path is not None
        else out_dir / f"sobol_ticket_pool_{int(args.pool_size)}.csv"
    )
    candidate_pool = generate_or_load_candidate_pool(pool_path=pool_path, pool_size=int(args.pool_size))

    rows: list[dict[str, object]] = []
    for offset, draw_date in enumerate(evaluation_dates):
        actual_draw_rows = history[history["draw_date"] == pd.Timestamp(draw_date)]
        if actual_draw_rows.empty:
            raise KeyError(f"History is missing draw_date={draw_date}.")
        actual_draw = actual_draw_rows.iloc[0]
        train_history = history[history["draw_date"] < pd.Timestamp(draw_date)].reset_index(drop=True)
        pair_diag = build_pair_z_diagnostics(train_history)
        _, pair_counts, _, main_n, _, _ = build_branch_frame(
            train_history,
            ratio_mode="raw",
            modulus=None,
            threshold=0.5,
        )

        per_method: dict[str, tuple[pd.DataFrame, int, float, float, int]] = {}
        for spec in TOP_GARCH_MODELS:
            predicted_poi, actual_poi = walk_forward_forecast(
                contexts[spec.name],
                draw_date=draw_date,
            )
            target_score = int(round(predicted_poi))
            shortlist, target_gap = build_garch_shortlist(
                candidate_pool,
                pair_counts=pair_counts,
                main_n=main_n,
                pair_diag=pair_diag,
                target_score=target_score,
                top_n=int(args.top_n),
            )
            per_method[spec.name] = (
                shortlist,
                target_gap,
                predicted_poi,
                actual_poi,
                target_score,
            )

        common_n = min(len(value[0]) for value in per_method.values())
        prize_record = prize_records[draw_date]

        for spec in TOP_GARCH_MODELS:
            shortlist, target_gap, predicted_poi, actual_poi, target_score = per_method[spec.name]
            shortlist = shortlist.head(common_n).reset_index(drop=True)
            scored = score_shortlist_returns(
                shortlist,
                actual_draw=actual_draw,
                prize_table=prize_record.prizes,
                ticket_cost=float(args.ticket_cost),
            )
            rows.append(
                {
                    "draw_date": draw_date,
                    "model": spec.name,
                    "predicted_poi": float(predicted_poi),
                    "actual_poi": float(actual_poi),
                    "abs_poi_error": float(abs(predicted_poi - actual_poi)),
                    "target_score": int(target_score),
                    "target_gap": int(target_gap),
                    "common_top_n": int(common_n),
                    "prize_source_url": prize_record.source_url,
                    "eval_step": int(offset),
                    **scored,
                }
            )

    step_frame = pd.DataFrame(rows).sort_values(["draw_date", "model"]).reset_index(drop=True)
    methods = {
        model: asdict(summarize_method(step_frame[step_frame["model"] == model].reset_index(drop=True)))
        for model in [spec.name for spec in TOP_GARCH_MODELS]
    }
    ranking_by_roi = sorted(
        methods,
        key=lambda name: (
            float(methods[name]["roi"]),
            float(methods[name]["net_profit"]),
            float(methods[name]["mean_best_ball_hits"]),
            float(methods[name]["any_payout_draw_rate"]),
        ),
        reverse=True,
    )

    manifest = {
        "benchmark_mode": "walk_forward_n_plus_1_with_realized_prize_per_winner",
        "holdout_steps": int(effective_holdout),
        "evaluation_start_date": evaluation_dates[0],
        "evaluation_end_date": evaluation_dates[-1],
        "requested_top_n": int(args.top_n),
        "candidate_pool_size": int(len(candidate_pool)),
        "ticket_cost": float(args.ticket_cost),
        "prize_alignment_note": "Prize tables were fetched by exact draw date from lottery.co.uk and scored on the Prize Per Winner column, then cached locally.",
        "models": [spec.name for spec in TOP_GARCH_MODELS],
        "model_windows": {
            name: {
                "start_date": context.spec.start_date,
                "kind": context.spec.kind,
                "residual_model": context.spec.residual_model,
                "fourier_order": context.spec.fourier_order,
            }
            for name, context in contexts.items()
        },
        "ranking_by_roi": ranking_by_roi,
        "methods": methods,
    }

    step_frame.to_csv(out_dir / "garch_return_benchmark_steps.csv", index=False)
    (out_dir / "garch_return_benchmark.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    (out_dir / "garch_return_benchmark.md").write_text(
        build_report(manifest),
        encoding="utf-8",
    )

    print("GARCH RETURN BENCHMARK")
    print(
        f"mode={manifest['benchmark_mode']} holdout={effective_holdout} "
        f"top_n={args.top_n} pool={len(candidate_pool)}"
    )
    for name in ranking_by_roi:
        summary = methods[name]
        print(
            f"{name}: roi={summary['roi']:.4f} "
            f"net_profit={summary['net_profit']:.2f} "
            f"return={summary['total_return']:.2f} "
            f"cost={summary['total_ticket_cost']:.2f} "
            f"mean_best_ball_hits={summary['mean_best_ball_hits']:.4f}"
        )
    print(out_dir / "garch_return_benchmark_steps.csv")
    print(out_dir / "garch_return_benchmark.json")
    print(out_dir / "garch_return_benchmark.md")


if __name__ == "__main__":
    main()
