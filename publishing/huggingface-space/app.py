from __future__ import annotations

import html
import json
import tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import gradio as gr
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROFILE_ROOT = ROOT / "data" / "profiles"
REFRESH_STATUS_PATH = ROOT / "data" / "refresh_status.json"
PROFILE_ORDER = ("synthetic", "euromillions", "nl-lotto")
HOUSE_AGENT = "uniform_random"

# Keep both navigation levels explicit: lottery profile first, then analysis screen.

AGENTS = {
    "uniform_random": {
        "name": "House / Null",
        "short": "HOUSE",
        "role": "Equal-budget fair-draw reference",
        "color": "#8194a3",
        "bet": 50,
        "house": 50,
    },
    "frequency": {
        "name": "Momentum Scout",
        "short": "MOMENTUM",
        "role": "Frequency-led bet engineering",
        "color": "#20d5cf",
        "bet": 94,
        "house": 30,
    },
    "unpopularity": {
        "name": "Crowd Escape",
        "short": "ESCAPE",
        "role": "Low-popularity payout-pressure lane",
        "color": "#ffb84d",
        "bet": 36,
        "house": 96,
    },
    "gingerm": {
        "name": "Co-occurrence",
        "short": "PAIR GRAPH",
        "role": "Recurring pair-structure search",
        "color": "#4ea8ff",
        "bet": 78,
        "house": 57,
    },
    "spectral_contrarian": {
        "name": "Spectral Contrarian",
        "short": "SPECTRAL",
        "role": "Graph structure with contrarian pressure",
        "color": "#ef84c5",
        "bet": 58,
        "house": 88,
    },
    "parallax": {
        "name": "Parallax Guard",
        "short": "PARALLAX",
        "role": "Residual and regime-shift guard",
        "color": "#ff7168",
        "bet": 69,
        "house": 73,
    },
    "coordinated_aggregation": {
        "name": "Market Coordinator",
        "short": "COORDINATOR",
        "role": "Equal-budget multi-agent portfolio",
        "color": "#d8f35c",
        "bet": 84,
        "house": 84,
    },
}

PROFILE_STYLE = {
    "synthetic": {
        "code": "LAB / 01",
        "kicker": "EUROMILLIONS FORMAT / DETERMINISTIC CONTROL",
        "accent": "#20d5cf",
        "draw_label": "CONTROL DRAW",
        "format": "5 numbers from 50 + 2 Lucky Stars from 12",
        "cadence": "Weekly generated control",
    },
    "euromillions": {
        "code": "EU / 02",
        "kicker": "EUROMILLIONS / OBSERVED PUBLIC HISTORY",
        "accent": "#ffd84d",
        "draw_label": "OFFICIAL CONTEST DRAW",
        "format": "5 numbers from 50 + 2 Lucky Stars from 12",
        "cadence": "Tuesday and Friday draws",
    },
    "nl-lotto": {
        "code": "NL / 03",
        "kicker": "NEDERLANDSE LOTTO / OPERATOR HISTORY",
        "accent": "#ff6b42",
        "draw_label": "OFFICIAL SATURDAY DRAW",
        "format": "6 numbers from 45",
        "cadence": "Saturday draw",
    },
}


@dataclass(frozen=True)
class Profile:
    key: str
    manifest: dict
    leaderboard: pd.DataFrame
    contests: pd.DataFrame
    tickets: pd.DataFrame
    prospective: pd.DataFrame
    directory: Path


def _load_profile(key: str) -> Profile:
    directory = PROFILE_ROOT / key
    return Profile(
        key=key,
        manifest=json.loads((directory / "manifest.json").read_text(encoding="utf-8")),
        leaderboard=pd.read_csv(directory / "leaderboard.csv"),
        contests=pd.read_csv(directory / "contests.csv"),
        tickets=pd.read_csv(directory / "tickets.csv"),
        prospective=pd.read_csv(directory / "prospective.csv"),
        directory=directory,
    )


PROFILES = {key: _load_profile(key) for key in PROFILE_ORDER}


def _load_refresh_status() -> dict:
    try:
        payload = json.loads(REFRESH_STATUS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"games": {}}
    return payload if isinstance(payload, dict) else {"games": {}}


REFRESH_STATUS = _load_refresh_status()


def _availability_notice() -> str:
    games = REFRESH_STATUS.get("games", {})
    unavailable = [
        (key, value)
        for key, value in games.items()
        if isinstance(value, dict) and value.get("available") is False
    ]
    if not unavailable:
        return ""
    labels = {"euromillions": "EuroMillions", "nl-lotto": "Nederlandse Lotto"}
    items = "".join(
        "<li><strong>"
        + html.escape(labels.get(key, key))
        + "</strong> — latest verified data remains online. "
        + html.escape(str(value.get("message", "Official source temporarily unavailable.")))
        + " <small>Checked "
        + html.escape(str(value.get("checked_utc", "unknown")))
        + "</small></li>"
        for key, value in unavailable
    )
    return (
        '<section role="status" style="margin:14px 0;padding:16px 19px;'
        'border:1px solid rgba(255,184,77,.45);border-radius:16px;'
        'background:rgba(255,184,77,.08);color:#d6dedf">'
        '<b style="color:#ffca70">DATA REFRESH NOTICE</b>'
        '<ul style="margin:9px 0 0;padding-left:20px">' + items + "</ul></section>"
    )


def _agent_meta(agent: str) -> dict:
    return AGENTS.get(
        agent,
        {
            "name": agent.replace("_", " ").title(),
            "short": "AGENT",
            "role": "Benchmark strategy",
            "color": "#8194a3",
            "bet": 50,
            "house": 50,
        },
    )


def _agent_name(agent: str) -> str:
    return str(_agent_meta(agent)["name"])


def _has_auxiliary(value: object) -> bool:
    return not pd.isna(value) and bool(str(value).strip())


def _number_balls(value: object, auxiliary: bool = False) -> str:
    if not _has_auxiliary(value):
        return ""
    class_name = "lottery-ball star" if auxiliary else "lottery-ball"
    prefix = '<i aria-hidden="true">&#9733;</i>' if auxiliary else ""
    return "".join(
        f'<span class="{class_name}">{prefix}<b>{html.escape(number)}</b></span>'
        for number in str(value).split()
    )


def _draw_line(main: object, auxiliary: object, label: str) -> str:
    auxiliary_balls = _number_balls(auxiliary, auxiliary=True)
    plus = '<span class="draw-plus">+</span>' if auxiliary_balls else ""
    return f"""
    <div class="draw-stage">
      <span>{html.escape(label)}</span>
      <div>{_number_balls(main)}{plus}{auxiliary_balls}</div>
    </div>
    """


def _global_hero() -> str:
    cards = []
    for key in PROFILE_ORDER:
        profile = PROFILES[key]
        style = PROFILE_STYLE[key]
        leader = profile.leaderboard.iloc[0]
        data_label = "LAB CONTROL" if key == "synthetic" else "OBSERVED DRAWS"
        cards.append(
            f"""
            <article class="profile-card" style="--profile:{style['accent']}">
              <div><span>{style['code']}</span><i>{data_label}</i></div>
              <h3>{html.escape(profile.manifest['display_name'])}</h3>
              <p>{style['format']}</p>
              <footer><b>{_agent_name(str(leader['agent']))}</b><strong>{leader['mean_roi_alpha_pp']:+.3f} pp</strong></footer>
            </article>
            """
        )
    return f"""
    <header class="global-hero">
      <div class="live-line"><i></i> FORWARD-ONLY LOTTERY AGENT BENCHMARK</div>
      <h1>Model the ticket.<br><em>Model the house.</em></h1>
      <p>Replay ranked agents on a lottery draw stage, compare ROI allocation against an equal-budget null, and freeze the next candidate sets before the following contest.</p>
      <div class="hero-facts">
        <span><b>3</b> separated profiles</span>
        <span><b>7</b> agents per market</span>
        <span><b>12</b> forward contests</span>
        <span><b>12</b> tickets per agent</span>
      </div>
    </header>
    <section class="profile-deck">{''.join(cards)}</section>
    """


PROTOCOL = """
<section class="protocol-strip">
  <div><b>01</b><span>LOCK</span><p>Commit every candidate hash before the draw.</p></div>
  <i></i>
  <div><b>02</b><span>DRAW</span><p>Close the round on the published result.</p></div>
  <i></i>
  <div><b>03</b><span>SCORE</span><p>Compare equal budgets against the uniform null.</p></div>
  <i></i>
  <div><b>04</b><span>ITERATE</span><p>Advance the animated walk by one contest.</p></div>
</section>
"""


def _profile_identity(profile: Profile) -> str:
    manifest = profile.manifest
    style = PROFILE_STYLE[profile.key]
    history = manifest["history"]
    source = manifest["source"]
    evaluation = manifest["evaluation"]
    leader = profile.leaderboard.iloc[0]
    source_link = (
        f'<a href="{html.escape(source["url"], quote=True)}" target="_blank" rel="noopener">View source record</a>'
        if source.get("url")
        else "Repository-generated control"
    )
    profile_class = "synthetic" if profile.key == "synthetic" else "observed"
    return f"""
    <section class="lottery-identity {profile_class}" style="--profile:{style['accent']}">
      <div class="identity-copy">
        <span class="identity-kicker">{style['kicker']}</span>
        <h2>{html.escape(manifest['display_name'])}</h2>
        <p class="game-format">{style['format']}</p>
        <p>{style['cadence']}. The arena evaluates {history['rows']:,} historical draws through <strong>{history['last_draw']}</strong> under one locked forward-only contract.</p>
      </div>
      <div class="identity-score">
        <span>HOLDOUT LEADER</span>
        <strong>{_agent_name(str(leader['agent']))}</strong>
        <b>{leader['mean_roi_alpha_pp']:+.3f} pp ROI alpha</b>
        <small>{int(leader['contests_above_null'])}/{int(leader['contests'])} contests above null</small>
      </div>
      <div class="identity-ledger">
        <div><span>DATA</span><b>{'SYNTHETIC' if profile.key == 'synthetic' else 'OBSERVED'}</b></div>
        <div><span>DRAW RANGE</span><b>{history['first_draw']} to {history['last_draw']}</b></div>
        <div><span>BUDGET</span><b>{evaluation['budget_per_agent']} tickets / agent</b></div>
        <div><span>SNAPSHOT</span><b class="mono">{history['snapshot_sha256'][:12]}</b></div>
        <div><span>SOURCE</span><b>{source_link}</b></div>
      </div>
    </section>
    """


def _overall_agent_ladder(profile: Profile) -> str:
    cards = []
    for row in profile.leaderboard.sort_values("rank").itertuples(index=False):
        meta = _agent_meta(str(row.agent))
        is_house = row.agent == HOUSE_AGENT
        badge = "REFERENCE" if is_house else f"{row.mean_roi_alpha_pp:+.3f} pp"
        consistency = max(0.0, min(100.0, float(row.consistency_pct)))
        cards.append(
            f"""
            <article class="overall-agent" style="--agent:{meta['color']}">
              <div class="overall-rank">{int(row.rank):02d}</div>
              <div class="overall-copy">
                <div><span>{meta['short']}</span><h3>{meta['name']}</h3><p>{meta['role']}</p></div>
                <strong class="{'null-badge' if is_house else 'alpha-badge'}">{badge}</strong>
              </div>
              <div class="overall-track"><i style="width:{consistency:.1f}%"></i></div>
              <footer><span>Above null <b>{'CONTROL' if is_house else f'{int(row.contests_above_null)}/{int(row.contests)}'}</b></span><span>Consistency <b>{row.consistency_pct:.0f}%</b></span><span>Pair reach <b>{row.mean_pair_coverage_pct:.2f}%</b></span></footer>
            </article>
            """
        )
    return f"""
    <section class="leader-section">
      <div class="panel-heading"><div><span>TWELVE-CONTEST TABLE</span><h2>Agents ranked against the house</h2></div><p>The principal result is ROI alpha, not absolute return. Consistency shows how often an agent remained above the equal-budget uniform reference.</p></div>
      <div class="overall-ladder">{''.join(cards)}</div>
    </section>
    """


def _contest_frame(profile: Profile, contest_number: int) -> pd.DataFrame:
    return profile.contests[
        profile.contests["contest_number"].astype(int) == int(contest_number)
    ].sort_values("contest_rank")


def _round_summary(profile: Profile, contest_number: int) -> str:
    frame = _contest_frame(profile, contest_number)
    winner = frame.iloc[0]
    house = frame[frame["agent"] == HOUSE_AGENT].iloc[0]
    edge_count = int((frame[frame["agent"] != HOUSE_AGENT]["roi_alpha_vs_house_pp"] > 0).sum())
    return f"""
    <section class="round-head" style="--profile:{PROFILE_STYLE[profile.key]['accent']}">
      <div>
        <span class="round-kicker">CONTEST {int(contest_number):02d} / {html.escape(str(winner['draw_date']))}</span>
        <h2>{html.escape(profile.manifest['display_name'])} market close</h2>
        {_draw_line(winner['actual_main'], winner['actual_auxiliary'], PROFILE_STYLE[profile.key]['draw_label'])}
      </div>
      <div class="round-kpis">
        <div><span>ROUND WINNER</span><strong>{_agent_name(str(winner['agent']))}</strong></div>
        <div><span>ROI ALPHA</span><strong class="positive">{winner['roi_alpha_vs_house_pp']:+.3f} pp</strong></div>
        <div><span>AGENTS OVER NULL</span><strong>{edge_count}/{len(frame) - 1}</strong></div>
        <div><span>NULL EXPECTED ROI</span><strong>{house['expected_roi_pct']:.2f}%</strong></div>
      </div>
    </section>
    """


def _round_agent_ladder(profile: Profile, contest_number: int) -> str:
    cards = []
    for row in _contest_frame(profile, contest_number).itertuples(index=False):
        meta = _agent_meta(str(row.agent))
        is_house = row.agent == HOUSE_AGENT
        badge = "NULL" if is_house else ("EDGE" if row.roi_alpha_vs_house_pp > 0 else "BELOW")
        width = max(2.0, min(100.0, float(row.win_rate_vs_house_pct)))
        cards.append(
            f"""
            <article class="agent-rank" style="--agent:{meta['color']}">
              <div class="rank-orbit">{int(row.contest_rank):02d}</div>
              <div class="agent-copy">
                <div class="agent-line">
                  <div><span class="agent-code">{meta['short']}</span><h3>{meta['name']}</h3></div>
                  <span class="edge-badge {'null' if is_house else 'positive' if row.roi_alpha_vs_house_pp > 0 else 'negative'}">{badge} {row.roi_alpha_vs_house_pp:+.3f} pp</span>
                </div>
                <p>{meta['role']} / best result match {int(row.best_main_hits)} main + {int(row.best_auxiliary_hits)} auxiliary</p>
                <div class="consistency-track"><i style="width:{width:.1f}%"></i></div>
                <div class="rank-metrics"><span>Consistency <b>{row.win_rate_vs_house_pct:.0f}%</b></span><span>Anomaly <b>{row.anomaly_index:.0f}</b></span><span>Pair reach <b>{row.pair_coverage_pct:.2f}%</b></span></div>
              </div>
            </article>
            """
        )
    return '<section class="agent-ladder">' + "".join(cards) + "</section>"


def _agent_walk(profile: Profile, contest_number: int) -> str:
    frame = profile.tickets[
        profile.tickets["contest_number"].astype(int) == int(contest_number)
    ].copy()
    width, height = 940, 470
    left, right, top, bottom = 80, 890, 44, 392
    svg_parts = []
    for position in range(0, 101, 20):
        x = left + (right - left) * position / 100
        y = bottom - (bottom - top) * position / 100
        svg_parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" class="grid-line" />')
        svg_parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" class="grid-line" />')

    for path_index, (agent, agent_frame) in enumerate(frame.groupby("agent", sort=False)):
        meta = _agent_meta(str(agent))
        agent_frame = agent_frame.sort_values("ticket_index")
        points = []
        coordinates = []
        for row in agent_frame.itertuples(index=False):
            x = left + (right - left) * float(row.crowd_escape_score) / 100
            y = bottom - (bottom - top) * float(row.momentum_score) / 100
            points.append(f"{x:.1f},{y:.1f}")
            coordinates.append((x, y, row))
        svg_parts.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{meta["color"]}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" class="walk-path" style="animation-delay:{path_index * 0.12:.2f}s" />'
        )
        for index, (x, y, row) in enumerate(coordinates):
            outcome = int(row.main_hits) + int(row.auxiliary_hits)
            radius = 4.5 + min(5.0, outcome * 1.5)
            auxiliary = "" if pd.isna(row.auxiliary_draw) else str(row.auxiliary_draw)
            ticket_text = f"{row.main_draw}{f' + {auxiliary}' if auxiliary else ''}"
            title = html.escape(
                f"{meta['name']} / {ticket_text} / result match {row.main_hits}+{row.auxiliary_hits}"
            )
            svg_parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{meta["color"]}" class="walk-node" style="animation-delay:{0.28 + path_index * 0.08 + index * 0.08:.2f}s"><title>{title}</title></circle>'
            )
            if outcome >= 3:
                svg_parts.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius + 7:.1f}" fill="none" stroke="{meta["color"]}" class="signal-ring" />'
                )

    legend = "".join(
        f'<span><i style="background:{meta["color"]}"></i>{meta["name"]}</span>'
        for meta in AGENTS.values()
    )
    gradient_id = f"arena-{profile.key}-{int(contest_number)}"
    return f"""
    <section class="walk-panel">
      <div class="panel-heading"><div><span>ANIMATED CONTEST REPLAY</span><h2>Agents walk the lottery set surface</h2></div><p>Each path follows the agent's twelve pre-contest tickets in submitted order. Coordinates are calculated before the draw; larger rings expose stronger matches only after close.</p></div>
      <div class="walk-legend">{legend}</div>
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="Animated agent ticket paths">
        <defs><linearGradient id="{gradient_id}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#102f3b" /><stop offset="1" stop-color="#111927" /></linearGradient></defs>
        <rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" rx="20" fill="url(#{gradient_id})" />
        {''.join(svg_parts)}
        <text x="{(left + right) / 2}" y="447" class="axis-label">HOUSE / DRAW ENGINEERING: CROWD ESCAPE</text>
        <text x="23" y="{(top + bottom) / 2}" transform="rotate(-90 23 {(top + bottom) / 2})" class="axis-label">BET ENGINEERING: MOMENTUM</text>
      </svg>
    </section>
    """


def _contest_table(profile: Profile, contest_number: int) -> pd.DataFrame:
    frame = _contest_frame(profile, contest_number)
    return pd.DataFrame(
        {
            "Rank": frame["contest_rank"].astype(int),
            "Agent": frame["agent"].map(_agent_name),
            "ROI alpha vs null": frame["roi_alpha_vs_house_pp"].map(lambda value: f"{value:+.3f} pp"),
            "Consistency to date": frame["win_rate_vs_house_pct"].map(lambda value: f"{value:.0f}%"),
            "Pair coverage": frame["pair_coverage_pct"].map(lambda value: f"{value:.2f}%"),
            "Best match": frame.apply(lambda row: f"{int(row['best_main_hits'])}+{int(row['best_auxiliary_hits'])}", axis=1),
            "Expected ROI": frame["expected_roi_pct"].map(lambda value: f"{value:.2f}%"),
        }
    ).reset_index(drop=True)


def _winning_tickets(profile: Profile, contest_number: int) -> pd.DataFrame:
    contest = _contest_frame(profile, contest_number)
    winner = contest.iloc[0]
    frame = profile.tickets[
        (profile.tickets["contest_number"].astype(int) == int(contest_number))
        & (profile.tickets["agent"] == winner["agent"])
    ].sort_values("ticket_index")
    return pd.DataFrame(
        {
            "Ticket": frame["ticket_index"].astype(int),
            "Main numbers": frame["main_draw"],
            "Lucky stars / Aux": frame["auxiliary_draw"].fillna(""),
            "Result match": frame.apply(lambda row: f"{int(row['main_hits'])}+{int(row['auxiliary_hits'])}", axis=1),
            "Crowd escape": frame["crowd_escape_score"].map(lambda value: f"{value:.0f}"),
            "Momentum": frame["momentum_score"].map(lambda value: f"{value:.0f}"),
            "Commitment": frame["commitment_sha256"].str[:12],
        }
    ).reset_index(drop=True)


def render_contest(key: str, contest_number: int) -> tuple[str, str, str, pd.DataFrame, pd.DataFrame]:
    profile = PROFILES[key]
    contest_number = int(contest_number)
    return (
        _round_summary(profile, contest_number),
        _round_agent_ladder(profile, contest_number),
        _agent_walk(profile, contest_number),
        _contest_table(profile, contest_number),
        _winning_tickets(profile, contest_number),
    )


def shift_contest(
    key: str, contest_number: int, delta: int
) -> tuple[int, str, str, str, pd.DataFrame, pd.DataFrame]:
    profile = PROFILES[key]
    contests = [
        int(value)
        for value in sorted(profile.contests["contest_number"].astype(int).unique())
    ]
    current = contests.index(int(contest_number))
    target = contests[(current + int(delta)) % len(contests)]
    return (target, *render_contest(key, target))


def advance_contest(
    key: str, contest_number: int
) -> tuple[int, str, str, str, pd.DataFrame, pd.DataFrame]:
    return shift_contest(key, contest_number, 1)


def set_timer(playing: bool) -> gr.Timer:
    return gr.Timer(active=bool(playing))


def _normalized_evidence(profile: Profile) -> dict[str, float]:
    board = profile.leaderboard.copy()
    minimum = float(board["mean_roi_alpha_pp"].min())
    span = float(board["mean_roi_alpha_pp"].max()) - minimum or 1.0
    return {
        str(row.agent): 0.70 * float(row.consistency_pct)
        + 0.30 * ((float(row.mean_roi_alpha_pp) - minimum) / span * 100)
        for row in board.itertuples(index=False)
    }


def _candidate_dataset(profile: Profile, house_balance: int, top_n: int) -> pd.DataFrame:
    frame = profile.prospective.copy()
    evidence = _normalized_evidence(profile)
    house_ratio = int(house_balance) / 100
    budget = int(profile.manifest["evaluation"]["budget_per_agent"])
    bet_scores = []
    house_scores = []
    evidence_scores = []
    allocation_scores = []
    for row in frame.itertuples(index=False):
        meta = _agent_meta(str(row.agent))
        rank_strength = 100 - (int(row.rank_within_agent) - 1) / max(1, budget - 1) * 24
        bet_score = 0.78 * float(meta["bet"]) + 0.22 * rank_strength
        house_score = 0.78 * float(meta["house"]) + 0.22 * rank_strength
        evidence_score = evidence[str(row.agent)]
        strategy_blend = (1 - house_ratio) * bet_score + house_ratio * house_score
        allocation_score = 0.60 * strategy_blend + 0.40 * evidence_score
        bet_scores.append(bet_score)
        house_scores.append(house_score)
        evidence_scores.append(evidence_score)
        allocation_scores.append(allocation_score)

    frame["bet_engineering_score"] = bet_scores
    frame["house_draw_engineering_score"] = house_scores
    frame["historical_evidence_score"] = evidence_scores
    frame["allocation_priority_score"] = allocation_scores
    frame["house_draw_weight_pct"] = int(house_balance)
    frame["bet_engineering_weight_pct"] = 100 - int(house_balance)
    frame = frame.sort_values(
        ["allocation_priority_score", "backtest_mean_roi_alpha_pp", "evidence_rank"],
        ascending=[False, False, True],
    ).head(int(top_n)).reset_index(drop=True)
    frame.insert(0, "allocation_rank", range(1, len(frame) + 1))
    frame["ranking_scope"] = "pending strategy allocation; not draw probability"
    return frame


def _candidate_map(frame: pd.DataFrame) -> str:
    width, height = 940, 410
    left, right, top, bottom = 80, 890, 40, 334
    points = []
    for row in frame.itertuples(index=False):
        x = left + (right - left) * float(row.house_draw_engineering_score) / 100
        y = bottom - (bottom - top) * float(row.bet_engineering_score) / 100
        meta = _agent_meta(str(row.agent))
        radius = 9 if int(row.allocation_rank) <= 3 else 5
        auxiliary = "" if pd.isna(row.auxiliary_draw) else str(row.auxiliary_draw)
        ticket_text = f"{row.main_draw}{f' + {auxiliary}' if auxiliary else ''}"
        title = html.escape(
            f"#{row.allocation_rank} {ticket_text} / {meta['name']} / priority {row.allocation_priority_score:.1f}"
        )
        points.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{meta["color"]}" class="candidate-dot" style="animation-delay:{int(row.allocation_rank) * 0.035:.3f}s"><title>{title}</title></circle>'
        )
    return f"""
    <section class="candidate-map">
      <div class="panel-heading"><div><span>PENDING STRATEGY SURFACE</span><h2>Where the frozen sets concentrate</h2></div><p>Position reflects strategy-family allocation, not a claim that one number is more likely to be drawn. The first three larger points lead the current allocation.</p></div>
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="Pending set allocation map">
        <rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" rx="20" class="map-bg" />
        <line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="map-axis" />
        <line x1="{left}" y1="{bottom}" x2="{left}" y2="{top}" class="map-axis" />
        <line x1="{left}" y1="{(top + bottom) / 2}" x2="{right}" y2="{(top + bottom) / 2}" class="map-mid" />
        <line x1="{(left + right) / 2}" y1="{top}" x2="{(left + right) / 2}" y2="{bottom}" class="map-mid" />
        {''.join(points)}
        <text x="{(left + right) / 2}" y="383" class="axis-label">HOUSE / DRAW ENGINEERING</text>
        <text x="23" y="{(top + bottom) / 2}" transform="rotate(-90 23 {(top + bottom) / 2})" class="axis-label">BET ENGINEERING</text>
      </svg>
    </section>
    """


def _candidate_display(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Rank": frame["allocation_rank"].astype(int),
            "Agent": frame["agent"].map(_agent_name),
            "Main numbers": frame["main_draw"],
            "Lucky stars / Aux": frame["auxiliary_draw"].fillna(""),
            "Priority": frame["allocation_priority_score"].map(lambda value: f"{value:.1f}"),
            "Bet engineering": frame["bet_engineering_score"].map(lambda value: f"{value:.1f}"),
            "House / draw": frame["house_draw_engineering_score"].map(lambda value: f"{value:.1f}"),
            "Historical alpha": frame["backtest_mean_roi_alpha_pp"].map(lambda value: f"{value:+.3f} pp"),
            "Status": frame["score_status"],
            "Commitment": frame["commitment_sha256"].str[:12],
        }
    )


def render_candidates(
    key: str, house_balance: int, top_n: int
) -> tuple[str, str, pd.DataFrame, str]:
    profile = PROFILES[key]
    frame = _candidate_dataset(profile, int(house_balance), int(top_n))
    leader = frame.iloc[0]
    meta = _agent_meta(str(leader["agent"]))
    auxiliary = _number_balls(leader["auxiliary_draw"], auxiliary=True)
    plus = '<span class="draw-plus">+</span>' if auxiliary else ""
    summary = f"""
    <section class="candidate-summary" style="--profile:{PROFILE_STYLE[key]['accent']}">
      <div class="candidate-pick"><span>TOP FROZEN SET / PENDING</span><div>{_number_balls(leader['main_draw'])}{plus}{auxiliary}</div></div>
      <div><span>SUPPORTING AGENT</span><strong>{meta['name']}</strong><small>{meta['role']}</small></div>
      <div><span>ALLOCATION</span><strong>{100-int(house_balance)}% bet / {int(house_balance)}% house</strong><small>Priority {leader['allocation_priority_score']:.1f} / 100</small></div>
      <div><span>HISTORICAL EVIDENCE</span><strong>{leader['backtest_mean_roi_alpha_pp']:+.3f} pp</strong><small>{leader['backtest_consistency_pct']:.0f}% above null</small></div>
    </section>
    """
    output = Path(tempfile.gettempdir()) / f"lottobench_{key}_pending_b{int(house_balance)}_n{int(top_n)}.csv"
    frame.to_csv(output, index=False, lineterminator="\n")
    return summary, _candidate_map(frame), _candidate_display(frame), str(output)


CLAIMS_NOTE = """
<section class="claims-note">
  <strong>Exact interpretation of “against the house”</strong>
  <p>ROI alpha is modeled expected-ROI displacement from the seeded uniform, equal-budget null. It is not realized profit and does not change the mechanical probability of a fair lottery draw. Pending sets are committed research outputs, not official forecasts or betting advice.</p>
</section>
"""

NAVIGATION_GUIDE = """
<section class="navigation-guide">
  <span>NAVIGATION</span>
  <div><b>1</b><strong>Choose a lottery profile</strong><small>Lab control, EuroMillions, or NL Lotto</small></div>
  <i></i>
  <div><b>2</b><strong>Choose a screen</strong><small>Agent Arena or Pending Set Lab</small></div>
</section>
"""

SCREEN_GUIDE = """
<div class="screen-guide"><span>SCREEN SWITCH</span><strong>Use the highlighted buttons below to move between scored agent performance and pending sets.</strong></div>
"""


def _build_profile_tab(profile: Profile) -> None:
    gr.HTML(_profile_identity(profile))
    gr.HTML(SCREEN_GUIDE)
    with gr.Tabs(elem_classes="screen-switcher"):
        with gr.Tab("SCREEN A / AGENT ARENA"):
            gr.HTML(_overall_agent_ladder(profile))
            contests = [
                int(value)
                for value in sorted(profile.contests["contest_number"].astype(int).unique())
            ]
            initial_contest = contests[-1]
            initial = render_contest(profile.key, initial_contest)
            with gr.Row(elem_classes="replay-controls"):
                contest_selector = gr.Slider(
                    minimum=contests[0],
                    maximum=contests[-1],
                    value=initial_contest,
                    step=1,
                    label="Contest replay",
                    info="Move through the twelve scored lottery closes.",
                )
                autoplay = gr.Checkbox(value=False, label="Play weekly replay")
            with gr.Row(elem_classes="contest-nav"):
                previous_button = gr.Button(
                    "PREVIOUS CONTEST", elem_classes="page-nav-button"
                )
                next_button = gr.Button(
                    "NEXT CONTEST", variant="primary", elem_classes="page-nav-button"
                )

            summary = gr.HTML(initial[0])
            ladder = gr.HTML(initial[1])
            walk = gr.HTML(initial[2])
            ranking = gr.Dataframe(initial[3], interactive=False, label="Exact contest ranking")
            winner_tickets = gr.Dataframe(
                initial[4], interactive=False, label="Winning agent's committed tickets"
            )
            outputs = [summary, ladder, walk, ranking, winner_tickets]
            contest_selector.change(
                fn=partial(render_contest, profile.key),
                inputs=contest_selector,
                outputs=outputs,
            )
            previous_button.click(
                fn=partial(shift_contest, profile.key, delta=-1),
                inputs=contest_selector,
                outputs=[contest_selector, *outputs],
            )
            next_button.click(
                fn=partial(shift_contest, profile.key, delta=1),
                inputs=contest_selector,
                outputs=[contest_selector, *outputs],
            )
            timer = gr.Timer(value=5.5, active=False)
            autoplay.change(fn=set_timer, inputs=autoplay, outputs=timer)
            timer.tick(
                fn=partial(advance_contest, profile.key),
                inputs=contest_selector,
                outputs=[contest_selector, *outputs],
            )
            with gr.Row():
                gr.File(value=str(profile.directory / "tickets.csv"), label="Download scored ticket ledger")
                gr.File(value=str(profile.directory / "contests.csv"), label="Download contest results")
            gr.HTML(CLAIMS_NOTE)

        with gr.Tab("SCREEN B / PENDING SET LAB"):
            gr.HTML(
                f"""
                <section class="lab-intro" style="--profile:{PROFILE_STYLE[profile.key]['accent']}">
                  <span>AFTER HISTORY CUTOFF {profile.manifest['history']['last_draw']} / UNSCORED</span>
                  <h2>Shift allocation between two strategy surfaces.</h2>
                  <p>Bet engineering favors momentum and pair structure. House/draw engineering favors crowd escape, dispersion, and contrarian structure. The control changes priority across already frozen agent submissions; it never edits the committed numbers.</p>
                </section>
                """
            )
            with gr.Row():
                balance = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=50,
                    step=5,
                    label="Allocation toward house / draw engineering",
                    info="0 = bet engineering, 100 = house/draw engineering.",
                )
                top_n = gr.Slider(
                    minimum=12,
                    maximum=56,
                    value=24,
                    step=4,
                    label="Sets to include in export",
                )
            rebuild = gr.Button(
                "RE-RANK FROZEN SETS", variant="primary", elem_classes="action-button"
            )
            initial_candidates = render_candidates(profile.key, 50, 24)
            candidate_summary = gr.HTML(initial_candidates[0])
            candidate_map = gr.HTML(initial_candidates[1])
            candidate_table = gr.Dataframe(
                initial_candidates[2], interactive=False, label="Ranked pending lottery sets"
            )
            with gr.Row():
                candidate_download = gr.File(
                    value=initial_candidates[3], label="Download allocation-ranked dataset"
                )
                gr.File(
                    value=str(profile.directory / "prospective.csv"),
                    label="Download original frozen commitments",
                )
            rebuild.click(
                fn=partial(render_candidates, profile.key),
                inputs=[balance, top_n],
                outputs=[candidate_summary, candidate_map, candidate_table, candidate_download],
            )
            gr.HTML(CLAIMS_NOTE)


CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root { --ink:#f4f8f8; --muted:#9bafb6; --bg:#071116; --panel:#0c1d24; --panel-2:#112832; --line:rgba(220,240,244,.14); --acid:#d8f35c; --teal:#20d5cf; --gold:#ffd84d; --orange:#ff6b42; }
body,.gradio-container { color:var(--ink); font-family:'Space Grotesk',sans-serif; background:radial-gradient(circle at 8% 0,rgba(32,213,207,.13),transparent 27rem),radial-gradient(circle at 90% 13%,rgba(255,216,77,.09),transparent 25rem),linear-gradient(rgba(255,255,255,.023) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.023) 1px,transparent 1px),var(--bg); background-size:auto,auto,36px 36px,36px 36px,auto; }
.gradio-container { max-width:1320px!important; padding:26px!important; }
.gradio-container .prose { color:var(--ink); }
.gradio-container .tabs .tab-container[role="tablist"] { display:flex!important; gap:9px!important; margin:14px 0 20px!important; padding:8px!important; border:1px solid rgba(216,243,92,.24)!important; border-radius:16px!important; background:#091a21!important; box-shadow:0 12px 34px rgba(0,0,0,.22); }
.gradio-container .tabs .tab-container[role="tablist"]>button[role="tab"] { min-height:48px!important; color:#dce9eb!important; background:#132d37!important; border:1px solid rgba(220,240,244,.22)!important; border-radius:10px!important; padding:12px 18px!important; font:600 11px 'DM Mono',monospace!important; letter-spacing:.075em!important; box-shadow:inset 0 0 0 1px rgba(255,255,255,.025); transition:background .18s ease,border-color .18s ease,color .18s ease,transform .18s ease!important; }
.gradio-container .tabs .tab-container[role="tablist"]>button[role="tab"]:hover { color:#071116!important; background:var(--gold)!important; border-color:var(--gold)!important; transform:translateY(-1px); }
.gradio-container .tabs .tab-container[role="tablist"]>button[role="tab"].selected,.gradio-container .tabs .tab-container[role="tablist"]>button[role="tab"][aria-selected="true"] { color:#071116!important; background:var(--acid)!important; border-color:var(--acid)!important; box-shadow:0 0 0 3px rgba(216,243,92,.16),0 8px 22px rgba(0,0,0,.24)!important; }
.gradio-container .tabs .tab-container[role="tablist"]>button[role="tab"]:focus-visible,.gradio-container button:focus-visible { outline:3px solid var(--gold)!important; outline-offset:3px!important; }
.gradio-container .tabs .overflow-menu:not(.hide)>button { min-width:132px!important; color:#071116!important; background:var(--gold)!important; border-color:var(--gold)!important; }
.gradio-container .tabs .overflow-menu:not(.hide)>button:after { content:'CHANGE PAGE'; margin-left:8px; font:700 9px 'DM Mono',monospace; letter-spacing:.08em; }
.gradio-container button.primary { background:var(--acid)!important; color:#071116!important; border:0!important; font-weight:700; }
.gradio-container button.secondary { border-color:var(--line)!important; }
.gradio-container .form { background:rgba(12,29,36,.9); border-color:var(--line); }
.global-hero { border:1px solid var(--line); border-radius:30px; padding:clamp(30px,6vw,72px); background:linear-gradient(125deg,rgba(14,48,52,.97),rgba(8,18,24,.96) 60%,rgba(43,46,21,.85)); box-shadow:0 30px 100px rgba(0,0,0,.34); position:relative; overflow:hidden; }
.global-hero:after { content:''; position:absolute; width:330px; height:330px; border:1px solid rgba(216,243,92,.2); border-radius:50%; right:-105px; top:-155px; box-shadow:0 0 0 46px rgba(216,243,92,.025),0 0 0 92px rgba(216,243,92,.018); }
.live-line,.panel-heading span,.identity-kicker,.round-kicker,.lab-intro>span { color:var(--teal); font:500 10px 'DM Mono',monospace; letter-spacing:.16em; text-transform:uppercase; }
.live-line i { display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--acid); margin-right:9px; animation:pulse 1.8s infinite; }
.global-hero h1 { max-width:980px; font-size:clamp(45px,7.4vw,92px); line-height:.91; letter-spacing:-.065em; margin:24px 0; }
.global-hero h1 em { color:var(--acid); font-style:normal; }
.global-hero>p { max-width:790px; color:#bfd0d2; font-size:18px; line-height:1.6; }
.hero-facts { display:flex; flex-wrap:wrap; gap:9px; margin-top:32px; position:relative; z-index:2; }
.hero-facts span { padding:10px 14px; border:1px solid var(--line); border-radius:99px; color:var(--muted); background:rgba(0,0,0,.14); font:11px 'DM Mono',monospace; }
.hero-facts b { color:var(--ink); }
.profile-deck { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:14px 0; }
.profile-card { border:1px solid var(--line); border-top:3px solid var(--profile); border-radius:18px; padding:19px; background:rgba(10,27,34,.92); }
.profile-card>div,.profile-card footer { display:flex; justify-content:space-between; gap:12px; align-items:center; }
.profile-card span,.profile-card i { color:var(--profile); font:9px 'DM Mono',monospace; letter-spacing:.1em; font-style:normal; }
.profile-card h3 { margin:16px 0 5px; font-size:20px; }
.profile-card p { color:var(--muted); font-size:12px; min-height:32px; }
.profile-card footer { border-top:1px solid var(--line); padding-top:13px; font-size:11px; }
.profile-card footer strong { color:var(--profile); font-family:'DM Mono',monospace; }
.protocol-strip { display:grid; grid-template-columns:1fr 25px 1fr 25px 1fr 25px 1fr; gap:12px; align-items:center; border:1px solid var(--line); border-radius:20px; padding:18px; margin:14px 0 24px; background:rgba(8,23,29,.88); }
.protocol-strip>i { height:1px; background:var(--line); }
.protocol-strip b { color:var(--acid); font:12px 'DM Mono',monospace; margin-right:7px; }
.protocol-strip span { font:500 10px 'DM Mono',monospace; letter-spacing:.1em; }
.protocol-strip p { color:var(--muted); font-size:11px; margin:6px 0 0; }
.navigation-guide { display:grid; grid-template-columns:auto 1fr 34px 1fr; gap:14px; align-items:center; margin:18px 0 8px; padding:17px 20px; border:1px solid rgba(255,216,77,.35); border-radius:18px; background:linear-gradient(90deg,rgba(255,216,77,.10),rgba(10,29,36,.92)); }
.navigation-guide>span { color:var(--gold); font:600 10px 'DM Mono',monospace; letter-spacing:.14em; }
.navigation-guide>div { display:grid; grid-template-columns:30px 1fr; column-gap:9px; align-items:center; }
.navigation-guide b { grid-row:1/3; width:27px; height:27px; display:grid; place-items:center; border-radius:50%; background:var(--gold); color:#071116; font:700 10px 'DM Mono',monospace; }
.navigation-guide strong { font-size:13px; }.navigation-guide small { color:var(--muted); font-size:10px; }
.navigation-guide>i { height:1px; background:rgba(255,216,77,.38); }
.screen-guide { display:flex; align-items:center; gap:14px; padding:11px 15px; border-left:3px solid var(--acid); color:var(--muted); background:rgba(216,243,92,.06); }
.screen-guide span { color:var(--acid); font:600 9px 'DM Mono',monospace; letter-spacing:.12em; white-space:nowrap; }.screen-guide strong { color:#c5d3d5; font-size:11px; }
.contest-nav { margin:4px 0 12px; }
.page-nav-button button,.page-nav-button { min-height:48px!important; border:1px solid rgba(216,243,92,.36)!important; font:700 11px 'DM Mono',monospace!important; letter-spacing:.09em!important; }
.action-button button,.action-button { min-height:50px!important; box-shadow:0 0 0 3px rgba(216,243,92,.10),0 10px 24px rgba(0,0,0,.22)!important; font:700 11px 'DM Mono',monospace!important; letter-spacing:.09em!important; }
.lottery-identity { display:grid; grid-template-columns:1.4fr .75fr; gap:24px; border:1px solid var(--line); border-left:4px solid var(--profile); border-radius:24px; padding:28px; margin:18px 0; background:linear-gradient(135deg,rgba(14,39,48,.97),rgba(8,20,27,.95)); }
.identity-copy h2 { font-size:clamp(35px,5vw,58px); letter-spacing:-.05em; margin:8px 0 5px; }
.identity-copy .game-format { color:var(--profile); font:500 14px 'DM Mono',monospace; margin:0 0 16px; }
.identity-copy>p:last-child { color:#b3c4c8; line-height:1.55; max-width:690px; }
.identity-score { border:1px solid color-mix(in srgb,var(--profile) 42%,transparent); border-radius:18px; padding:20px; background:rgba(0,0,0,.14); }
.identity-score span,.identity-ledger span { display:block; color:var(--muted); font:9px 'DM Mono',monospace; letter-spacing:.1em; margin-bottom:7px; }
.identity-score strong { display:block; font-size:22px; margin:16px 0 8px; }
.identity-score b { display:block; color:var(--profile); font:500 14px 'DM Mono',monospace; }
.identity-score small { display:block; color:var(--muted); margin-top:8px; }
.identity-ledger { grid-column:1/-1; display:grid; grid-template-columns:repeat(5,1fr); border-top:1px solid var(--line); padding-top:18px; }
.identity-ledger div { padding:2px 14px; border-right:1px solid var(--line); }
.identity-ledger div:first-child { padding-left:0; }.identity-ledger div:last-child { border:0; }
.identity-ledger b { font-size:11px; line-height:1.5; }.identity-ledger a { color:var(--profile); text-decoration:none; }
.mono { font-family:'DM Mono',monospace; }
.leader-section,.walk-panel,.candidate-map { border:1px solid var(--line); border-radius:24px; padding:22px; background:rgba(8,22,28,.94); margin:16px 0; }
.panel-heading { display:flex; justify-content:space-between; gap:28px; align-items:start; }
.panel-heading h2 { font-size:28px; letter-spacing:-.035em; margin:7px 0 17px; }
.panel-heading p { max-width:470px; color:var(--muted); font-size:12px; line-height:1.55; }
.overall-ladder { display:grid; grid-template-columns:repeat(2,1fr); gap:10px; }
.overall-agent { position:relative; overflow:hidden; border:1px solid var(--line); border-radius:17px; padding:16px 16px 14px 72px; background:linear-gradient(145deg,rgba(17,40,49,.95),rgba(8,20,26,.96)); }
.overall-agent:before { content:''; position:absolute; inset:0 auto 0 0; width:3px; background:var(--agent); }
.overall-rank { position:absolute; left:16px; top:17px; width:40px; height:40px; border:1px solid var(--agent); color:var(--agent); border-radius:50%; display:grid; place-items:center; font:12px 'DM Mono',monospace; }
.overall-copy { display:flex; justify-content:space-between; gap:12px; align-items:start; }
.overall-copy span { color:var(--agent); font:9px 'DM Mono',monospace; letter-spacing:.1em; }
.overall-copy h3 { margin:3px 0; font-size:17px; }.overall-copy p { color:var(--muted); font-size:11px; margin:0; }
.alpha-badge,.null-badge { color:var(--acid); border:1px solid currentColor; border-radius:99px; padding:6px 8px; font:9px 'DM Mono',monospace; white-space:nowrap; }
.null-badge { color:var(--muted); }
.overall-track,.consistency-track { height:4px; background:rgba(255,255,255,.07); border-radius:8px; overflow:hidden; margin-top:12px; }
.overall-track i,.consistency-track i { display:block; height:100%; background:var(--agent); }
.overall-agent footer,.rank-metrics { display:flex; flex-wrap:wrap; gap:12px; color:var(--muted); font:9px 'DM Mono',monospace; margin-top:10px; }
.overall-agent footer b,.rank-metrics b { color:var(--ink); }
.round-head { display:grid; grid-template-columns:1.25fr .75fr; gap:25px; border:1px solid var(--line); border-top:3px solid var(--profile); border-radius:24px; padding:24px; margin:18px 0 12px; background:linear-gradient(135deg,rgba(15,43,51,.98),rgba(9,20,27,.97)); }
.round-head h2 { font-size:28px; margin:7px 0 17px; }
.draw-stage>span,.candidate-pick>span { display:block; color:var(--muted); font:9px 'DM Mono',monospace; letter-spacing:.12em; margin-bottom:10px; }
.draw-stage>div,.candidate-pick>div { display:flex; flex-wrap:wrap; align-items:center; gap:7px; }
.lottery-ball { width:39px; height:39px; display:inline-flex; align-items:center; justify-content:center; border-radius:50%; background:linear-gradient(145deg,#fffdf3,#dfe7e6); color:#071116; box-shadow:inset 0 -6px 11px rgba(0,0,0,.18),0 5px 12px rgba(0,0,0,.2); font:700 13px 'DM Mono',monospace; }
.lottery-ball.star { position:relative; background:linear-gradient(145deg,#ffe878,#ffb52f); color:#372b00; }
.lottery-ball.star i { position:absolute; color:rgba(255,255,255,.55); font-size:29px; font-style:normal; line-height:1; }
.lottery-ball.star b { position:relative; z-index:1; }
.draw-plus { color:var(--muted); padding:0 3px; }
.round-kpis { display:grid; grid-template-columns:1fr 1fr; gap:9px; }
.round-kpis div { border-left:2px solid var(--line); padding:9px 12px; }
.round-kpis span,.candidate-summary>div>span { display:block; color:var(--muted); font:9px 'DM Mono',monospace; letter-spacing:.08em; margin-bottom:7px; }
.round-kpis strong { font-size:16px; }.positive { color:var(--acid)!important; }.negative,.null { color:var(--muted)!important; }
.agent-ladder { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:16px; }
.agent-rank { display:grid; grid-template-columns:50px 1fr; gap:14px; position:relative; overflow:hidden; border:1px solid var(--line); border-radius:18px; padding:15px; background:linear-gradient(145deg,rgba(17,39,48,.97),rgba(8,20,26,.97)); }
.agent-rank:before { content:''; position:absolute; inset:0 auto 0 0; width:3px; background:var(--agent); }
.rank-orbit { width:44px; height:44px; display:grid; place-items:center; border:1px solid var(--agent); color:var(--agent); border-radius:50%; font:12px 'DM Mono',monospace; }
.agent-line { display:flex; justify-content:space-between; gap:10px; align-items:start; }.agent-line h3 { margin:3px 0 0; font-size:17px; }
.agent-code { color:var(--agent); font:9px 'DM Mono',monospace; letter-spacing:.1em; }
.edge-badge { border:1px solid currentColor; border-radius:99px; padding:6px 8px; font:9px 'DM Mono',monospace; white-space:nowrap; }
.agent-copy p { color:var(--muted); font-size:11px; margin:6px 0 10px; }
.walk-legend { display:flex; flex-wrap:wrap; gap:14px; color:var(--muted); font:9px 'DM Mono',monospace; }
.walk-legend span { display:inline-flex; align-items:center; gap:6px; }.walk-legend i { width:8px; height:8px; border-radius:50%; }
.walk-panel svg,.candidate-map svg { display:block; width:100%; height:auto; margin-top:10px; }
.grid-line { stroke:rgba(255,255,255,.045); }.axis-label { fill:#789097; font:9px 'DM Mono',monospace; letter-spacing:.08em; text-anchor:middle; }
.walk-path { stroke-dasharray:1200; stroke-dashoffset:1200; animation:trace 3s ease forwards; opacity:.78; }.walk-node,.candidate-dot { opacity:0; animation:nodeIn .35s ease forwards; }.signal-ring { transform-box:fill-box; transform-origin:center; animation:ring 1.8s ease-out infinite; }
.lab-intro { border:1px solid var(--line); border-left:4px solid var(--profile); border-radius:22px; padding:25px; margin:17px 0; background:linear-gradient(135deg,rgba(14,42,49,.97),rgba(9,20,27,.97)); }
.lab-intro>span { color:var(--profile); }.lab-intro h2 { font-size:31px; margin:9px 0; }.lab-intro p { max-width:820px; color:#b4c5c8; line-height:1.6; }
.candidate-summary { display:grid; grid-template-columns:1.35fr 1fr 1fr 1fr; gap:10px; border:1px solid var(--line); border-top:3px solid var(--profile); border-radius:22px; padding:19px; margin:16px 0; background:rgba(11,28,35,.95); }
.candidate-summary>div { border-left:1px solid var(--line); padding:7px 13px; }.candidate-summary>div:first-child { border:0; }
.candidate-summary strong { display:block; font-size:15px; margin-top:10px; }.candidate-summary small { display:block; color:var(--muted); margin-top:6px; }
.candidate-summary .lottery-ball { width:34px; height:34px; font-size:11px; }.candidate-summary .lottery-ball.star i { font-size:25px; }
.map-bg { fill:#0b222b; }.map-axis { stroke:rgba(255,255,255,.18); }.map-mid { stroke:rgba(255,255,255,.07); stroke-dasharray:4 7; }
.claims-note { border:1px solid rgba(255,184,77,.32); border-radius:17px; padding:17px; margin:16px 0 35px; background:rgba(255,184,77,.07); }
.claims-note strong { color:#ffca70; }.claims-note p { color:#b7c4c4; font-size:12px; line-height:1.6; margin:6px 0 0; }
@keyframes pulse { 70% { box-shadow:0 0 0 9px rgba(216,243,92,0); } 100% { box-shadow:0 0 0 0 rgba(216,243,92,0); } }
@keyframes trace { to { stroke-dashoffset:0; } } @keyframes nodeIn { from { opacity:0; transform:scale(.35); transform-origin:center; } to { opacity:1; transform:scale(1); } } @keyframes ring { 0% { opacity:.8; transform:scale(.6); } 80%,100% { opacity:0; transform:scale(1.45); } }
@media(max-width:900px) { .gradio-container{padding:12px!important}.profile-deck,.overall-ladder,.agent-ladder,.round-head,.candidate-summary{grid-template-columns:1fr}.protocol-strip{grid-template-columns:1fr 1fr}.protocol-strip>i{display:none}.navigation-guide{grid-template-columns:1fr 1fr}.navigation-guide>span{grid-column:1/-1}.navigation-guide>i{display:none}.lottery-identity{grid-template-columns:1fr}.identity-ledger{grid-template-columns:1fr 1fr}.identity-ledger div{border-right:0;border-bottom:1px solid var(--line);padding:10px 0}.panel-heading{display:block}.global-hero h1{font-size:46px}.profile-card p{min-height:0} }
@media(max-width:560px) { .profile-deck,.protocol-strip,.identity-ledger,.round-kpis{grid-template-columns:1fr}.global-hero{padding:27px 22px}.global-hero h1{font-size:39px}.lottery-identity,.leader-section,.walk-panel,.candidate-map{padding:17px}.overall-agent{padding-left:64px}.lottery-ball{width:34px;height:34px;font-size:11px}.agent-rank{grid-template-columns:1fr}.rank-orbit{width:36px;height:36px}.agent-line{display:block}.edge-badge{display:inline-block;margin-top:8px} }
"""


with gr.Blocks(title="LottoBench Lottery Agent Arena", css=CSS) as demo:
    gr.HTML(_global_hero())
    gr.HTML(_availability_notice())
    gr.HTML(PROTOCOL)
    gr.HTML(NAVIGATION_GUIDE)
    with gr.Tabs(elem_classes="profile-switcher"):
        with gr.Tab("LOTTERY 01 / EUROMILLIONS LAB"):
            _build_profile_tab(PROFILES["synthetic"])
        with gr.Tab("LOTTERY 02 / EUROMILLIONS"):
            _build_profile_tab(PROFILES["euromillions"])
        with gr.Tab("LOTTERY 03 / NL LOTTO"):
            _build_profile_tab(PROFILES["nl-lotto"])


if __name__ == "__main__":
    demo.launch()
