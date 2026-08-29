"""outcome_tracker.py -- forward-only, long-horizon evidence ledger for the lottery repo.

Purpose (stated plainly): this repo's hypothesis is that coordinated, coverage-and-co-occurrence-aware
set selection produces *better tracked outcomes than random chance*. That is an empirical claim, and
the evaluation method is to record predictions BEFORE each draw and score them AFTER, over a
long window (the owner's plan: ~3 years), then read off whether there is a consistent edge -- and PARK
the project if there is not. This module is that ledger.

It never bets, never handles money, and makes no promise of winnings. Against a fair draw the expected
edge is zero; the point is to measure, not to assume. Every draw is also scored against a matched
uniform-random control generated from the same draw key, so "did the method beat luck?" is answerable.

Workflow
--------
    # 1. Before a draw: generate and log this draw's portfolio (history must NOT contain the
    #    draw). History path, ledger path and ticket price all default per game, so a scheduled
    #    run needs no flags beyond the draw being recorded.
    lotto-track record --draw-key 2026-08-18 --methods all

    # 2. After the draw: settle it with the official result. Re-running is safe: settlement is
    #    idempotent, and an already-settled draw is skipped rather than scored twice.
    lotto-track settle --draw-key 2026-08-18 --actual-main 4,17,23,38,45 --actual-stars 3,9 \
        --payout-table official-payouts/2026-08-18.json

    # Without --payout-table the draw still settles, but every money column is recorded as NaN
    # rather than as a zero prize, so ROI is never silently understated.

    # 3. Any time: read the cumulative verdict.
    lotto-track report
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import storage
from .envelope import data_sha256
from .likely_set_generator import PRESETS, GameConfig, build_portfolio, resolve_config
from .protocol import GameSpec
from .realized_roi import ROI_SCHEMA_VERSION, canonical_sha256, result_digest

PENDING = "pending_predictions.jsonl"
SETTLED = "settled_predictions.jsonl"
RESULTS = "results.csv"

#: Canonical local history store, matching ``lottobench fetch`` and the optional API.
DEFAULT_HISTORY = os.environ.get("LOTTERIES_HISTORY", "data/lotteries.db")

#: The game assumed when none is named. The tracked experiment in this repo is the EuroMillions
#: one; any other game still works, it just has to be named.
DEFAULT_GAME = "euromillions"

#: Identity of a result row. Settling the same draw twice replaces the row instead of appending a
#: second one, so a crashed or repeated run cannot inflate the evidence base.
RESULT_KEY = ("draw_key", "method")

#: Per-game operational defaults, so an unattended run needs no flags. Only values actually known
#: for the market this repo tracks are listed; a game absent here records fine, it simply has no
#: money columns unless ``--ticket-price`` is passed.
GAME_DEFAULTS: dict[str, dict] = {
    "euromillions": {"ticket_price": 2.50, "currency": "EUR"},
}


def _package_version() -> str:
    try:
        return importlib.metadata.version("lottobench")
    except importlib.metadata.PackageNotFoundError:
        from . import __version__

        return __version__


def _provider_version(method: str, label: str) -> str:
    from . import registry

    if method == "cooccurrence":
        return "1.0.0"
    name = _REGISTRY_ALIASES.get(method, label)
    return registry.version(name)

#: Methods that can be entered into the tracked competition. ``cooccurrence`` is the owner's
#: level-set generator and stays the default so existing ledgers keep their method label. The rest
#: are :class:`~lotteries_core.protocol.InferenceProvider` implementations, recorded through the
#: same protocol so any future contribution joins by adding one registry entry.
PROVIDER_METHODS = (
    "gingerm",
    "spectral_contrarian",
    "parallax",
    "garch_markov_branch",
    "sequence_transformer",
    "frequency",
    "unpopularity",
    "uniform_random",
    "perron_frobenius",
    "parallax_guard",
    "parallax_ablation",
)
METHOD_CHOICES = ("cooccurrence", *PROVIDER_METHODS)


def _spec_from_config(cfg: GameConfig) -> GameSpec:
    return GameSpec(cfg.name, cfg.main_n, cfg.main_k, cfg.star_n, cfg.star_k)


#: Ledger CLI names kept stable for existing ledgers, mapped onto :mod:`lotteries_core.registry`.
#: The registry is the single list of providers; this only preserves the shorter CLI spellings.
_REGISTRY_ALIASES = {
    "perron_frobenius": "spectral_contrarian",
    "parallax_guard": "parallax",
    "parallax_ablation": "parallax_guard_ablation",
}


def _make_provider(method: str):
    """Instantiate a registered provider by its ledger CLI name."""
    from . import registry

    name = _REGISTRY_ALIASES.get(method, method)
    try:
        return registry.create(name)
    except KeyError as exc:
        raise ValueError(f"unknown provider method {method!r}; choose from {PROVIDER_METHODS}") from exc


def _portfolio_for_method(method: str, df: pd.DataFrame, cfg: GameConfig, args) -> tuple[str, list, float | None]:
    """Return ``(method_label, tickets, target)`` for one entrant.

    The RNG is seeded from the draw key so a recorded portfolio is reproducible from the ledger
    alone; providers that ignore the RNG are deterministic anyway.
    """
    if method == "cooccurrence":
        pf = build_portfolio(
            df, cfg, target_mode=args.target_mode, window=args.window,
            pairing=args.pairing, n_sets=args.n_sets,
        )
        return f"cooccurrence:{args.target_mode}:{args.pairing}", pf["tickets"], pf["target"]

    provider = _make_provider(method)
    spec = _spec_from_config(cfg)
    try:
        provider.fit(df, spec)
    except TypeError:
        provider.fit(df)
    seed = int.from_bytes(hashlib.sha256(f"record:{method}:{args.draw_key}".encode()).digest()[:8], "big")
    result = provider.propose(spec, args.n_sets, np.random.default_rng(seed % (2**32)))
    return provider.name, list(result.tickets), None


# ------------------------------------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------------------------------------


def _score_portfolio(tickets, actual_main: set[int], actual_stars: set[int], cfg: GameConfig) -> dict:
    best_main = 0
    best_star = 0
    mean_main = 0.0
    jackpot = 0
    top_tier = (0, 0)
    tier_counts: dict[str, int] = {}
    for mains, stars in tickets:
        mh = len(set(mains) & actual_main)
        sh = len(set(stars) & actual_stars)
        best_main = max(best_main, mh)
        best_star = max(best_star, sh)
        mean_main += mh
        top_tier = max(top_tier, (mh, sh))
        tier = f"{mh}+{sh}"
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        if mh == cfg.main_k and sh == cfg.star_k:
            jackpot += 1
    n = max(len(tickets), 1)
    return {
        "n_sets": len(tickets),
        "best_main": best_main,
        "best_star": best_star,
        "mean_main": mean_main / n,
        "top_tier_main": top_tier[0],
        "top_tier_star": top_tier[1],
        "jackpot": jackpot,
        "tier_counts": tier_counts,
    }


def _random_control(cfg: GameConfig, n_sets: int, draw_key: str) -> list[tuple]:
    seed_bytes = hashlib.sha256(f"control:{cfg.name}:{draw_key}".encode()).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big")
    rng = np.random.default_rng(seed)
    out: list[tuple] = []
    seen: set[tuple] = set()
    while len(out) < n_sets:
        mains = tuple(sorted(int(x) for x in rng.choice(np.arange(1, cfg.main_n + 1), cfg.main_k, replace=False)))
        stars = (
            tuple(sorted(int(x) for x in rng.choice(np.arange(1, cfg.star_n + 1), cfg.star_k, replace=False)))
            if cfg.star_k > 0 else ()
        )
        ticket = (mains, stars)
        if ticket not in seen:
            seen.add(ticket)
            out.append(ticket)
    return out


#: Keys written at settlement time. They are excluded from ``record_sha256`` deliberately: that
#: digest proves what was preregistered *before* the draw, so it must not move when the draw is
#: scored afterwards. The settlement inputs carry their own digest instead. ``actual_main`` and
#: ``actual_stars`` are listed because records settled by earlier versions stored them at the top
#: level, and those records must still reproduce their original preregistration digest.
_SETTLEMENT_KEYS = ("settlement", "settlement_sha256", "actual_main", "actual_stars")


def _record_digest(record: dict) -> str:
    unsigned = {
        k: v for k, v in record.items() if k != "record_sha256" and k not in _SETTLEMENT_KEYS
    }
    payload = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _settlement_digest(settlement: dict) -> str:
    payload = json.dumps(settlement, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_record(record: dict) -> None:
    """Check the preregistration half: the portfolio is what was logged before the draw."""
    expected = record.get("record_sha256")
    if not expected or expected != _record_digest(record):
        raise ValueError(f"prediction record failed integrity check: {record.get('draw_key')}")


def _validate_settlement(record: dict) -> None:
    """Check the settlement half: the scoring inputs are what they were when the draw was scored.

    Records settled before settlement digests existed carry no ``settlement`` block; there is
    nothing to compare them against, so they pass.
    """
    settlement = record.get("settlement")
    if not isinstance(settlement, dict):
        return
    expected = record.get("settlement_sha256")
    if not expected or expected != _settlement_digest(settlement):
        raise ValueError(f"settlement failed integrity check: {record.get('draw_key')}")


# ------------------------------------------------------------------------------------------------
# Ledger IO
# ------------------------------------------------------------------------------------------------


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _atomic_write_text(path: Path, text: str) -> None:
    """Write through a temporary file in the same directory, then rename over the target.

    The ledger is the experiment's only evidence, and settlement rewrites whole files. A crash
    mid-write has to leave the previous file intact rather than a truncated one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _rewrite_jsonl(path: Path, records: list[dict]) -> None:
    _atomic_write_text(path, "".join(json.dumps(r, sort_keys=True) + "\n" for r in records))


# ------------------------------------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------------------------------------


def _default_ledger(game: str) -> Path:
    """Per-game ledger directory, so ``--ledger`` is only needed to override it."""
    return Path("ledger") / game


def _resolve_ledger(args, game: str) -> Path:
    return Path(args.ledger) if args.ledger else _default_ledger(game)


def _game_default(game: str, key: str, fallback=None):
    return GAME_DEFAULTS.get(game, {}).get(key, fallback)


def _resolve_methods(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(METHOD_CHOICES)
    methods = [m.strip() for m in raw.split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHOD_CHOICES]
    if unknown:
        raise SystemExit(f"unknown method(s) {unknown}; choose from {list(METHOD_CHOICES)} or 'all'")
    return methods


def _apply_game_default(args) -> None:
    """Fill in the default preset when no game was named at all.

    A fully custom spec (``--main-n``/``--main-k``) stays custom: defaulting the preset there
    would relabel someone's own game as euromillions.
    """
    if not getattr(args, "preset", None) and not (args.main_n and args.main_k):
        args.preset = DEFAULT_GAME


def cmd_record(args) -> None:
    _apply_game_default(args)
    cfg = resolve_config(args)
    ledger = _resolve_ledger(args, cfg.name)
    df = storage.read_history(args.history, game=cfg.name)
    methods = _resolve_methods(args.methods)

    # An unattended run should not have to know the price of a ticket, and a run without one has
    # to say so now: a missing price is only felt hours later, as NaN money columns at settlement.
    ticket_price = args.ticket_price
    if ticket_price is None:
        ticket_price = _game_default(cfg.name, "ticket_price")
    if ticket_price is None:
        print(f"[warn] no ticket price known for {cfg.name}: prize, net return and ROI will "
              "settle as NaN. Pass --ticket-price to track money.")

    # One control per draw key, shared by every entrant, so the competition is paired: all methods
    # are judged against the identical luck baseline on the identical draw.
    control = _random_control(cfg, args.n_sets, args.draw_key)
    history_hash = data_sha256(df)
    pending = _read_jsonl(ledger / PENDING)
    settled = _read_jsonl(ledger / SETTLED)
    already = {(r["draw_key"], r["method"]) for r in pending + settled}

    recorded = 0
    for method in methods:
        label, tickets, target = _portfolio_for_method(method, df, cfg, args)
        if (args.draw_key, label) in already:
            print(f"[skip] {label} already recorded for draw {args.draw_key}")
            continue
        record = {
            "schema_version": ROI_SCHEMA_VERSION,
            "lottobench_version": _package_version(),
            "draw_key": args.draw_key,
            "game": cfg.name,
            "config": asdict(cfg),
            "method": label,
            "provider_name": label,
            "provider_version": _provider_version(method, label),
            "target": target,
            "n_sets": len(tickets),
            "tickets": [[list(m), list(s)] for m, s in tickets],
            "control_tickets": [[list(m), list(s)] for m, s in control[: len(tickets)]],
            "generated_from_rows": len(df),
            "history_sha256": history_hash,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "ticket_proof_sha256": args.ticket_proof_sha256 or None,
            "ticket_price": ticket_price,
        }
        provider_config = {
            "provider_name": record["provider_name"],
            "provider_version": record["provider_version"],
            "game_config": record["config"],
            "target_mode": args.target_mode,
            "pairing": args.pairing,
            "window": args.window,
            "n_sets": record["n_sets"],
        }
        record["provider_config_sha256"] = canonical_sha256(provider_config)
        record["benchmark_id"] = canonical_sha256(
            {
                "game": record["game"],
                "draw_key": record["draw_key"],
                "history_sha256": record["history_sha256"],
                "n_sets": record["n_sets"],
            }
        )
        record["record_sha256"] = _record_digest(record)
        _append_jsonl(ledger / PENDING, record)
        already.add((args.draw_key, label))
        recorded += 1
        target_note = f", target={target:.1f}" if target is not None else ""
        print(f"[record] {label}: {len(tickets)} sets for draw {args.draw_key} "
              f"({cfg.name}{target_note})")

    if recorded:
        print(f"[record] {recorded} entrant(s) -> {ledger / PENDING}")


def _parse_nums(s: str) -> set[int]:
    return {int(x) for x in str(s).replace(" ", "").split(",") if x != ""}


def cmd_settle(args) -> None:
    ledger = _resolve_ledger(args, args.game)
    pending = _read_jsonl(ledger / PENDING)
    settled = _read_jsonl(ledger / SETTLED)
    actual_main = _parse_nums(args.actual_main)
    actual_stars = _parse_nums(args.actual_stars) if args.actual_stars else set()

    # One payout table per settlement run, resolved once. No table means "no official prize data
    # for this draw", which is recorded as NaN and never as a zero prize: a zero would bias ROI
    # downward for every draw settled without a table, invisibly.
    payout_table = _load_payout_table(args.payout_table)
    # Presence is "a table was supplied", not "the table paid out". An official breakdown under
    # which the portfolio won nothing is real evidence of a zero prize; no table at all is not.
    payout_present = bool(args.payout_table)
    payout_source = args.payout_source if payout_present else "none"
    payout_sha = (
        hashlib.sha256(Path(args.payout_table).read_bytes()).hexdigest() if payout_present else ""
    )
    if not payout_present:
        print("[settle] no payout table: prize, net return and ROI are recorded as NaN for this "
              "draw, and report counts it as a non-monetary settlement.")

    remaining = [r for r in pending if r["draw_key"] != args.draw_key]
    candidates = [r for r in pending if r["draw_key"] == args.draw_key]
    previously = [r for r in settled if r["draw_key"] == args.draw_key]

    if args.force:
        kept_settled = [r for r in settled if r["draw_key"] != args.draw_key]
        candidates += previously
    else:
        kept_settled = settled
        done = {(r["draw_key"], r["method"]) for r in previously}
        for rec in previously:
            _validate_settlement(rec)
            print(f"[skip] {rec['method']} already settled for draw {rec['draw_key']}")
        candidates = [r for r in candidates if (r["draw_key"], r["method"]) not in done]

    # Score every entrant before writing anything: a draw that fails validation halfway through
    # must not leave half of its entrants scored in the ledger.
    rows, settled_now = [], []
    for rec in candidates:
        _validate_record(rec)
        cfg = GameConfig(**rec["config"])
        if len(actual_main) != cfg.main_k or any(not (1 <= n <= cfg.main_n) for n in actual_main):
            raise ValueError(f"actual mains must contain {cfg.main_k} distinct values in 1..{cfg.main_n}")
        if len(actual_stars) != cfg.star_k or any(not (1 <= n <= cfg.star_n) for n in actual_stars):
            raise ValueError(f"actual stars must contain {cfg.star_k} distinct values in 1..{cfg.star_n}")
        tickets = [(tuple(m), tuple(s)) for m, s in rec["tickets"]]
        method = _score_portfolio(tickets, actual_main, actual_stars, cfg)
        control_tickets = [(tuple(m), tuple(s)) for m, s in rec["control_tickets"]]
        control = _score_portfolio(control_tickets, actual_main, actual_stars, cfg)
        nan = float("nan")
        method_prize = _portfolio_prize(method["tier_counts"], payout_table) if payout_present else nan
        control_prize = _portfolio_prize(control["tier_counts"], payout_table) if payout_present else nan
        ticket_price = rec.get("ticket_price")
        stake = float(ticket_price) * len(tickets) if ticket_price is not None else nan
        # NaN propagates through the arithmetic, which is the point: a draw missing either the
        # stake or the payout table has no monetary result, rather than a misleading one.
        method_net = method_prize - stake
        control_net = control_prize - stake
        monetary = payout_present and not np.isnan(stake) and stake > 0
        currency = args.currency or _game_default(rec["game"], "currency", "EUR")
        row = {
            "schema_version": rec.get("schema_version", 1),
            "lottobench_version": rec.get("lottobench_version", "unknown"),
            "draw_key": rec["draw_key"],
            "game": rec["game"],
            "method": rec["method"],
            "provider_name": rec.get("provider_name", rec["method"]),
            "provider_version": rec.get("provider_version", "legacy-unversioned"),
            "provider_config_sha256": rec.get("provider_config_sha256", ""),
            "benchmark_id": rec.get("benchmark_id", ""),
            "n_sets": method["n_sets"],
            "m_best_main": method["best_main"],
            "m_best_star": method["best_star"],
            "m_mean_main": round(method["mean_main"], 4),
            "m_top_tier": f"{method['top_tier_main']}+{method['top_tier_star']}",
            "m_jackpot": method["jackpot"],
            "m_tier_counts": json.dumps(method["tier_counts"], sort_keys=True),
            "m_portfolio_prize": method_prize,
            "m_net_return": method_net,
            "realized_roi": method_net / stake if monetary else nan,
            "c_best_main": control["best_main"],
            "c_mean_main": round(control["mean_main"], 4),
            "c_jackpot": control["jackpot"],
            "c_tier_counts": json.dumps(control["tier_counts"], sort_keys=True),
            "c_portfolio_prize": control_prize,
            "c_net_return": control_net,
            "control_realized_roi": control_net / stake if monetary else nan,
            "realized_roi_lift": (method_net - control_net) / stake if monetary else nan,
            "stake": stake,
            "lift_mean_main": round(method["mean_main"] - control["mean_main"], 4),
            "prize_lift": method_prize - control_prize,
            "currency": currency,
            "outcome_source": args.outcome_source,
            "payout_table_present": int(payout_present),
            "payout_source": payout_source,
            "payout_table_sha256": payout_sha,
            "settled_utc": datetime.now(timezone.utc).isoformat(),
            "purchase_proof_hash_present": int(bool(rec.get("ticket_proof_sha256"))),
            "record_sha256": rec["record_sha256"],
        }
        row["result_sha256"] = result_digest(row) if monetary else ""
        rows.append(row)

        # Settlement facts live in their own object with their own digest. record_sha256 proves
        # what was preregistered before the draw; settlement_sha256 proves the inputs the draw was
        # scored against. Neither can invalidate the other.
        settlement = {
            "actual_main": sorted(actual_main),
            "actual_stars": sorted(actual_stars),
            "settled_utc": row["settled_utc"],
            "outcome_source": args.outcome_source,
            "payout_source": payout_source,
            "payout_table_sha256": payout_sha,
            "result_sha256": row["result_sha256"],
        }
        rec = {k: v for k, v in rec.items() if k not in _SETTLEMENT_KEYS}
        rec["settlement"] = settlement
        rec["settlement_sha256"] = _settlement_digest(settlement)
        settled_now.append(rec)

    if not settled_now:
        if not previously:
            print(f"[settle] no pending predictions for draw_key {args.draw_key}")
        return

    # Results first. The upsert is keyed on (draw_key, method), so a crash before the JSONL
    # rewrites converges on a re-run instead of duplicating rows into the evidence base.
    _upsert_results(ledger / RESULTS, rows)
    _rewrite_jsonl(ledger / SETTLED, kept_settled + settled_now)
    _rewrite_jsonl(ledger / PENDING, remaining)
    print(f"[settle] scored {len(settled_now)} prediction(s) for {args.draw_key}; "
          f"written to {ledger / RESULTS}")


def _upsert_results(path: Path, rows: list[dict]) -> None:
    """Write result rows, replacing any existing row carrying the same ``RESULT_KEY``.

    Settlement has to be safe to repeat: a crashed run, a retried workflow step, or an operator
    running the command twice must converge on one row per (draw, entrant) rather than stacking
    duplicates into a ledger whose whole purpose is to be counted.
    """
    if not rows:
        return
    # keep="last" also covers the corrupt case of a draw appearing in both ledger files:
    # one row per (draw, entrant) leaves the ledger, whatever reached this function.
    incoming = pd.DataFrame(rows).drop_duplicates(subset=list(RESULT_KEY), keep="last")
    if path.exists():
        existing = pd.read_csv(path)
        if not existing.empty and set(RESULT_KEY) <= set(existing.columns):
            keys = set(zip(incoming["draw_key"].astype(str), incoming["method"].astype(str)))
            keep = [
                (str(draw), str(method)) not in keys
                for draw, method in zip(existing["draw_key"], existing["method"])
            ]
            existing = existing[keep]
        incoming = pd.concat([existing, incoming], ignore_index=True)
    _atomic_write_text(path, incoming.to_csv(index=False))


def _load_payout_table(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    # utf-8-sig so payout tables authored on Windows (Notepad, PowerShell Out-File) are accepted;
    # it strips a leading BOM if present and is a no-op otherwise.
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    tiers = raw.get("tiers", raw.get("prizes", raw)) if isinstance(raw, dict) else raw
    if not isinstance(tiers, dict) or any(isinstance(v, (dict, list)) for v in tiers.values()):
        raise SystemExit(
            f"{path}: expected a JSON object mapping tiers to per-ticket payouts, either at the "
            'top level or under a "tiers" (or "prizes") key'
        )
    # "5_2" and "5+2" are both accepted; published breakdowns use either spelling.
    return {str(k).replace("_", "+"): float(v) for k, v in tiers.items()}


def _portfolio_prize(tier_counts: dict[str, int], payouts: dict[str, float]) -> float:
    return float(sum(count * payouts.get(tier, 0.0) for tier, count in tier_counts.items()))


def _permutation_pvalue(diffs: np.ndarray, iters: int = 10000, seed: int = 0) -> float:
    """One-sided p-value that mean(diffs) > 0 via sign-flip permutation (paired test)."""
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    observed = diffs.mean()
    count = 0
    for _ in range(iters):
        signs = rng.choice([-1.0, 1.0], size=len(diffs))
        if (signs * np.abs(diffs)).mean() >= observed:
            count += 1
    return (count + 1) / (iters + 1)


def _print_leaderboard(df: pd.DataFrame) -> None:
    """Per-method standings. Only meaningful once several entrants share settled draws."""
    methods = sorted(df["method"].dropna().unique())
    if len(methods) < 2:
        return
    print("competition standings (paired against the same control per draw)")
    print(f"  {'method':34}{'draws':>7}{'mean lift':>12}{'best main':>11}{'p(lift>0)':>11}")
    rows = []
    for method in methods:
        sub = df[df["method"] == method]
        diffs = sub["lift_mean_main"].to_numpy(dtype=float)
        rows.append((float(np.nanmean(diffs)), method, len(sub), float(sub["m_best_main"].mean()),
                     _permutation_pvalue(diffs, iters=2000)))
    for lift, method, draws, best, pval in sorted(rows, reverse=True):
        print(f"  {method:34}{draws:>7}{lift:>+12.4f}{best:>11.3f}{pval:>11.4f}")
    print("-" * 60)
    print("  Standings are evidence, not a ranking of predictive power. On a fair draw every")
    print("  entrant's true edge is zero, so early leads are luck until the window is long.")
    print("-" * 60)


def _monetary_mask(df: pd.DataFrame) -> pd.Series:
    """Rows that carry an official payout table, and can therefore be counted in money.

    Rows settled without a table hold NaN prizes on purpose; counting them as zero would drag
    every ROI aggregate down by exactly the draws nobody has prize data for. Legacy rows predate
    the explicit column and are recognised by their recorded payout-table hash.
    """
    if "payout_table_present" in df:
        return df["payout_table_present"].fillna(0).astype(float) > 0
    if "payout_table_sha256" in df:
        return df["payout_table_sha256"].fillna("").astype(str).str.len() > 0
    return pd.Series(False, index=df.index)


def cmd_report(args) -> None:
    ledger = _resolve_ledger(args, args.game)
    path = ledger / RESULTS
    if not path.exists():
        print("[report] no settled results yet. Record and settle some draws first.")
        return
    df = pd.read_csv(path)
    n = len(df)
    diffs = df["lift_mean_main"].to_numpy(dtype=float)
    method_best = df["m_best_main"].mean()
    control_best = df["c_best_main"].mean()
    mean_lift = float(np.nanmean(diffs))
    pval = _permutation_pvalue(diffs)
    jackpots = int(df["m_jackpot"].sum())
    # Money is summed over draws with an official payout table only. The hit-rate statistics
    # above use every settled draw; prizes cannot, because most draws have no prize data.
    paid = df[_monetary_mask(df)]
    portfolio_prize = float(paid.get("m_portfolio_prize", pd.Series(dtype=float)).sum(min_count=1))
    stake = float(paid.get("stake", pd.Series(dtype=float)).sum(min_count=1))
    net_return = float(paid.get("m_net_return", pd.Series(dtype=float)).sum(min_count=1))
    currency = str(df["currency"].dropna().iloc[-1]) if "currency" in df and df["currency"].notna().any() else ""

    print("=" * 60)
    print(f"Outcome report  ({ledger})")
    print("=" * 60)
    _print_leaderboard(df)
    print(f"settled draws              : {n}")
    print(f"  with official payouts    : {len(paid)}  (money below covers these draws only)")
    print(f"method mean best-main hits : {method_best:.3f}")
    print(f"control mean best-main hits: {control_best:.3f}")
    print(f"mean per-draw lift (method-control, mean-main): {mean_lift:+.4f}")
    print(f"one-sided permutation p (lift>0): {pval:.4f}")
    print(f"method jackpots            : {jackpots}")
    if len(paid) and not np.isnan(stake):
        print(f"tracked portfolio prizes   : {portfolio_prize:.2f} {currency}".rstrip())
        print(f"tracked stake              : {stake:.2f} {currency}".rstrip())
        print(f"tracked net return         : {net_return:+.2f} {currency}".rstrip())
    else:
        print("tracked money              : not reported (no settled draw carries an official "
              "payout table)")
    print("-" * 60)
    print("verdict:", _verdict(n, mean_lift, pval))
    print("=" * 60)


def _verdict(n: int, mean_lift: float, pval: float) -> str:
    if n < 30:
        return f"INSUFFICIENT DATA ({n} draws) — keep tracking; a 3-year window is the goal."
    if np.isnan(pval):
        return "INSUFFICIENT DATA — keep tracking."
    if mean_lift <= 0:
        return "NO EDGE vs random so far — trending toward PARK if this persists."
    if pval < 0.05:
        return "POSITIVE SIGNAL (p<0.05) — keep tracking to confirm stability over the full window."
    return "WEAK/INCONCLUSIVE positive — keep tracking; not yet distinguishable from luck."


# ------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------


def _add_game_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--preset", choices=sorted(PRESETS))
    p.add_argument("--main-n", type=int)
    p.add_argument("--main-k", type=int)
    p.add_argument("--star-n", type=int)
    p.add_argument("--star-k", type=int)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    rec = sub.add_parser("record", help="Generate and log a draw's portfolio (before the draw).")
    rec.add_argument(
        "--history",
        default=DEFAULT_HISTORY,
        help=f"history store, SQLite or CSV (default: {DEFAULT_HISTORY})",
    )
    rec.add_argument("--draw-key", required=True)
    rec.add_argument("--ledger", default=None, help="ledger directory (default: ledger/<game>)")
    rec.add_argument("--n-sets", type=int, default=20)
    rec.add_argument(
        "--methods",
        default="cooccurrence",
        help=(
            "comma-separated entrants, or 'all'. Choices: " + ", ".join(METHOD_CHOICES) +
            ". Every entrant is scored against the same shared random control for the draw."
        ),
    )
    rec.add_argument("--target-mode", choices=["observed", "predicted"], default="predicted")
    rec.add_argument("--pairing", choices=["cross", "main", "pooled"], default="cross")
    rec.add_argument("--window", type=int, default=26)
    rec.add_argument(
        "--ticket-price",
        type=float,
        default=None,
        help="official per-ticket price; enables prospective net-return reporting",
    )
    rec.add_argument(
        "--ticket-proof-sha256",
        default="",
        help="optional hash of a purchase receipt; the receipt itself is never stored",
    )
    _add_game_args(rec)
    rec.set_defaults(func=cmd_record)

    st = sub.add_parser("settle", help="Score a logged draw against the official result (after the draw).")
    st.add_argument("--ledger", default=None, help="ledger directory (default: ledger/<game>)")
    st.add_argument(
        "--game",
        choices=sorted(PRESETS),
        default=DEFAULT_GAME,
        help="game whose ledger is being settled; selects the default ledger path",
    )
    st.add_argument("--draw-key", required=True)
    st.add_argument("--actual-main", required=True, help="comma-separated main numbers")
    st.add_argument("--actual-stars", default="", help="comma-separated star numbers (optional)")
    st.add_argument(
        "--payout-table",
        default=None,
        help='optional JSON file mapping tiers such as "5+2" to official per-ticket payouts',
    )
    st.add_argument(
        "--payout-source",
        choices=["official", "approximate"],
        default="official",
        help=(
            "provenance of --payout-table. 'approximate' marks a static or estimated table, so "
            "ROI stays continuous without the ledger claiming precision it does not have"
        ),
    )
    st.add_argument("--currency", default=None, help="default: the game's currency, else EUR")
    st.add_argument(
        "--force",
        action="store_true",
        help="re-score a draw that is already settled, replacing its rows (use to correct a "
             "mistyped result)",
    )
    st.add_argument(
        "--outcome-source",
        choices=["self_reported", "operator_verified"],
        default="self_reported",
        help="provenance label for the submitted draw result and payout table",
    )
    st.set_defaults(func=cmd_settle)

    rep = sub.add_parser("report", help="Cumulative verdict across all settled draws.")
    rep.add_argument("--ledger", default=None, help="ledger directory (default: ledger/<game>)")
    rep.add_argument(
        "--game",
        choices=sorted(PRESETS),
        default=DEFAULT_GAME,
        help="game whose ledger is being reported; selects the default ledger path",
    )
    rep.set_defaults(func=cmd_report)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
