"""Command-line utilities for the local LottoBench data store.

The canonical journey is two commands::

    lottobench fetch --game euromillions
    lottobench benchmark --game euromillions

``fetch`` retrieves published history, validates it against the game's declared shape, and writes
it into the multi-game SQLite database with provenance. ``benchmark`` reads that database and runs
the registered providers forward-only at equal budget. Neither needs a CSV handed to it, and
neither needs anything beyond the base install.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from lotteries_core import dataset, registry, storage
from lotteries_core.evaluation import evaluate_forward
from lotteries_core.poi_g_artifacts import (
    build_poi_g_artifacts,
    settle_poi_g_artifacts,
)
from lotteries_core.sources import fetch_euromillions, fetch_netherlands
from lotteries_core.sources.euromillions import SOURCE_CHOICES

from .games import BACKLOG_GAMES, GAMES, game

DEFAULT_DB = Path("data/lotteries.db")


def _add_games(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--game", default="euromillions", help="supported game key")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="local SQLite database")


def _resolve(key: str, parser: argparse.ArgumentParser):
    try:
        return game(key)
    except KeyError as exc:
        parser.error(str(exc).strip('"'))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lottobench", description="Auditable benchmarking for lottery strategies"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("games", help="list supported game definitions")
    commands.add_parser("providers", help="list registered strategies and local availability")

    fetcher = commands.add_parser("fetch", help="retrieve published history into the database")
    _add_games(fetcher)
    fetcher.add_argument(
        "--source", default="auto",
        help="game-specific source (EuroMillions: " + ", ".join(SOURCE_CHOICES) + ")",
    )
    fetcher.add_argument("--from", dest="date_from", default=None, help="inclusive ISO start date")
    fetcher.add_argument("--to", dest="date_to", default=None, help="inclusive ISO end date")
    fetcher.add_argument("--no-cache", action="store_true", help="ignore any cached payload")
    fetcher.add_argument("--timeout", type=float, default=15.0)

    runner = commands.add_parser("benchmark", help="run providers against the stored history")
    _add_games(runner)
    runner.add_argument("--budget", type=int, default=25)
    runner.add_argument("--holdout", type=int, default=20)
    runner.add_argument("--seed", type=int, default=1234)
    runner.add_argument("--out", type=Path, default=None, help="write the JSON summary here")

    importer = commands.add_parser("import-csv", help="import a legacy history CSV")
    importer.add_argument("csv", type=Path)
    _add_games(importer)

    exporter = commands.add_parser("export-csv", help="export a history for legacy tools")
    exporter.add_argument("csv", type=Path)
    _add_games(exporter)

    poi_export = commands.add_parser(
        "poi-export",
        help="seal an extensive POI-G candidate subset and its fixed-budget selection",
    )
    _add_games(poi_export)
    poi_export.add_argument("--draw-key", required=True, help="target draw identifier")
    poi_export.add_argument("--subset-size", type=int, default=500)
    poi_export.add_argument("--budget", type=int, default=25)
    poi_export.add_argument("--window", type=int, default=26)
    poi_export.add_argument("--pairing", choices=["cross", "main", "pooled"], default="cross")
    poi_export.add_argument("--seed", type=int, default=0)
    poi_export.add_argument(
        "--created-utc",
        default=None,
        help="optional ISO-8601 seal time; defaults to the current UTC time",
    )
    poi_export.add_argument("--out", type=Path, required=True, help="artifact output directory")

    poi_settle = commands.add_parser(
        "poi-settle", help="settle a sealed POI-G fixed-budget selection"
    )
    poi_settle.add_argument("bundle", type=Path, help="directory produced by poi-export")
    poi_settle.add_argument("--actual-main", required=True, help="comma-separated main numbers")
    poi_settle.add_argument(
        "--actual-auxiliary", default="", help="comma-separated star/auxiliary numbers"
    )
    poi_settle.add_argument(
        "--payout-table", type=Path, default=None, help="optional JSON tier-to-payout mapping"
    )
    poi_settle.add_argument("--ticket-price", type=float, default=None)
    poi_settle.add_argument("--currency", default="EUR")
    poi_settle.add_argument("--outcome-source", default="official")
    poi_settle.add_argument("--payout-source", default="official")
    poi_settle.add_argument("--settled-utc", default=None)
    return parser


def _cmd_games() -> int:
    for key, definition in GAMES.items():
        print(f"{key:20} {definition.country_code:2}  {definition.display_name}")
    noun = "game" if len(GAMES) == 1 else "games"
    print(f"\n{len(GAMES)} {noun} supported end to end (fetch, store, benchmark, settle).")
    if BACKLOG_GAMES:
        print("\nDefined but not yet supported -- see docs/wiki/Backlog.md:")
        for key, entry in BACKLOG_GAMES.items():
            print(f"  {key:20} {entry.country_code:2}  {entry.display_name}")
    return 0


def _cmd_providers() -> int:
    ready = set(registry.available())
    print(f"{'identity':32} {'version':10} {'status':28} implementation family")
    for name, spec in registry.PROVIDERS.items():
        if name in ready:
            status = "available"
        elif spec.install_extra:
            status = f"install [{spec.install_extra}] extra"
        else:
            status = "optional dependency missing"
        print(f"{name:32} {spec.version:10} {status:28} {spec.implementation}")
    families = {spec.implementation for spec in registry.PROVIDERS.values()}
    print(
        f"\n{len(registry.PROVIDERS)} selectable entrants backed by "
        f"{len(families)} implementation families; signal-off controls are included."
    )
    return 0


def _cmd_fetch(args, parser) -> int:
    definition = _resolve(args.game, parser)
    if definition.key == "euromillions":
        frame = fetch_euromillions(
            source=args.source,
            date_from=args.date_from,
            date_to=args.date_to,
            use_cache=not args.no_cache,
            timeout=args.timeout,
        )
        source_label = args.source
    elif definition.key == "nl-lotto":
        if args.source != "auto":
            parser.error("nl-lotto currently supports only --source auto (official operator API)")
        frame = fetch_netherlands(
            date_from=args.date_from, date_to=args.date_to, timeout=args.timeout
        )
        source_label = "official-operator-api"
    else:  # pragma: no cover - every GAMES entry must have an explicit adapter above
        parser.error(f"no retrieval adapter registered for {definition.key}")
    if frame.empty:
        parser.error("retrieval returned no draws; refusing to write an empty history")

    storage.write_history(args.db, frame, game=definition.key)
    # Describe what was actually stored, not what was retrieved: the digest then covers the rows a
    # later benchmark will really read, so a silent write problem cannot pass provenance checks.
    described = dataset.describe(
        args.db, game=definition.key, source=f"lottobench.fetch:{source_label}"
    )
    storage.write_metadata(args.db, game=definition.key, metadata=asdict(described))

    first = frame["draw_date"].min().date()
    last = frame["draw_date"].max().date()
    print(f"{definition.display_name}: {len(frame)} draws {first} -> {last}")
    print(f"Stored in {args.db} (source: {source_label}, digest {described.content_sha256[:12]})")
    print(f"Next: lottobench benchmark --game {definition.key} --db {args.db}")
    return 0


def _cmd_benchmark(args, parser) -> int:
    definition = _resolve(args.game, parser)
    # is_database() only inspects the suffix, so an absent path still looks like a database.
    # Check for the file itself, or the user gets a sqlite "no such table" instead of the fix.
    if not args.db.exists() or not storage.is_database(args.db):
        parser.error(
            f"no LottoBench database at {args.db}. Run: "
            f"lottobench fetch --game {definition.key} --db {args.db}"
        )
    history = storage.read_history(args.db, game=definition.key)
    if history.empty:
        parser.error(
            f"no stored draws for {definition.key}. Run: "
            f"lottobench fetch --game {definition.key} --db {args.db}"
        )
    if len(history) <= args.holdout:
        parser.error(
            f"holdout {args.holdout} needs more than {len(history)} stored draws; "
            "fetch a longer history or lower --holdout"
        )

    providers = [registry.create(name) for name in registry.available()]
    summary = evaluate_forward(
        history,
        definition.spec,
        providers,
        budget=args.budget,
        holdout=args.holdout,
        seed=args.seed,
    )
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 0


def _require_history(args, parser):
    definition = _resolve(args.game, parser)
    if not args.db.exists() or not storage.is_database(args.db):
        parser.error(
            f"no LottoBench database at {args.db}. Run: "
            f"lottobench fetch --game {definition.key} --db {args.db}"
        )
    history = storage.read_history(args.db, game=definition.key)
    if history.empty:
        parser.error(
            f"no stored draws for {definition.key}. Run: "
            f"lottobench fetch --game {definition.key} --db {args.db}"
        )
    return definition, history


def _cmd_poi_export(args, parser) -> int:
    definition, history = _require_history(args, parser)
    created_utc = args.created_utc or datetime.now(timezone.utc).isoformat()
    bundle = build_poi_g_artifacts(
        history,
        definition.spec,
        draw_key=args.draw_key,
        subset_size=args.subset_size,
        budget=args.budget,
        window=args.window,
        pairing=args.pairing,
        seed=args.seed,
        created_utc=created_utc,
        evidence_kind="prospective",
        repo_dir=Path.cwd(),
    )
    paths = bundle.write(args.out)
    print(
        f"POI-G: {len(bundle.candidates)} ranked candidates; "
        f"{len(bundle.selection)} sealed for ROI"
    )
    print(f"Prediction: {bundle.manifest['prediction_id']}")
    print(f"Artifacts: {paths['manifest'].parent}")
    return 0


def _read_payout_table(path: Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    tiers = raw.get("tiers", raw.get("prizes", raw)) if isinstance(raw, dict) else raw
    if not isinstance(tiers, dict) or any(isinstance(value, (dict, list)) for value in tiers.values()):
        raise ValueError(f"{path}: expected a JSON object mapping tiers to payouts")
    return {str(key): float(value) for key, value in tiers.items()}


def _cmd_poi_settle(args, parser) -> int:
    try:
        payouts = _read_payout_table(args.payout_table)
        result = settle_poi_g_artifacts(
            args.bundle,
            actual_main=args.actual_main,
            actual_auxiliary=args.actual_auxiliary,
            payout_table=payouts,
            ticket_price=args.ticket_price,
            currency=args.currency,
            outcome_source=args.outcome_source,
            payout_source=args.payout_source,
            settled_utc=args.settled_utc,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    roi = result["realized_roi"]
    roi_text = "not available (no complete payout evidence)" if roi is None else f"{roi:+.6f}"
    print(f"Settled {result['prediction_id']} for {result['draw_key']}")
    print(f"Realized ROI: {roi_text}")
    print(f"Result: {args.bundle / 'poi_g_settlement.json'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "games":
        return _cmd_games()
    if args.command == "providers":
        return _cmd_providers()
    if args.command == "fetch":
        return _cmd_fetch(args, parser)
    if args.command == "benchmark":
        return _cmd_benchmark(args, parser)
    if args.command == "poi-export":
        return _cmd_poi_export(args, parser)
    if args.command == "poi-settle":
        return _cmd_poi_settle(args, parser)

    definition = _resolve(args.game, parser)
    if args.command == "import-csv":
        rows = storage.import_csv(args.csv, args.db, game=definition.key)
        print(f"Imported {rows} rows into {args.db}")
        return 0
    rows = storage.export_csv(args.db, args.csv, game=definition.key)
    print(f"Exported {rows} rows to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
