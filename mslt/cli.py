"""Command-line utilities for the local MSLT data store."""

from __future__ import annotations

import argparse
from pathlib import Path

from lotteries_core import storage

from .games import GAMES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mslt", description="Multi-Strategy Lottery Tests")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("games", help="list supported national game definitions")

    importer = commands.add_parser("import-csv", help="import a legacy history CSV")
    importer.add_argument("csv", type=Path)
    importer.add_argument("--db", type=Path, default=Path("data/lotteries.db"))
    importer.add_argument("--game", required=True)

    exporter = commands.add_parser("export-csv", help="export a history for legacy tools")
    exporter.add_argument("csv", type=Path)
    exporter.add_argument("--db", type=Path, default=Path("data/lotteries.db"))
    exporter.add_argument("--game", required=True)

    args = parser.parse_args(argv)
    if args.command == "games":
        for key, definition in GAMES.items():
            print(f"{key:20} {definition.country_code:2}  {definition.display_name}")
        return 0
    if args.game not in GAMES:
        parser.error(f"unknown game {args.game!r}; choose from {sorted(GAMES)}")
    if args.command == "import-csv":
        rows = storage.import_csv(args.csv, args.db, game=args.game)
        print(f"Imported {rows} rows into {args.db}")
        return 0
    rows = storage.export_csv(args.db, args.csv, game=args.game)
    print(f"Exported {rows} rows to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
