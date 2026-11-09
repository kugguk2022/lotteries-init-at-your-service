"""Run the original main-pair PoI benchmark from a validated local database or CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

from lotteries_core import storage
from lotteries_core.dataset import content_digest
from lotteries_core.poi_benchmark import build_poi_benchmark
from lottobench.games import GAMES
from scripts.build_space_profiles import _next_draw_date


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--game", choices=tuple(GAMES), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--holdout", type=int, default=52)
    args = parser.parse_args()
    history = storage.read_history(args.history, game=args.game)
    result = build_poi_benchmark(
        history, GAMES[args.game].spec, output_directory=args.out,
        target_draw_date=_next_draw_date(args.game, str(history.draw_date.max())[:10]),
        snapshot_sha256=content_digest(history), holdout=args.holdout,
    )
    print(f"{args.game}: {result['holdout']} held-out draws; ROI unassessed")


if __name__ == "__main__":
    main()
