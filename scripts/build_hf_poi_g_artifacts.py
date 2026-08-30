"""Build the additive POI-G candidate configs for the Hugging Face dataset mirror.

This command deliberately targets only ``publishing/huggingface/data``.  It does not touch the
Space bundle, Kaggle bundle, GitHub workflows, schedules, or deployment triggers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from lotteries_core.poi_g_artifacts import build_poi_g_artifacts, validate_poi_g_artifacts
from lotteries_core.protocol import GameSpec
from lotteries_core.roi import JackpotModel

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY = ROOT / "publishing" / "huggingface" / "data" / "synthetic_history.csv"
DEFAULT_OUT = ROOT / "publishing" / "huggingface" / "data"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--subset-size", type=int, default=500)
    parser.add_argument("--budget", type=int, default=5)
    args = parser.parse_args(argv)

    history = pd.read_csv(args.history)
    bundle = build_poi_g_artifacts(
        history,
        GameSpec("synthetic-community", main_n=12, main_k=4, star_n=4, star_k=1),
        draw_key="synthetic-community-next",
        subset_size=args.subset_size,
        budget=args.budget,
        window=26,
        pairing="cross",
        seed=20260828,
        created_utc="",
        jackpot=JackpotModel(jackpot=1_000.0, ticket_price=2.0, n_other_tickets=10_000.0),
        benchmark_version="3.0.0",
        evidence_kind="deterministic_synthetic_snapshot",
        repo_dir=ROOT,
    )
    paths = bundle.write(args.out)
    validate_poi_g_artifacts(args.out)
    print(
        f"{paths['candidates']}: {len(bundle.candidates)} candidates; "
        f"{len(bundle.selection)} fixed-budget selections"
    )
    print(paths["manifest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
