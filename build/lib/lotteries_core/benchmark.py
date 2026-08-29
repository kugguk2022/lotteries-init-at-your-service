"""CLI: forward-only, equal-budget benchmark of single providers vs coordinated aggregation.

Example
-------
    python -m lotteries_core.benchmark --history data/euromillions.csv \
        --game euromillions --budget 25 --holdout 20 \
        --out outputs/euromillions/distributed_inference_benchmark.json

The output is a reproducible JSON summary (also readable by the regression test in
``tests/test_core_inference.py``). It reports coverage, diversity, and unpopularity-adjusted ROI
per provider and for the aggregated portfolio. It does NOT tell you to play, and moves no money.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import registry
from .evaluation import evaluate_forward
from .protocol import GameSpec
from .providers import (
    CooccurrenceLevelSetProvider,
    FrequencyProvider,
    ParallaxGuardProvider,
    PerronFrobeniusProvider,
    UnpopularityProvider,
)

_GAMES = {
    "euromillions": GameSpec.euromillions,
    "totoloto": GameSpec.totoloto,
    "eurodreams": GameSpec.eurodreams,
}


def build_providers(
    include_ml: bool,
    include_cooccurrence: bool = False,
    include_spectral: bool = False,
    include_parallax: bool = False,
):
    providers = [FrequencyProvider(), UnpopularityProvider()]
    if include_cooccurrence:
        providers.append(CooccurrenceLevelSetProvider())
    if include_parallax:
        # Guarded and ablation together: identical machinery, signal on vs off, so any gap is the
        # replicated residual and anything they share is the portfolio optimiser.
        providers.append(ParallaxGuardProvider(mode="guarded"))
        providers.append(ParallaxGuardProvider(mode="ablation"))
    if include_spectral:
        # Both orientations enter the field (the ranking may be informative in either direction),
        # plus the sampler-only ablation so a win can be attributed to the spectral signal or denied it.
        providers.append(PerronFrobeniusProvider(orientation="affinity"))
        providers.append(PerronFrobeniusProvider(orientation="contrarian"))
        providers.append(PerronFrobeniusProvider(orientation="uniform"))
    if include_ml:
        from .providers import load_ml_ensemble

        providers.append(load_ml_ensemble()())
    return providers


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--history", required=True, help="Normalised draw-history CSV")
    ap.add_argument("--game", choices=sorted(_GAMES), default="euromillions")
    ap.add_argument("--budget", type=int, default=25)
    ap.add_argument("--holdout", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--with-ml", action="store_true", help="Include the GLM+GBM(+DL) ensemble provider")
    ap.add_argument(
        "--with-cooccurrence",
        action="store_true",
        help="Include the owner's forward-only pair-co-occurrence level-set provider",
    )
    ap.add_argument(
        "--with-spectral",
        action="store_true",
        help="Include the Perron-Frobenius (PageRank) co-occurrence providers, both orientations",
    )
    ap.add_argument(
        "--with-parallax",
        action="store_true",
        help="Include the Parallax Guard providers (guarded + signal-off ablation)",
    )
    ap.add_argument(
        "--all-providers",
        action="store_true",
        help="Run every provider currently available in the canonical registry",
    )
    ap.add_argument("--out", default=None, help="Optional path to write the JSON summary")
    args = ap.parse_args(argv)

    history = pd.read_csv(args.history)
    spec = _GAMES[args.game]()
    every = args.all_providers
    providers = (
        [registry.create(name) for name in registry.available()]
        if every
        else build_providers(
            args.with_ml,
            args.with_cooccurrence,
            args.with_spectral,
            args.with_parallax,
        )
    )
    summary = evaluate_forward(
        history, spec, providers, budget=args.budget, holdout=args.holdout, seed=args.seed
    )
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
