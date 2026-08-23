# LottoBench

**Auditable benchmarking for lottery strategies.**

LottoBench is an experimental, local Python toolkit for comparing lottery ticket-selection
strategies at equal budget. It provides reproducible game definitions, strategy providers,
fixed-seed portfolios, forward-only evaluation, coverage metrics, a multi-game SQLite store, and an
optional read-only HTTP API.

It does not predict winning numbers, improve the mechanical odds of a fair draw, place wagers, sell
tickets, or provide financial or gambling advice. Lottery participation is ordinarily negative-sum.
See the bundled `NOTICE.md` and `LICENSE` before relying on or redistributing the software.

## Install

```bash
pip install lottobench==0.1.0a1
```

Optional local API:

```bash
pip install "lottobench[api]==0.1.0a1"
lotto-serve
```

## Library

```python
import numpy as np
import pandas as pd
import lottobench

definition = lottobench.game("uk-lotto")
provider = lottobench.create("frequency")
provider.fit(pd.DataFrame({
    "ball_1": [1, 2], "ball_2": [8, 9], "ball_3": [19, 20],
    "ball_4": [34, 35], "ball_5": [47, 48], "ball_6": [52, 53],
}))
result = provider.propose(definition.spec, budget=5, rng=np.random.default_rng(7))
print(result.tickets)
```

The stable public identity is `lottobench`. The lower-level `lotteries_core` namespace remains
available for compatibility and research extensions.

## Local data

```bash
lottobench games
lottobench import-csv history.csv --game euromillions --db data/lotteries.db
lottobench export-csv exported.csv --game euromillions --db data/lotteries.db
```

SQLite databases and exported CSV histories are runtime data and are not included in the package.

## Current game definitions

- Denmark: Danske Lotto
- Germany: LOTTO 6aus49
- United Kingdom: Lotto
- Netherlands: Lotto
- Sweden: Lotto
- EuroMillions

Game definitions do not imply that every country already has an automated results adapter. Always
verify game rules and official results with the relevant operator.

## Experimental status

Version `0.1.0a1` is an alpha. APIs, data schemas, strategies, and country support may change. A
passing benchmark or a high metric value is not evidence of future draw prediction unless the exact
metric, holdout, data cutoff, ticket budget, baseline, and leakage controls are supplied and
reproducible.
