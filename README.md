# LottoBench

Experimental, auditable benchmarking for lottery strategies.

LottoBench compares strategy providers at equal ticket budgets using forward-only evaluation,
coverage, diversity, and explicit ROI metrics. It includes a local SQLite history store, provenance
checks, a prospective outcome ledger, and an optional local HTTP API.

> **Experimental software — no warranty or advice.** LottoBench does not predict winning numbers,
> improve the mechanical odds of a fair draw, place wagers, sell tickets, or provide financial or
> gambling advice. Lottery participation is ordinarily negative-sum. See [NOTICE.md](NOTICE.md),
> [LICENSE](LICENSE), and [Experimental Use and Liability](docs/wiki/Experimental-Use-and-Liability.md).

## Quick start

Python 3.10–3.14 is supported by the core package.

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux:       source .venv/bin/activate
python -m pip install --upgrade pip
make setup PYTHON=python
make test PYTHON=python
```

`make setup` matters: installing `pytest` alone does not install LottoBench, pandas, NumPy, or the
optional API test dependencies. `make doctor PYTHON=python` reports missing setup requirements with
an actionable message.

## Benchmark the providers

The canonical registry exposes twelve provider identities, including named strategies, baselines,
and ablation controls. Some identities intentionally share an implementation so their experimental
claim can be compared with its signal-off control.

```bash
make providers
make benchmark HISTORY=data/euromillions.csv GAME=euromillions BUDGET=25 HOLDOUT=20
```

The benchmark is forward-only and gives every provider the same ticket budget. Reported expected
ROI remains negative for ordinary fair-draw lottery play; a less-negative value is not a promise of
profit or evidence of predictive power.

## Library

```python
import numpy as np
import pandas as pd
import lottobench

game = lottobench.game("uk-lotto")
provider = lottobench.create("frequency")
history = pd.DataFrame({
    "ball_1": [1, 2], "ball_2": [8, 9], "ball_3": [19, 20],
    "ball_4": [34, 35], "ball_5": [47, 48], "ball_6": [52, 53],
})
provider.fit(history, game.spec)
result = provider.propose(game.spec, budget=5, rng=np.random.default_rng(7))
print(result.tickets)
```

The stable package identity is `lottobench`. The lower-level `lotteries_core` namespace remains
available for provider authors and compatibility.

## Local data and ROI evidence

```bash
lottobench games
lottobench import-csv history.csv --game euromillions --db data/lotteries.db
lottobench export-csv exported.csv --game euromillions --db data/lotteries.db
make roi-report LEDGER=ledger/euromillions
make roi-export LEDGER=ledger/euromillions ROI_OUT=roi-benchmark.json
```

Runtime databases and exported histories are not distributed in the wheel. Realized user ROI is
reported only from settled prospective ledger entries; absent settled draws, LottoBench says there
is no result instead of substituting a backtest.

ROI exports contain model/configuration versions, dataset and prediction provenance, actual stake,
payout and realized ROI, plus integrity hashes. They exclude tickets, receipts and user/device
identifiers. Compare bundles from multiple installations with:

```bash
lotto-roi validate roi-benchmark.json
lotto-roi compare submission-a.json submission-b.json
```

See [Versioned ROI Benchmark](docs/VERSIONED_ROI_BENCHMARK.md) for the evolution contract.

## API

```bash
python -m pip install -e ".[api]"
lotto-serve
```

The API runs locally by default and has no payment, wagering, or ticket-purchase capability.

## Core versus experiments

The PyPI product is deliberately small:

- `lottobench/` — public library and CLI
- `lotteries_core/` — provider protocol, benchmark, ROI, storage, provenance, and local API
- `tests/` — tests for the shipped packages
- `experiments/` — preserved EuroMillions HMM/GARCH/branch work, agents, scrapers, and legacy tests

Experiments are not shipped to PyPI and do not block the normal user installation:

```bash
make setup-experiments PYTHON=python
make test-experiments PYTHON=python
```

See [experiments/README.md](experiments/README.md) before running historical research code.

## Release validation

```bash
make check PYTHON=python
```

This runs core linting, tests, an offline end-to-end journey, and wheel/source-distribution checks.
Publishing instructions are in [docs/PUBLISHING.md](docs/PUBLISHING.md); limitations are maintained
in the [wiki](docs/wiki/Home.md).
