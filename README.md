# LottoBench

Experimental, auditable benchmarking for lottery strategies.

LottoBench compares strategy providers at equal ticket budgets using forward-only evaluation,
coverage, diversity, and explicit ROI metrics. It includes a local SQLite history store, provenance
checks, a prospective outcome ledger, and an optional local HTTP API.

> **Experimental software — no warranty or advice.** LottoBench does not predict winning numbers,
> improve the mechanical odds of a fair draw, place wagers, sell tickets, or provide financial or
> gambling advice. Lottery participation is ordinarily negative-sum. See [NOTICE.md](NOTICE.md),
> [LICENSE](LICENSE), and [Experimental Use and Liability](docs/wiki/Experimental-Use-and-Liability.md).

## Best options today — ranked by objective

There is **no realized-ROI winner yet**: the public ledger has no settled records containing both
actual cost and payout. LottoBench therefore does not turn backtests or modeled ROI into a claimed
user-return ranking. This table is the current evidence-based decision guide:

| Rank | User objective | Current option | Evidence and limitation |
|---:|---|---|---|
| 1 | Preserve money | **Do not play** | The only option that avoids the expected loss of a negative-sum lottery. It is a financial baseline, not a provider. |
| 2 | Contribute trustworthy realized ROI | **`frequency`** | Fast, simple reference control. It has no claimed predictive edge and gives future provider versions a stable baseline. |
| 3 | Optimize modeled payout conditional on winning | **`unpopularity`** | Best stored modeled ROI/ticket, **−0.7141**, and unpopularity lift, **1.1890**. This does not improve win probability and is not realized user ROI. |
| 4 | Maximize portfolio coverage | **`parallax_guard_ablation`** | Best stored pair coverage, **0.2039**, and number coverage, **1.000**. It matched `parallax_guard` exactly, so the measured value came from portfolio construction rather than its experimental signal. |
| Unranked | Test the owner's GINGERM hypothesis | **`gingerm`** | Retained as an experimental entrant. Its stored short holdout is not comparable with the 40-draw table, and it has no settled user-ROI evidence yet. |

Snapshot: EuroMillions, equal budget of 25 tickets, 40-draw forward holdout, seed 1234, except
GINGERM's separately reported five-draw run. See the exact artifact and caveats in
[Methods and Findings](docs/wiki/Methods-and-Findings.md). All stored modeled ROI values remain
negative. Rankings should move only when versioned prospective evidence justifies it.

### Fastest useful workflow for ROI contributors

```bash
# Published package (after the first PyPI release)
python -m pip install lottobench
lottobench providers

# Current source checkout (works before the first PyPI release)
python -m pip install -e .
python -m lottobench.cli providers

# Before the draw: record a fixed-budget portfolio and its model/data provenance
lotto-track record \
  --history data/euromillions.csv --preset euromillions \
  --draw-key YYYY-MM-DD --methods frequency \
  --n-sets 10 --ticket-price YOUR_ACTUAL_PRICE --ledger ledger/euromillions

# After the draw: settle against official numbers and an official payout table
lotto-track settle \
  --ledger ledger/euromillions --draw-key YYYY-MM-DD \
  --actual-main 1,2,3,4,5 --actual-stars 1,2 \
  --payout-table payouts.json --outcome-source operator_verified

# Read and export provider/model-version ROI evidence
lotto-roi report --ledger ledger/euromillions
lotto-roi export --ledger ledger/euromillions --out roi-benchmark.json
lotto-roi validate roi-benchmark.json
```

Use the actual ticket price and official results/payouts for the relevant draw. Export is explicit,
offline by default, and contains no tickets, receipt contents, user identity, or device identifier.

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

The canonical registry exposes **11 distinct benchmark entrants backed by eight implementation
families**. Named aliases are not duplicated as competitors; ablations remain because they are
necessary signal-off controls.
The identities include named strategies (`gingerm`, `spectral_contrarian`, and `parallax`), technical
names, baselines, and ablation controls. For example, `gingerm` is backed by the co-occurrence
implementation and `spectral_contrarian` by the contrarian Perron–Frobenius implementation; they are
present even though there are no misleading duplicate source files bearing those names.

The prospective ledger currently containing six entrants is a historical cohort, not the provider
registry. Use `--methods all` for a new tracking cohort containing every locally available identity.

The temporal-model candidates are named after their mechanisms: `garch_markov_branch` and
`sequence_transformer`. The latter requires the `ml` extra; neither is part of GINGERM.

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
