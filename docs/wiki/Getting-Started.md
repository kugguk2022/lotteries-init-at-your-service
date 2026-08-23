# Getting Started

## Core development

Python 3.10–3.14 is supported.

```bash
git clone https://github.com/kugguk2022/lotteries-init-at-your-service.git
cd lotteries-init-at-your-service
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux:       source .venv/bin/activate
python -m pip install --upgrade pip
make setup PYTHON=python
make test PYTHON=python
```

Installing `pytest` by itself is insufficient. `make setup` installs the editable LottoBench
package, NumPy, pandas, API dependencies, and development tooling. Use `make doctor PYTHON=python`
to diagnose the selected interpreter.

## Run the benchmark

```bash
make providers
make benchmark HISTORY=data/euromillions.csv GAME=euromillions BUDGET=25 HOLDOUT=20
```

`--all-providers` uses the canonical registry. Evaluation is forward-only and equal-budget.
Coverage and expected ROI are reported honestly; they are not claims that a fair draw is predictable.

## Local SQLite data

```bash
lottobench games
lottobench import-csv history.csv --game euromillions --db data/lotteries.db
lottobench export-csv exported.csv --game euromillions --db data/lotteries.db
```

The database is runtime data and is excluded from packages and normal Git changes.

## Prospective ROI

```bash
lotto-track report --ledger ledger/euromillions
# equivalent:
make roi-report LEDGER=ledger/euromillions
```

Realized ROI exists only after a prediction was recorded before a draw and later settled against an
official result and payout table. Backtests are never substituted for missing prospective outcomes.

## Local API

```bash
lotto-serve
# Open http://127.0.0.1:8007/docs
```

The API is local/read-mostly and never places wagers or moves money.

## Historical experiments

The old EuroMillions, Totoloto, EuroDreams, HMM, GARCH, branch, and agent programs are preserved
under `experiments/` and are not installed from PyPI.

```bash
make setup-experiments PYTHON=python
make test-experiments PYTHON=python
cd experiments
python -m euromillions.get_draws --help
```

See [Experiments](../../experiments/README.md) for the support boundary.
