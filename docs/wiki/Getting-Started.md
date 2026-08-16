# Getting Started

Every command on this page was executed against the repository before being written down. Commands
are marked:

- **[verified]** — run successfully, offline, exactly as shown.
- **[network]** — needs internet; falls back to cached/bundled data.
- **[broken]** — currently fails. Listed anyway, because silently omitting them is how documentation
  starts lying. Details in [Current State](Current-State.md).

## Install

```bash
git clone https://github.com/kugguk2022/lotteries-init-at-your-service
cd lotteries-init-at-your-service
python -m venv .venv
.\.venv\Scripts\activate      # Windows
# source .venv/bin/activate   # macOS / Linux
python -m pip install -U pip
pip install -e ".[dev]"
```

Python 3.10+ is required. The optional `ml` extra (`pip install -e ".[dev,ml]"`) adds `torch` and
`xgboost`; everything degrades gracefully without them.

## Verify your install **[verified]**

Run the maintained core, which is what CI gates:

```bash
pytest -q tests/test_core_inference.py tests/test_outcome_tracking.py \
          tests/test_benchmark_regression.py tests/test_causal_poi.py
```

Expected: **24 passed**.

> Do **not** start with `make test`. It runs `ruff check .` across the whole repository and currently
> reports 61 errors, so it fails for reasons unrelated to your install. See
> [Current State](Current-State.md).

## The main thing this repo does **[verified]**

Compare ticket-selection strategies forward-only, at equal budget, on the metrics that are actually
controllable (coverage and jackpot-sharing) rather than on an assumption of predictive power:

```bash
python -m lotteries_core.benchmark \
    --history euromillions/euromillions_2016_2025.csv \
    --game euromillions --budget 25 --holdout 40 \
    --with-spectral \
    --out outputs/euromillions/distributed_inference_benchmark.json
```

Add `--with-cooccurrence` to include the level-set generator. Be warned: it enumerates all 2,118,760
main combinations against each of 66 star combinations *per holdout step*, so use a small `--holdout`
(5 or so) unless you are prepared to wait. Add `--with-ml` for the GLM + gradient-boosting popularity
ensemble.

How to read the output is covered in [Methods and Findings](Methods-and-Findings.md).

## Generate candidate tickets **[verified]**

Baseline frequency-weighted sampler:

```bash
python -m euromillions.infer --history euromillions/euromillions_2016_2025.csv \
    --n 5 --out runs/euromillions_candidates.csv
```

Co-occurrence level-set generator (the repository's own method):

```bash
python -m lotteries_core.likely_set_generator \
    --history euromillions/euromillions_2016_2025.csv \
    --preset euromillions --target-mode predicted --top-n 25 \
    --out runs/level_set.csv
```

Also available as the console script `lotto-likely-sets`.

## Record a prospective prediction **[verified]**

This is the actual experiment — see [Outcome Tracking](Outcome-Tracking.md) for the rules that make
it valid.

```bash
# BEFORE the draw. History must contain completed draws only.
lotto-track record --history euromillions/euromillions_2016_2025.csv \
    --preset euromillions --draw-key 2026-08-18 --ledger ledger/euromillions --n-sets 20

# AFTER the draw.
lotto-track settle --ledger ledger/euromillions --draw-key 2026-08-18 \
    --actual-main 4,17,23,38,45 --actual-stars 3,9

# Any time.
lotto-track report --ledger ledger/euromillions
```

## Refresh the draw history **[network]**

```bash
python -m euromillions.get_draws --out data/euromillions.csv --append
```

This tries several upstream sources in order and caches responses under `.cache/euromillions`. Some
upstream sources have drifted (one returns 404, another the wrong content type), but the archive
source currently succeeds.

`--allow-stale` is documented as the offline fallback, but its final bundled-CSV fallback is
**[broken]** — see [Current State](Current-State.md). If you are offline, pass the bundled file
directly with `--history euromillions/euromillions_2016_2025.csv` instead.

> The bundled history ends **2025-08-12**. Refresh it before recording any prospective prediction.

## Run everything end to end **[network]**

```bash
python run_all.py --n-candidates 200 --skip-fetch
```

Fetches (unless `--skip-fetch`) all three lotteries, generates candidates, and runs a walk-forward
baseline evaluation with a permutation test. Windows one-click equivalents exist as
`start_euromillions.bat`, `start_totoloto.bat`, `start_eurodreams.bat`.

## Known-failing entry points **[broken]**

- `make test` — repository-wide lint failures (61) and two uncollectable test modules.
- `pytest -q` with no arguments — fails at collection on `tests/test_infer.py` and
  `tests/test_end_to_end_paths.py`.
- `from euromillions import EuroMillionsGuess, evaluate_guess, normalize` — the package-level API
  advertised in `CONTRIBUTING.md` does not exist.
- `python -m euromillions.roi` — documented as planned, not implemented; the CLI errors by design.
