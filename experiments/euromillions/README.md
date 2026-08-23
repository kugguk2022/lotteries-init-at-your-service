# EuroMillions (stable core)

This folder hosts the stable, typed EuroMillions API plus two labs.

- **Stable**: `get_draws.py` (fetch + normalize), `guess.py` (ticket parsing/scoring), `schema.py` (Pandera schema), `__init__.py` (public exports + `load_history` helper).
- **Lab**: `roi.py` (EV gating, walk-forward bankroll), `grok.py` (modelling sandbox).
- **Deprecated**: `*.r` scripts (kept for provenance only).

## Quickstart

```bash
# Full history with cache + dedupe
python -m euromillions.get_draws --out data/euromillions.csv --append

# Year window
python -m euromillions.get_draws --from 2023-01-01 --to 2024-12-31 --out data/euromillions_2023_2024.csv
```

Programmatic:

```python
from euromillions import EuroMillionsGuess, evaluate_guess, load_history, normalize
from euromillions.get_draws import fetch_and_normalize

df = fetch_and_normalize().dataframe  # network + cache
# or: df = normalize(open("raw.csv").read())  # custom source

guess = EuroMillionsGuess([1, 2, 20, 30, 40], [3, 11])
ball_hits, star_hits = evaluate_guess(df.iloc[-1], guess)
```

## Inputs & Outputs

- Inputs: MerseyWorld CSV endpoint (`resultsArchive.php?format=csv`), optional date filters.
- Outputs: normalized CSV with columns `draw_date, ball_1..ball_5, star_1, star_2` plus Pandera validation for ranges and deduplication on `draw_date`.

## Labs (baseline reproduction)

- `roi.py`: run `python -m euromillions.roi --help` (in-progress). Use `data/euromillions.csv` as input; outputs walk-forward bankroll metrics under `roi_runs/`.
- `grok.py`: see `labs/README.md` for data expectations; run `python -m euromillions.grok` to reproduce the baseline transformer experiment.
