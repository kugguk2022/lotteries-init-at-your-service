# EuroMillions Agent Lab

This folder is experimental. It contains a stacked agent/discriminator/grok/mixer pipeline for EuroMillions analysis.

## Entry points

- `python euromillions_agent/lotto_lab.py --csv data/euromillions.csv --mode all --outdir lotto_lab_out --debug`
- `python euromillions_agent/fetch_prizes.py --out data/prizes.json`
- `python euromillions_agent/fetch_prizes_range.py --start 2020 --end 2024 --out data/prizes_range.json`

## Inputs

- Normalized EuroMillions draws (`data/euromillions.csv`) from `python -m euromillions.get_draws`.
- Optional prize breakdown JSON (used by ROI objectives).

## Outputs

- Plots + metrics written under `--outdir` (calibration curves, Brier scores, permutation histograms).
- Console logs describing forward-only splits and mixer objectives.

## Reproducing the baseline

1) Fetch draws: `python -m euromillions.get_draws --out data/euromillions.csv --append`.
2) Grab prizes (optional): `python euromillions_agent/fetch_prizes.py --out data/prizes.json`.
3) Run: `python euromillions_agent/lotto_lab.py --csv data/euromillions.csv --mode all --outdir lotto_lab_out --debug`.
4) Inspect plots in `lotto_lab_out/` and iterate with your own hyperparameters.
