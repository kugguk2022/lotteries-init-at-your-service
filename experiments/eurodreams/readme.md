# EuroDreams (lab)

EuroDreams draw fetchers (multiple sources) for annuity-style analysis. Lab status: HTML structures may change; keep scripts isolated from the stable API.

## Quickstart

```bash
# All history (primary Irish archive)
python eurodreams/eurodreams_get_draws.py --out data/eurodreams_all.csv

# Limit by year range
python eurodreams/eurodreams_get_draws.py --out data/eurodreams_2023_2025.csv --start-year 2023 --end-year 2025

# JSON output instead of CSV
python eurodreams/eurodreams_get_draws.py --format json --out data/eurodreams_all.json

# Force a specific source
python eurodreams/eurodreams_get_draws.py --source irish      # primary full archive
python eurodreams/eurodreams_get_draws.py --source euro       # per-year pages with draw_code
python eurodreams/eurodreams_get_draws.py --source lottery_ie # last ~90 days
```

## Inputs & Outputs

- Inputs: Irish archive (full), euro-millions.com/PT per-year pages, lottery.ie recent-history page.
- Outputs: CSV/JSON with `date, weekday, n1..n6, dream, draw_code, source_url`, deduped chronologically.

## Notes

- This is a research lab; expect to refresh parsers when upstream HTML changes.
- Prize breakdowns are not included.
