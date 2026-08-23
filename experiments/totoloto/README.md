# Totoloto (lab)

Portuguese Totoloto fetcher and parsing utilities. Treated as a lab; expect the HTML to drift and file formats to evolve.

## Quickstart

```bash
# Full history (default ranges 1-49 mains, 1-13 bonus)
python totoloto/totoloto_get_draws.py --out data/totoloto.csv

# Limit by year range
python totoloto/totoloto_get_draws.py --out data/totoloto_2015_2025.csv --start-year 2015 --end-year 2025

# JSON output
python totoloto/totoloto_get_draws.py --format json --out data/totoloto.json

# Override rules if the lottery changes
python totoloto/totoloto_get_draws.py --out data/totoloto.csv --ball-range 1 49 --bonus-range 1 13
```

## Inputs & Outputs

- Inputs: per-year HTML archive pages from euro-millions.com/PT.
- Outputs: CSV/JSON with `draw_date, weekday, ball_1..ball_5, bonus, draw_code, source_url`, deduped chronologically.

## Notes

- Lab status: parsing heuristics are best-effort and may need refreshes.
- Legacy R scripts (`*.r`) are deprecated but kept for provenance.
