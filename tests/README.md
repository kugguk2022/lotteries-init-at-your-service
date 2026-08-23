# Tests

What’s covered
- EuroMillions schema + normalization + ticket scoring.
- `get_draws` failure modes (network error, malformed CSV, duplicates, cache fallback).
- End-to-end “fetch → normalize → evaluate → simple metric” paths for EuroMillions, Totoloto, and EuroDreams using offline fixtures.

How to run
- `make test` (runs Ruff then pytest).
- Runtime: ~5–10 seconds on a laptop; no network calls required.
