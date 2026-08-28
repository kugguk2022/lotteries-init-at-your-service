# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- A draw settled without `--payout-table` no longer records a EUR 0 prize. Prize, net return
  and ROI are written as NaN, the row carries `payout_table_present` and `payout_source`, and
  `report` sums money only over draws that have a table and says how many that is. Recording
  zero understated ROI for every such draw, invisibly, over the whole tracking window.
- `settle` is idempotent and crash-safe. Result rows are upserted on `(draw_key, method)`,
  every entrant is scored before anything is written, the ledger's JSONL files are rewritten
  atomically, and an already-settled draw is skipped instead of scored twice. A crashed or
  retried run can no longer duplicate rows in `results.csv` while leaving records pending.
- Settled records pass their own integrity check. Settlement facts now live in a `settlement`
  object with its own `settlement_sha256`, instead of being added to the record after
  `record_sha256` was computed -- which made every settled entry look tampered with.
- `ledger/**/*.csv` is exempt from the repository's `*.csv` ignore rule. `results.csv`, the
  experiment's primary artifact, could not be committed at all.
- `realized_roi.normalize_record` no longer back-computes ROI from a missing net return as if
  it were 0.0, and the graduation gate treats a draw with no payout data as carrying no ROI
  evidence rather than as a failed integrity check.

### Added

- The read-only LottoBench Analytics API now exposes validated realized-ROI summaries and
  cumulative evolution by provider version, with incomplete evidence counted but excluded.
- A GPT Action setup guide uses the API's generated OpenAPI schema; no retired plugin manifest or
  OpenAI API key is required for the LottoBench server itself.
- Redistribution-safe Hugging Face dataset/Space and Kaggle dataset/notebook bundles provide a
  deterministic synthetic community benchmark without publishing operator or user data.
- Per-game operational defaults for unattended runs: `--history` defaults to the canonical
  store, `--ledger` to `ledger/<game>`, the game to `euromillions`, and the ticket price to
  the game's official price. `record` warns at record time when no price is known, rather
  than surfacing it as NaN money columns hours later at settlement.
- `settle --force` re-scores an already-settled draw in place, for correcting a mistyped
  result, and `--payout-source approximate` marks an estimated rather than official table.
- `.gitattributes` gives the append-only ledger files a union merge and fixed line endings.
- Payout tables may use `5_2` or `5+2` tier keys and a `tiers` or `prizes` wrapper; a file
  that is neither now fails with an explanatory message instead of a `TypeError`.

## [0.1.0a2] - 2026-08-23

### Added

- Packaged data retrieval: `lotteries_core.sources` ships the EuroMillions CSV
  archives, their HTML-archive fallback, and the shared schema validation, so
  `lottobench fetch` completes the documented journey from a plain
  `pip install lottobench`. The retriever previously lived in `experiments/`,
  which is not distributed.
- Nederlandse Lotto (`nl-lotto`) as a second end-to-end game, retrieved from the
  official operator API. Lotto XL and the Super Saturday second draw are kept out
  of the primary training history rather than silently merged into it.
- `BACKLOG_GAMES`: games whose shape is known but whose data path is not
  implemented are listed separately and raise an explanatory `KeyError`, so the
  catalogue never advertises a game the package cannot fetch.
- Realized-ROI settlement (`lotteries_core.realized_roi`, `lotto-roi`) comparing
  provider selections against settled draws.
- Temporal providers `garch_markov_branch` and `sequence_transformer`, and a
  matched `uniform_random` signal-off control that always runs alongside them.
- `lottobench games` and `lottobench providers`, the latter reporting per-provider
  local availability and the extra needed to enable a missing one.
- Multi-game SQLite storage with per-game provenance: row count, source label, and
  a content digest covering the rows a later benchmark actually reads.
- Release and health tooling: `scripts/validate_user_journey.py`,
  `scripts/doctor.py`, `scripts/check_graduation.py`, and the monthly
  `graduation-watch` workflow.

### Changed

- `requests` and `beautifulsoup4` are base dependencies rather than an optional
  extra, because automatic retrieval must work in a clean installation. There is
  no `scrape` extra; a damaged install is reported as a reinstall instruction.
- New `transformer` extra for the Torch-only provider; `ml` and `transformer` now
  require `torch>=2.10.0`, and the `api` extra adds `anyio`.
- `scripts/wheel_smoke.py` asserts the published contract for every advertised
  game and pins the version against the installed distribution metadata instead of
  a literal, so it cannot go stale on a version bump.

### Removed

- `scripts/curate_garch_outputs.py`, `scripts/predict_next_draw.py`, and
  `scripts/refresh_history.py`, superseded by the packaged fetch/benchmark path.

### Fixed

- Unused `asyncio_default_fixture_loop_scope` pytest option removed; it belongs to
  `pytest-asyncio`, which the test suite does not use, and emitted a
  `PytestConfigWarning` on every run.

## [0.1.0a1] - 2026-08-23

This release marks a pivot toward a **distributed-inference + combinatorial-coverage
research framework**.

### Added

- Common inference protocol shared across analysis backends, enabling
  distributed, backend-agnostic experimentation.
- Reproducible envelopes that capture inputs, seeds, and configuration so runs
  can be replayed deterministically.
- Diversity-aware, equal-budget aggregation for combining candidate sets while
  maximizing combinatorial coverage under a fixed budget.
- Scope documentation clarifying that the project is simulation-only:
  it handles no money, no ticket purchasing, no pooling of funds, and makes no
  claim of guaranteed winnings.

## [0.1.0]

### Added

- Initial baseline release of the lottery data-analysis research framework,
  providing simulation and combinatorial-analysis primitives.
