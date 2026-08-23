# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
