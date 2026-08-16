# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

This release marks a pivot toward a **distributed-inference + combinatorial-coverage
research framework**.

### Added

- Common inference protocol shared across analysis backends, enabling
  distributed, backend-agnostic experimentation.
- Reproducible envelopes that capture inputs, seeds, and configuration so runs
  can be replayed deterministically.
- Diversity-aware, equal-budget aggregation for combining candidate sets while
  maximizing combinatorial coverage under a fixed budget.
- Scope and ethics documentation clarifying that the project is simulation-only:
  it handles no money, no ticket purchasing, no pooling of funds, and makes no
  claim of guaranteed winnings.

## [0.1.0]

### Added

- Initial baseline release of the lottery data-analysis research framework,
  providing simulation and combinatorial-analysis primitives.
