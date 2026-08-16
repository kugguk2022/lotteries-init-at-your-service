# Contributing

Thanks for helping! This project is research-focused; please keep experiments isolated and documented.

## Getting started

- Read [`docs/wiki/Home.md`](docs/wiki/Home.md) first — especially
  [Scope and Honesty](docs/wiki/Scope-and-Honesty.md) and [Current State](docs/wiki/Current-State.md).
- Fork + clone, create a virtualenv, then `pip install -e ".[dev]"`.
- Before sending a PR, run the maintained-core gate (this is what CI blocks on, and it passes):

  ```bash
  pytest -q tests/test_core_inference.py tests/test_outcome_tracking.py \
           tests/test_benchmark_regression.py tests/test_causal_poi.py
  ruff check <the files you changed>
  ```

  `make test` runs the full gate and currently **fails** for pre-existing reasons unrelated to your
  change — two uncollectable test modules and 61 repository-wide lint errors. Fixing either is a
  welcome PR; see [Current State](docs/wiki/Current-State.md).

## Good first experiments

- Add documentation examples (e.g., small notebooks showing new metrics).
- Tighten parsers when upstream HTML changes (Totoloto/EuroDreams).
- Extend tests around caching or schema edge cases.
- Contribute another lottery fetcher with a clear schema + tests.

## Guidelines

- Prefer pure-Python, offline tests; avoid hitting remote endpoints in CI.
- Document inputs/outputs for any new scripts; lab work should live in `labs/` or a dedicated subfolder.
- `lotteries_core` is the stable surface — the provider protocol, envelopes, aggregation, evaluation, and the outcome tracker. Keep changes to it minimal and tested. (`euromillions` was previously described here as the stable surface; it has no package-level exports today, which is [a tracked issue](docs/wiki/Current-State.md).)
- Mark anything experimental in docstrings/README notes.
- Open a GitHub Discussion if you want to propose a larger research direction before coding.

## Releases

Tag milestones (e.g., `v0.1` for stable EuroMillions fetch + schema + scoring) so downstream users can pin versions and cite stable URIs.
