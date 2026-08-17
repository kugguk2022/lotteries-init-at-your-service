# Contributing

Thanks for helping! This project is research-focused; please keep experiments isolated and documented.

## Getting started

- Read [`docs/wiki/Home.md`](docs/wiki/Home.md) first — especially
  [Project Scope](docs/wiki/Scope.md) and [Current State](docs/wiki/Current-State.md).
- Fork + clone, create a virtualenv, then `pip install -e ".[dev]"`.
- Before sending a PR, run `make test` (`ruff check .` + `pytest -q`). Both are repository-wide and
  both must pass; CI blocks on exactly this.
- Adding a new strategy? Follow [Contributing a Provider](docs/wiki/Contributing-a-Provider.md) — it
  is one file plus two registry lines, and it explains the two things a contribution needs to be
  taken seriously here: an **ablation** that turns your signal off, and a **null calibration** that
  shows what you found is not noise.

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
