# Current State

An honest status page. Everything below was verified by running it, not inferred from the code.

**Last verified:** 2026-08-16, on `agent/integrate-outcome-tracking`.

## Summary

| Area | State |
|---|---|
| `pytest` (whole suite, no arguments) | **Green locally** — 85 passed. |
| `ruff check .` (whole repository) | **Green locally** — all checks passed. |
| `make test` | **Green locally** — both of the above. |
| CI | **Blocks on both, repository-wide.** Was **red** on 2026-08-16 from linter version drift, not from repository code — see Open item 1. |
| HTTP API | **Working** — `lotto-serve`, 9 selectable providers, OpenAPI schema, 15 tests. |
| Draw history | **Current and tracked** — `data/euromillions.csv`, 1,972 draws through 2026-08-14, with a metadata sidecar and a scheduled refresh workflow. |
| Data fetchers | **Working**, including the offline `--allow-stale` fallback. |
| Public API | **Exists and is tested** — `euromillions` exports resolve; `lotteries_core` is the stable surface. |
| Prospective ledger | **Live** — six entrants recorded for the 2026-08-18 draw. |

The repository was red on the first four of those rows as recently as this morning. What follows is
what was wrong and what was done, kept as a record rather than deleted.

## Resolved

### Two test modules could not be imported

`tests/test_end_to_end_paths.py` and `tests/test_infer.py` failed at collection, so a bare `pytest`
never ran anything.

- `euromillions/` had no `__init__.py`, so `from euromillions import EuroMillionsGuess, ...` could not
  resolve. Added one, with **lazy PEP 562 exports** — eager imports would make
  `python -m euromillions.get_draws` import the module twice and emit a `RuntimeWarning`.
- `euromillions/infer.py` was missing `random_candidates`, and `generate_candidates` /
  `probability_tables` had been generalized to require explicit game-shape arguments that the tests
  did not pass. Added `random_candidates` (the uniform control) and gave the shape parameters
  EuroMillions defaults, so both the general and the simple call sites work.

### `CONTRIBUTING.md` documented an API that did not exist

It named `euromillions` exports as the stable surface. Corrected to `lotteries_core`, with the
history noted rather than quietly rewritten.

### Repository-wide lint was red (61 errors)

36 auto-fixed (import ordering, unused imports). Semicolon statements, `== True` comparisons, and a
bare `except` were fixed by hand. The remaining 15 are unused locals in long research scripts and
module-level imports in single-file analysis scripts; they are scoped to **explicit per-file-ignores
in `pyproject.toml`**, each annotated with why. That list is a cleanup queue, not an exemption — do
not add to it.

### `--allow-stale` fallback was broken

`_load_existing` read the fallback CSV with `parse_dates=["draw_date"]`, but the bundled file's column
is `date`, so the documented offline path failed exactly when it was needed. Column aliasing is now
shared between `normalize` and the fallback loader via `canonicalize_columns`, and the alias table
covers the bundled `n1..n5` / `star1` spelling.

### Two coexisting CSV schemas

Still two spellings on disk, but they now converge through one alias table
(`COLUMN_ALIASES` / `CANONICAL_COLUMNS` in `euromillions/get_draws.py`) instead of through ad-hoc
prefix sniffing in each loader.

### Silent uniform stars in `infer.py`

When a history used an unrecognized star column, `probability_tables` skipped it and returned
**uniform** star probabilities instead of failing — a wrong answer rather than an error. It now tries
the known spellings and exits with the available columns listed if none match.

### CI only gated four files

The workflow ran four test files as blocking and the real suite with `continue-on-error: true`, and
linted only changed files. Both gates are now repository-wide and blocking.

### Aggregation lost coverage against the best single provider

The framework's headline claim is that coordination should not reduce combinatorial reach. It did:
aggregated pair coverage came in **0.02649 below** the best single provider on the 40-draw benchmark.

This was never a regression. `aggregation.py` had not been modified since commit `203e0f1`, which
created it, and a run made before any of the current providers existed showed the same gap at the
same magnitude (−0.0272). Adding stronger providers lifted both numbers and left the gap intact —
what a fixed property of the aggregator looks like.

**Cause.** `marginal_coverage` divided a ticket's new-pair count by the game's *entire* pair universe
(1,225 for 5-of-50), capping that term near 0.008 while the consensus and unpopularity terms were
normalised to 0–1. The coverage lever was carrying roughly a hundredth of the weight it was given.

**Fix.** Marginal reach is now the fraction of *the ticket's own* pairs that are new, so it lives on
0–1 like the other terms. A `coverage_floor` backstop was added on top: if the blend still lands
below the best equal-budget single-provider portfolio, that portfolio is returned instead.

**Result.** Aggregated pair coverage went from 0.15780 (−0.02720 against best single) to **0.18851
(+0.00349 against best single)** — coordination now beats every single provider rather than merely
tying, and **the floor never fires**. That distinction matters: a floor that has to fire means the
objective is still broken and the symptom is being papered over.

`test_aggregation_coverage_not_worse_than_best_single_provider` now asserts against `max(single_cov)`.
It previously asserted against `min` — satisfied by almost anything, which is why a real shortfall
survived the entire benchmark history unnoticed. A second test pins floor-on against floor-off.

### Nothing tracked dataset provenance or staleness

Documented results were once computed on a history that had been a year stale, with nothing in the
repository saying so, and a ledger entry recorded against stale history is quietly invalid — the
method never sees the draws immediately before the one it predicts.

Each history now carries a `<name>.meta.json` sidecar ([`lotteries_core/dataset.py`](../../lotteries_core/dataset.py))
recording row count, date span, column schema, a content SHA-256, and the fetch time. The digest is
taken over the *canonicalised* frame, so reordering columns or changing line endings does not read as
a data change. [`scripts/refresh_history.py`](../../scripts/refresh_history.py) refreshes and
regenerates it; `--check` reports drift and staleness and exits non-zero.
[`.github/workflows/refresh-history.yml`](../../.github/workflows/refresh-history.yml) runs after each
draw, gates the new data through ruff and the full suite, and opens a pull request rather than
pushing — draw data feeds published results and a sealed ledger, so a human should see the diff.
Covered by `tests/test_dataset_metadata.py`.

## Open

### 1. CI and local development run different dependency stacks

Every runtime floor in `pyproject.toml` is open-ended (`pandas>=2.0`, `numpy>=1.25`, …), so CI
resolves the newest release of everything while development happens on whatever was installed months
ago. Observed on the 2026-08-16 run:

| package | local | CI |
|---|---|---|
| ruff | 0.14.5 | 0.16.3 |
| pandas | 2.3.3 | 3.0.5 |
| numpy | 1.26.4 | 2.4.6 |
| scikit-learn | 1.7.2 | 1.9.0 |

**This is why a green local gate did not mean a green CI gate.** The ruff half is fixed — the linter
is pinned to a minor line and the rule set is now selected explicitly — but the pandas and numpy
major-version gaps remain, and the test step has still never been observed passing against pandas 3 /
numpy 2 (the lint step failed first, so pytest never ran).

A scan found no removed numpy 2 aliases and no `DataFrame.append` usage, which is encouraging but is
not the same as a passing run. The options are to pin the runtime stack, add a CI matrix covering
both, or accept the drift knowingly. Until one is chosen, treat "passes locally" as weak evidence.

### 2. `euromillions/roi.py` is a documented stub

Its CLI errors by design. Disclosed in the README; listed here so it is not mistaken for a regression.

### 3. The per-file-ignore queue

Nine files carry `F841` or `E402` ignores. Each is a small cleanup for whoever next works in that
script with the context to judge whether an unused local was a deliberate intermediate.

### 4. The bundled CSV is still the stale one

`euromillions/euromillions_2016_2025.csv` (ends 2025-08-12) is unchanged and is still what a few
offline examples use. `data/euromillions.csv` is the canonical file — current, metadata-tracked, and
what the ledger and published results use.

### 5. The 2016 rules cutoff is enforced per-provider, not repository-wide

`parallax_guard` restricts residual inference to draws under the current game matrix
(`_current_rules_history`, from 2016-09-27), and a test covers it. Nothing enforces the same cutoff
for other providers, which train on all 1,972 rows including pre-2016 draws under a smaller star
pool. For frequency-style methods that is mostly harmless; for anything reading star structure it is
not. Worth making a `GameSpec`-level property rather than one provider's private rule.

## How to keep this page honest

Re-verify on any change to the build, the entry points, or the data schema. If a command is listed as
working, someone ran it. If it is broken, it stays listed as broken until it is fixed — deleting the
entry is not the same as resolving it. The resolved section stays too: a status page that only ever
shows green teaches a reader nothing about whether to trust it.
