# Current State

An honest status page. Everything below was verified by running it, not inferred from the code.

**Last verified:** 2026-08-16, on `agent/integrate-outcome-tracking`.

## Summary

| Area | State |
|---|---|
| Maintained core (`lotteries_core/`) | **Green.** 24/24 tests pass, lint clean, gated in CI. |
| Full test suite | **Red at collection.** 49 pass once 2 modules are excluded; those 2 cannot be imported. |
| Repository-wide lint | **Red.** 61 `ruff` errors, 37 auto-fixable. |
| Advertised quality gate (`make test`) | **Fails**, for both reasons above. |
| Bundled draw history | **Stale.** Ends 2025-08-12. |
| Data fetchers | **Working**, with one dead upstream source and a broken offline fallback. |
| Public API promised in `CONTRIBUTING.md` | **Does not exist.** |

CI currently reflects this honestly rather than hiding it: only the four maintained-core test files
block, and a full-suite audit runs with `continue-on-error: true`. That is a reasonable containment
strategy, but it is a holding position, not a destination.

## Issues, in the order worth fixing

### 1. Two test modules cannot be imported — blocks `pytest` with no arguments

```
tests/test_end_to_end_paths.py
  ImportError: cannot import name 'EuroMillionsGuess' from 'euromillions' (unknown location)
tests/test_infer.py
  ImportError: cannot import name 'random_candidates' from 'euromillions.infer'
```

**Cause.** `euromillions/` has no `__init__.py`, so it resolves as an implicit namespace package with
no package-level exports. `random_candidates` is genuinely absent from `euromillions/infer.py` (which
defines `probability_tables`, `generate_candidates`, `load_history`, `save_candidates`, `main`).

**Fix.** Add `euromillions/__init__.py` re-exporting the intended public surface, and either implement
`random_candidates` (a uniform control sampler, which is what the test uses it as) or update the test.
Both are small. This one change turns a bare `pytest` from *red* to *green*.

**Why it matters most.** A contributor's first command is `pytest`. Right now it fails at collection
before running anything, which reads as "this repository is abandoned".

### 2. `CONTRIBUTING.md` documents an API that does not exist

It states that "`euromillions` exports are the stable surface". There are no package-level exports.
Fixed by item 1, or by correcting the sentence. Until then the document is actively misleading.

### 3. Repository-wide lint is red

61 errors, dominated by import ordering (`I001`) and module-level imports after code (`E402`, e.g.
`totoloto/grok.py:143`). 37 are auto-fixable with `ruff check --fix .`.

**Recommended approach.** Fix the maintained core and any actively developed lab to zero, then add
per-file ignores for legacy scripts rather than rewriting them. Do not run a blanket auto-fix across
lab code without checking that `E402` imports were not deliberately placed after configuration.

### 4. Bundled history is a year stale

`euromillions/euromillions_2016_2025.csv` ends **2025-08-12** (1,004 draws). Every example in the
README uses it, so every documented result is computed on year-old data. Refresh before recording any
prospective prediction to the ledger.

### 5. `--allow-stale` fallback is broken

When all network sources fail, `get_draws` falls back to the bundled CSV and raises:

```
Missing column provided to 'parse_dates': 'draw_date'
```

**Cause.** `euromillions/get_draws.py:382` reads the fallback with `parse_dates=["draw_date"]`, but the
bundled file's date column is named `date`. So the documented offline path fails precisely when it is
needed. **Fix:** normalize the column before parsing, or sniff the name as the other loaders do.

**Note:** the fetcher itself works. The `merseyworld` source 404s and `pedro` returns the wrong content
type, but the `archive` source succeeds and results are cached under `.cache/euromillions`.

### 6. Two coexisting CSV schemas

`date`/`n1..n5`/`star1` (bundled) versus `draw_date`/`ball_1..ball_5`/`star_1` (fetcher output).
Loaders paper over this by prefix-sniffing column names. It is the root cause of item 5 and a standing
trap. Worth choosing one canonical schema and converting the other at the boundary.

### 7. `euromillions/roi.py` is a documented stub

Its CLI errors by design. Correctly disclosed in the README; listed here so it is not mistaken for a
regression.

### 8. Aggregation underperformed the best single provider

On the 40-draw benchmark, aggregated pair coverage came in 0.027 *below* the best single provider,
contrary to the framework's headline claim. See [Methods and Findings](Methods-and-Findings.md). This
is a research question, not a build failure, but it should not sit undocumented.

## Path to green

1. Add `euromillions/__init__.py`; resolve `random_candidates`. → bare `pytest` passes.
2. Lint the maintained core and active labs to zero; per-file-ignore the legacy scripts. → `make test`
   passes.
3. Fix the `--allow-stale` fallback and pick one canonical CSV schema.
4. Refresh the bundled history and re-run the documented benchmarks so published numbers are current.
5. Promote the full suite from `continue-on-error` to blocking in CI.

Steps 1–3 are small and mechanical. Step 5 is the one that makes the standard stick, because after it
the repository can no longer quietly drift back.

## How to keep this page honest

Re-verify on any change to the build, the entry points, or the data schema. If a command is listed as
working, someone ran it. If it is broken, it stays listed as broken until it is fixed — deleting the
entry is not the same as resolving it.
