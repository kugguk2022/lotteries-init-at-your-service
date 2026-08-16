# Repository Map

Three tiers. Knowing which tier you are in tells you what guarantees to expect.

| Tier | Meaning | CI coverage |
|---|---|---|
| **Maintained core** | Tested, linted, gated. Safe to build on. | Blocking |
| **Labs** | Real research code, unstable interfaces, uneven tests. Read before trusting. | Non-blocking |
| **Legacy** | Kept for provenance. Do not extend. | None |

## Maintained core — `lotteries_core/`

The distributed-inference framework. This is the part with a stable contract.

| Module | Role |
|---|---|
| [`protocol.py`](../../lotteries_core/protocol.py) | `GameSpec`, `Ticket`, `InferenceProvider`, `ProviderResult`. The interface every strategy implements. |
| [`providers/`](../../lotteries_core/providers/) | The strategies themselves — see [Methods and Findings](Methods-and-Findings.md). |
| [`likely_set_generator.py`](../../lotteries_core/likely_set_generator.py) | The owner's pair-co-occurrence level-set generator, plus its provider wrapper and CLI. |
| [`aggregation.py`](../../lotteries_core/aggregation.py) | Deterministic, diversity-aware combination of provider proposals under a shared budget. |
| [`envelope.py`](../../lotteries_core/envelope.py) | Reproducible proposal envelopes with SHA-256 provenance over the training data. |
| [`evaluation.py`](../../lotteries_core/evaluation.py) | Forward-only, equal-budget benchmark. The single source of truth for "is it better?". |
| [`coverage.py`](../../lotteries_core/coverage.py) | Pair/number coverage and Jaccard diversity metrics. |
| [`popularity.py`](../../lotteries_core/popularity.py) | Model of how the crowd picks numbers (calendar bias, "lucky" numbers, low-number bias). |
| [`roi.py`](../../lotteries_core/roi.py) | Jackpot-sharing expected value, and `InstantGamePool` finite-deck remaining EV. |
| [`outcome_tracker.py`](../../lotteries_core/outcome_tracker.py) | The prospective ledger. See [Outcome Tracking](Outcome-Tracking.md). |
| [`benchmark.py`](../../lotteries_core/benchmark.py) | CLI entry point for `evaluation.py`. |

Gated in CI by `tests/test_core_inference.py`, `tests/test_outcome_tracking.py`,
`tests/test_benchmark_regression.py`, `tests/test_causal_poi.py`.

## Labs

Research code. Interfaces change; tests are partial.

### `euromillions/`

The largest lab and the oldest code.

- `get_draws.py` — multi-source fetcher with caching, retry, and normalization. **Works** (one
  upstream source has 404'd; the archive source carries it).
- `schema.py` — canonical column set and range validation. Tested.
- `infer.py` — frequency-weighted baseline generator. Works; missing one function its test expects.
- `arithmetic_branch.py` — the currently best-validated forecasting mode (`classic`). See
  [Methods and Findings](Methods-and-Findings.md).
- `branch_hmm_v3.py`, `branch_hmm_v4.py` — hidden Markov branch models.
- `garchx*.py`, `garch_*_benchmark.py` — GARCH volatility experiments on draw-derived series.
- `branch_shortlist_benchmark.py`, `model_compare.py`, `split_backtest.py` — comparison harnesses.
- `diagnostics3.py`, `lottology.py`, `guess.py`, `roi.py` — assorted. `roi.py` is a **stub**; its CLI
  errors by design.

### `euromillions_agent/`

- `lotto_lab.py` — the "lab": logistic agent, co-occurrence discriminator, tiny transformer, RL mixer.
- `phase2_sobol.py` — Sobol low-discrepancy ticket generation and POI feature extraction.
- `grokky.py`, `fetch_prizes.py`, `fetch_prizes_range.py` — transformer experiment and prize scrapers.

### `totoloto/`, `eurodreams/`

Fetchers and parsing for the Portuguese Totoloto (5+1 from 49/13) and EuroDreams (6+1 from 40/5).
HTML-scraping heuristics; they break when upstream markup drifts.

### `scripts/`

- `predict_next_draw.py` — logistic agent over engineered features.
- `curate_garch_outputs.py` — ranks and archives GARCH run artifacts.

## Legacy

- `euromillions/euromillions_legacy_check.r` and other R files — historical reference only. The
  lineage runs R pair co-occurrence → `eurodreams/Edreams.py` → `lotteries_core/likely_set_generator.py`.
- `grok.py` at the repository root — superseded by `euromillions_agent/`.

## Data and outputs

| Path | Contents |
|---|---|
| `euromillions/euromillions_2016_2025.csv` | Bundled history, 1,004 draws, **ends 2025-08-12**. Columns: `draw_no,date,weekday,n1..n5,star1,star2,jackpot,jackpot_wins`. |
| `data/` | Fetched histories, prize caches (`prizes.json`, `prizes_range.json`). |
| `outputs/` | Benchmark artifacts, plots, curated model rankings. |
| `.cache/` | HTTP response cache for the fetchers. |

> **Column-name warning.** Two schemas coexist. The bundled CSV uses `date` / `n1..n5` / `star1`;
> the fetcher's normalized output uses `draw_date` / `ball_1..ball_5` / `star_1`. Loaders here accept
> both by prefix-sniffing, but this mismatch is the direct cause of the broken `--allow-stale`
> fallback. See [Current State](Current-State.md).

## Packaging note

`lotteries_core` is a proper package. `euromillions/`, `totoloto/`, `eurodreams/`, and
`euromillions_agent/` have **no `__init__.py`** — they resolve as implicit namespace packages. Module
execution (`python -m euromillions.infer`) works; package-level imports
(`from euromillions import ...`) do not.
