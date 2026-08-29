---
license: mit
pretty_name: LottoBench Community Benchmark
size_categories:
- n<1K
annotations_creators:
- machine-generated
source_datasets:
- original
tags:
- tabular
- timeseries
- benchmark
- reproducibility
- lottery-research
- synthetic
configs:
- config_name: history
  data_files:
  - split: train
    path: data/synthetic_history.csv
- config_name: results
  data_files:
  - split: train
    path: data/benchmark_results.csv
- config_name: poi_g_subsets
  data_files:
  - split: train
    path: data/poi_g_subset_results.csv
---

# LottoBench Community Benchmark

A deterministic, redistribution-safe **software** benchmark for comparing lottery strategy providers
under forward-only, equal-budget evaluation. It measures implementation behaviour — portfolio
coverage, diversity, crowding, and reproducibility — on synthetic draws. It is not evidence that a
fair lottery is predictable, and it contains no operator-sourced data.

- **Leaderboard Space:** [kugguk/lottobench-community-leaderboard](https://huggingface.co/spaces/kugguk/lottobench-community-leaderboard)
- **Source code:** [kugguk2022/lotteries-init-at-your-service](https://github.com/kugguk2022/lotteries-init-at-your-service)
- **Benchmark version:** `2.0.0` · **LottoBench:** `0.1.0a4` · **seed:** `20260828`
- **Snapshot digest:** `6bf3ff8045062e942d7445be27e2d994d16826b8735ab7b2f6f7e734c0017caf`

## What this benchmark does and does not show

| Question | Answer from this dataset |
|---|---|
| Does coordinating several providers beat the best single provider at equal budget? | Yes on the primary metric — coordinated aggregation reaches **0.405** pair coverage vs **0.383** for the best single provider (**+5.9 %**). |
| Does any provider predict the next draw? | No. `hit_recall` differences over a 4-draw holdout are noise, and POI-G candidate subsets contained the true ticket **0 of 4 times** at every size. |
| Does any strategy produce positive ROI? | No. Modelled jackpot-tier ROI sits between **−0.92 and −0.95** per ticket everywhere. Unpopularity buys a *less negative* number, never a positive one. |
| Is the run reproducible? | Yes. Fixed seed, frozen game shape, and a SHA-256 over the draw snapshot pinned in the manifest. |

## Results at a glance

### Provider track — primary metric `pair_coverage`

| provider | hit_recall | pair_coverage | number_coverage | mean_jaccard_diversity | expected_roi_per_ticket | unpopularity_lift |
|---|---:|---:|---:|---:|---:|---:|
| **coordinated_aggregation** | 0.3375 | **0.4053** | 0.9167 | 0.8112 | −0.9401 | 1.2058 |
| uniform_random | 0.3250 | 0.3826 | 0.8750 | 0.7814 | −0.9481 | 1.0447 |
| frequency | 0.3375 | 0.3561 | 0.8542 | 0.7226 | −0.9438 | 1.1318 |
| unpopularity | 0.2750 | 0.2841 | 0.6458 | 0.6152 | **−0.9222** | **1.5658** |

Coordinated aggregation wins the primary metric, which is the design claim under test: recombining
already-proposed tickets under a shared budget reaches more of the combinatorial space than any
contributing provider spending the same five tickets. The `unpopularity` provider deliberately
sacrifices coverage to capture the shared-jackpot lever, which gives it the highest
`unpopularity_lift` and the least-negative ROI — still a losing number.

### POI-G track — primary metric `containment_lift`

| subset_size | universe_fraction | reduction_factor | contained_draws | containment_rate | random_expected | containment_lift | modelled ROI (5 selected) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 1.01 % | 99.0x | 0 / 4 | 0.0000 | 0.0101 | 0.00 | −0.9364 |
| 100 | 5.05 % | 19.8x | 0 / 4 | 0.0000 | 0.0505 | 0.00 | −0.9364 |
| 500 | 25.25 % | 3.96x | 0 / 4 | 0.0000 | 0.2525 | 0.00 | −0.9360 |

**Read this null result carefully.** A 4-draw holdout cannot separate a useless reducer from a good
one. Even a shortlist with exactly random containment would show zero hits with probability 0.96
(size 20), 0.81 (size 100), and 0.31 (size 500). The honest conclusion is *no measurable
containment lift, and not enough holdout to measure one* — not *POI-G is refuted*. The rows are
published because suppressing a null result would make the leaderboard dishonest.

## Files and schema

### `data/synthetic_history.csv` — config `history`, 48 rows

Weekly synthetic draws for a 4-of-12 main pool plus a 1-of-4 auxiliary pool, generated from
`numpy.random.default_rng(20260828)`. No operator data of any kind.

| column | type | meaning |
|---|---|---|
| `draw_date` | string (ISO date) | Synthetic weekly draw date, starting 2025-01-01. |
| `ball_1` … `ball_4` | int | Main-pool numbers, ascending, drawn without replacement from 1–12. |
| `star_1` | int | Auxiliary-pool number from 1–4. |

### `data/benchmark_results.csv` — config `results`, 4 rows

One row per provider plus one for the coordinated aggregator, scored on the final 4 draws.

| column | type | meaning |
|---|---|---|
| `benchmark_version`, `dataset_sha256` | string | Comparability keys — rows are only comparable when both match. |
| `provider` | string | `uniform_random`, `frequency`, `unpopularity`, or `coordinated_aggregation`. |
| `budget`, `holdout`, `seed` | int | Tickets per draw (5), scored draws (4), RNG seed. |
| `hit_recall` | float 0–1 | Mean fraction of a ticket's main numbers appearing in the actual draw. |
| `pair_coverage` | float 0–1 | **Primary.** Fraction of all C(12,2) = 66 number pairs covered by at least one ticket. |
| `number_coverage` | float 0–1 | Fraction of the 12-number main pool touched by the portfolio. |
| `mean_jaccard_diversity` | float 0–1 | 1 − mean pairwise Jaccard similarity; 1 = fully disjoint tickets. |
| `expected_roi_per_ticket` | float | Modelled jackpot-tier `E[payout] · P(win) / price − 1`. Always negative. |
| `unpopularity_lift` | float | Portfolio expected payout ÷ that of an equally sized uniformly popular portfolio. Above 1 means the portfolio leans unpopular. |

### `data/poi_g_subset_results.csv` — config `poi_g_subsets`, 3 rows

POI-G is a **search-space reducer**, not a five-ticket provider. It ranks legal tickets by distance
from the next causal pair-co-occurrence target, fitted on a 26-draw trailing window of strictly
earlier rows. Containment is scored on the whole shortlist; ROI is scored only on the five
top-ranked tickets, because those are the only ones anyone would actually buy.

| column | type | meaning |
|---|---|---|
| `method`, `subset_size` | string, int | `poi_g_causal` and the requested shortlist size. |
| `universe_size`, `universe_fraction` | int, float | 1980 legal tickets, and the shortlist's share of them. |
| `reduction_factor` | float | `universe_size / subset_size` — how much search space was removed. |
| `contained_draws`, `containment_rate` | int, float | Holdout draws whose true ticket fell inside the shortlist. |
| `random_expected_containment_rate` | float | What an equally sized random shortlist would achieve. |
| `containment_lift` | float | **Primary.** `containment_rate / random_expected_containment_rate`. 1.0 = no better than random. |
| `selection_budget` | int | Tickets taken off the top of the shortlist for the ROI column (5). |
| `modeled_expected_roi_per_selected_ticket` | float | Modelled ROI of *those five tickets only* — never of buying the shortlist. |

### `data/benchmark_manifest.json`

Frozen game shape, seed, budget, holdout, provider list, metric names, snapshot digest, and the
claims boundary. Load this first when comparing runs.

## Load it

```python
from datasets import load_dataset

history = load_dataset("kugguk/lottobench-community-benchmark", "history")["train"]
results = load_dataset("kugguk/lottobench-community-benchmark", "results")["train"]
poi_g   = load_dataset("kugguk/lottobench-community-benchmark", "poi_g_subsets")["train"]
```

The manifest is a plain file rather than a config:

```python
import json
from huggingface_hub import hf_hub_download

manifest = json.load(open(hf_hub_download(
    "kugguk/lottobench-community-benchmark",
    "data/benchmark_manifest.json",
    repo_type="dataset",
)))
```

## Evaluation protocol

- **Forward-only.** At each holdout step `t`, every provider is refitted on `history[:t]` alone. No
  row at or after `t` is visible during fitting, so there is no leakage path.
- **Equal budget.** Every provider — and the aggregator — proposes exactly 5 tickets per step.
  Coverage comparisons are meaningless at unequal budgets.
- **Frozen game.** `main_n=12, main_k=4, auxiliary_n=4, auxiliary_k=1`, giving 1980 legal tickets.
- **Deterministic.** Seed `20260828` throughout; the aggregator is a greedy submodular selection
  that is deterministic given identical envelopes.
- **ROI model.** Jackpot 1000, ticket price 2.0, 10 000 other tickets. Jackpot tier only — lower
  prize tiers are not modelled, so ROI here is a comparative diagnostic, not a payout forecast.

## Reproduce and verify

```bash
git clone https://github.com/kugguk2022/lotteries-init-at-your-service
cd lotteries-init-at-your-service
pip install -e .
python scripts/build_platform_bundles.py
sha256sum publishing/common/synthetic_history.csv
```

The digest must equal `dataset_sha256` in `benchmark_manifest.json`. If it differs, the snapshot
changed and your numbers are not comparable to the ones above.

## Submitting results

Results may be compared under the same `benchmark_version` **only** when the snapshot digest,
provider set, budget, holdout, seed, and scoring code are all unchanged. Change any of them and bump
the version instead — a leaderboard whose rows were produced under different conditions is worse
than no leaderboard. Open an issue or discussion on the
[source repository](https://github.com/kugguk2022/lotteries-init-at-your-service/issues) to submit.

## Limitations and responsible use

- The draws are synthetic and uniformly random by construction. Nothing here transfers to an
  operated lottery, and no result should be quoted as if it did.
- The 4-draw holdout is far too small for a predictive claim in either direction. Coverage and
  diversity metrics are stable at this size; `hit_recall`, containment, and ROI are not.
- `expected_roi_per_ticket` and `modeled_expected_roi_per_selected_ticket` are **modelled**
  quantities from a single-tier jackpot model, not realised returns.
- LottoBench does not place wagers, recommend gambling, improve mechanical odds, or give financial
  advice. The safest financial baseline is not to play. If gambling is causing harm, contact a local
  support service such as [BeGambleAware](https://www.begambleaware.org/).
- The MIT license covers these synthetic artifacts and the LottoBench code. It grants no rights to
  third-party lottery data, names, marks, or services.

## Citation

```bibtex
@software{lottobench_community_benchmark_2026,
  title   = {LottoBench Community Benchmark},
  author  = {kugguk},
  year    = {2026},
  version = {2.0.0},
  url     = {https://huggingface.co/datasets/kugguk/lottobench-community-benchmark},
  note    = {Deterministic synthetic benchmark; LottoBench 0.1.0a4}
}
```
