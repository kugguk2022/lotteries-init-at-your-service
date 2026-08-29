---
license: mit
pretty_name: LottoBench Community Benchmark
tags:
- tabular
- timeseries
- benchmark
- reproducibility
- lottery-research
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

A deterministic, redistribution-safe software benchmark for comparing lottery strategy providers
under forward-only, equal-budget evaluation. It evaluates implementation behavior—not whether a
fair lottery is predictable.

## Contents

- `synthetic_history.csv`: generated integer-pool draws with no operator data.
- `benchmark_results.csv`: provider and coordinated-aggregation metrics.
- `poi_g_subset_results.csv`: causal POI-G candidate-set containment, matched random expectation,
  reduction factor, and modeled ROI for a separate fixed-budget selection.
- `benchmark_manifest.json`: frozen game shape, seed, budget, holdout and dataset digest.

The provider track's primary metric is pair coverage. The POI-G track's primary metric is
next-draw containment lift over an equally sized random subset. Its ROI column applies only to the
five ranked tickets selected from each subset, not to buying the entire subset. Expected ROI is a
modeled diagnostic and the four-draw synthetic holdout is far too small for predictive claims.

## Reproduce

From LottoBench `0.1.0a4` source:

```bash
python scripts/build_platform_bundles.py
```

Compare the resulting SHA-256 in `benchmark_manifest.json`. Results must not be submitted under the
same benchmark version when the snapshot, providers, budget, holdout, seed, or scoring code differs.

## Limitations and responsible use

This dataset is synthetic and unsuitable for claims about operated lotteries. LottoBench does not
place wagers, recommend gambling, improve mechanical odds, or provide financial advice. The safest
financial baseline is not to play.

The MIT license covers these synthetic artifacts and LottoBench code. It does not grant rights to
third-party lottery data, names, marks, or services.
