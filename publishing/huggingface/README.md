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
---

# LottoBench Community Benchmark

A deterministic, redistribution-safe software benchmark for comparing lottery strategy providers
under forward-only, equal-budget evaluation. It evaluates implementation behavior—not whether a
fair lottery is predictable.

## Contents

- `synthetic_history.csv`: generated integer-pool draws with no operator data.
- `benchmark_results.csv`: provider and coordinated-aggregation metrics.
- `benchmark_manifest.json`: frozen game shape, seed, budget, holdout and dataset digest.

The primary metric is pair coverage. Expected ROI is a modeled diagnostic and hit recall is
high-variance; neither is evidence of future performance or positive user returns.

## Reproduce

From LottoBench `0.1.0a3` source:

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
