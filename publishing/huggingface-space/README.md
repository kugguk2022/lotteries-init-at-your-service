---
title: LottoBench Community Leaderboard
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.44.1
app_file: app.py
pinned: false
license: mit
short_description: Forward-only, equal-budget results on deterministic synthetic lottery draws
tags:
- benchmark
- leaderboard
- reproducibility
- synthetic
datasets:
- kugguk/lottobench-community-benchmark
---

# LottoBench Community Leaderboard

Read-only visualization of the deterministic synthetic LottoBench community benchmark. This Space
does not make predictions, place wagers, or establish performance on operated lotteries.

Every number shown is read from `data/`, which is a byte-identical copy of the
[benchmark dataset](https://huggingface.co/datasets/kugguk/lottobench-community-benchmark). The
Space computes nothing at runtime — it renders a frozen, seeded run so that what you see here and
what you reproduce locally are the same artifact.

## What the two tables mean

- **Provider track** — four strategies scored on the same 52-draw holdout at the same 5-ticket
  budget. Primary metric is `pair_coverage`. Coordinated aggregation leads it, which is the design
  claim being tested: recombining proposals beats any single contributor at equal budget.
- **POI-G track** — a search-space reducer scored over 104 draws on **two** containment axes.
  Main-only containment (against the main-pool universe) is primary; full-ticket containment is
  reported beside it. They have different denominators on purpose — POI-G barely ranks on the
  auxiliary axis at small shortlist sizes, so the full-ticket figure alone understates it.

Modelled ROI is negative everywhere and always will be; a fair lottery is a negative-sum game.

**Every containment lift here sits near 1.0, and that is the only possible outcome.** The draw
history is uniform by construction, so it holds no structure for any reducer to find. These rows
are a calibration check on the harness, not a verdict on POI-G — that would need a non-uniform or
real-data track, which this benchmark does not yet have.

## Rebuild

```bash
git clone https://github.com/kugguk2022/lotteries-init-at-your-service
cd lotteries-init-at-your-service
pip install -e .
python scripts/build_platform_bundles.py
```

Not financial or gambling advice. The safest financial baseline is not to play.
