---
title: LottoBench Lottery Agent Arena
emoji: "🎯"
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 5.44.1
app_file: app.py
pinned: false
license: mit
short_description: Three lottery profiles ranked against an equal-budget null
tags:
- benchmark
- leaderboard
- reproducibility
- lottery
datasets:
- kugguk/lottobench-community-benchmark
---

# LottoBench Lottery Agent Arena

Three isolated lottery profiles share one forward-only, equal-budget scoring contract:

1. **EuroMillions lab control**: deterministic `5/50 + 2/12` draws used to verify the machinery.
2. **EuroMillions**: Irish National Lottery official results validated against a digest-pinned
   history package.
3. **Nederlandse Lotto**: primary `6/45` results from the operator API.

Each profile provides two screens:

- **Agent Arena** replays twelve contests, ranks seven agents against the uniform null, charts
  cumulative ROI-alpha movement, animates every committed ticket walk, and exposes the exact
  scored ledgers.
- **Pending Set Lab** publishes the 12 sealed Crowd Escape draws with crowding and conditional
  jackpot-sharing estimates, the transformer + GARCH million-ticket branch set, an exhaustive
  pre-draw historical pair-density raster, and a transparent allocation across frozen agent
  submissions.

Raw observed histories are not redistributed. The public Space contains source provenance,
snapshot hashes, derived benchmark results, scored commitments, and pending commitments.
Each refreshed observed profile also provides a compact machine-readable `summary.json`, the top
250 exhaustive pair-density rows, and the exact full-universe score distribution. EuroMillions
scores all 139,838,160 legal tickets; NL Lotto scores all 8,145,060 legal tickets. These pending
artifacts use history only through the displayed cutoff and target the next expected draw.
Crowd Escape forecasts human ticket popularity rather than lottery results: its selections have
the same fair-draw probability as every other legal ticket.

ROI alpha is modeled expected-ROI percentage-point difference from an equal-budget uniform null.
It is not realized profit, increased draw probability, evidence of operator manipulation, or
betting advice. The pending allocation score is a research ranking lens, not a forecast
probability.
