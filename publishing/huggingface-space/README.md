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
short_description: Three lottery profiles ranking agents against an equal-budget house null
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
2. **EuroMillions**: 1,113 normalized public-archive draws through 2026-08-28.
3. **Nederlandse Lotto**: 52 primary `6/45` draws from the operator API through 2026-08-22.

Each profile provides two screens:

- **Agent Arena** replays twelve contests, ranks seven agents against the uniform null, animates
  every committed ticket walk, and exposes the exact scored ledgers.
- **Pending Set Lab** shifts a transparent allocation between bet engineering and house/draw
  engineering, then exports a re-ranked dataset of frozen, unscored submissions.

Raw observed histories are not redistributed. The public Space contains source provenance,
snapshot hashes, derived benchmark results, scored commitments, and pending commitments.

ROI alpha is modeled expected-ROI percentage-point difference from an equal-budget uniform null.
It is not realized profit, increased draw probability, evidence of operator manipulation, or
betting advice. The pending allocation score is a research ranking lens, not a forecast
probability.
