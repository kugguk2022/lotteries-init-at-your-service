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
- **Pre-draw Set Lab** publishes the 12 sealed Crowd Escape draws with crowding and conditional
  jackpot-sharing estimates, the transformer + GARCH million-ticket branch set, an exhaustive
  pre-draw historical pair-density raster, and a transparent allocation across frozen agent
  submissions.

Open the in-Space **START HERE / COMPLETE SPACE WIKI** or read
[`SPACE_GUIDE.md`](SPACE_GUIDE.md) for the complete profile lifecycle, exact meaning of `PENDING`,
publication conditions, artifact map, verification procedure, and evidence boundaries.

Raw observed histories are not redistributed. The public Space contains source provenance,
snapshot hashes, derived benchmark results, scored commitments, and pending commitments.
Each refreshed observed profile also provides a compact machine-readable `summary.json`, the top
250 exhaustive pair-density rows, and the exact full-universe score distribution. EuroMillions
scores all 139,838,160 legal tickets; NL Lotto scores all 8,145,060 legal tickets. These pending
artifacts use history only through the displayed cutoff and target the next expected draw.
Crowd Escape forecasts human ticket popularity rather than lottery results: its selections have
the same fair-draw probability as every other legal ticket.

## Publication timing and independent checks

The Space opens on the observed EuroMillions profile. The synthetic lab is a fixed demonstration
and has no scheduled pre-draw publication.

New selections are published **after the previous verified result**, ahead of the following draw:

| Profile | Scheduled refresh (UTC) | Following target |
| --- | --- | --- |
| EuroMillions | Wednesday 08:15 | Friday draw |
| EuroMillions | Friday 23:15 | Tuesday draw |
| NL Lotto | Sunday 08:15 | Following Saturday draw |

GitHub can delay scheduled runs. A source failure preserves the last verified files and their
original target date; it does not make old selections new or establish a pre-draw publication.
Downloads stay public through the Space repository even during an application restart.

In **Pre-draw Set Lab**, the lifecycle and publication panels link directly to the files and their
dated history. `PENDING` means that the artifact was generated for the displayed target draw and has
not been advanced by a successful post-draw result refresh. It is not an approval, confidence,
purchase, profitability, or settlement claim. The synthetic profile is labelled `DEMO_ONLY` and is
never scheduled against a real draw.

The current Space is a current-publication view: a successful refresh advances the displayed target
and preserves the previous files in Hugging Face revision history. Space-specific artifacts are not
currently rewritten as first-class `SETTLED` rows; the repository's prospective outcome tracker is
the canonical settlement mechanism.

The publication panel links directly to the files and their dated history.
Its automatic check compares the displayed Crowd Escape CSV and manifest with an exact public
Space revision, validates the complete file SHA-256, and reconstructs each ticket commitment.
Keep the resulting revision URL and a downloaded copy, then compare that same set after the draw.
Previous files are reachable from each profile's **Dated publication history** link after the
latest files advance. The million-ticket download has its own artifact hashes in
`temporal_hybrid_summary.json`; the Crowd Escape check does not verify that separate archive.

The publication date is Hugging Face's commit record, not the history cutoff, a newly generated
local timestamp, or independent timestamp notarization. A same-day publication explicitly needs
comparison with the official draw time; a later publication is not pre-draw evidence. A hash by
itself only detects changed content. If the public record cannot be checked, the UI says so and
keeps downloads and history links available. Historical forward replays do not substitute for
dated public pre-draw evidence, and neither establishes realized profit without settled payouts.

ROI alpha is modeled expected-ROI percentage-point difference from an equal-budget uniform null.
It is not realized profit, increased draw probability, evidence of operator manipulation, or
betting advice. The pending allocation score is a research ranking lens, not a forecast
probability.
