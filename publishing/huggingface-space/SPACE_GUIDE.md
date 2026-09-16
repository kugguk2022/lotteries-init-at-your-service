# LottoBench Space Guide

This page is the operational guide for the public LottoBench Space. It explains what each screen
shows, exactly what `PENDING` means, when files change, how to verify a pre-draw publication, and
which conclusions the evidence does and does not support.

## Quick start

1. Choose **EuroMillions** or **NL Lotto** for a real, dated profile. **EuroMillions Lab** is a
   fixed synthetic control and is never attached to a real draw.
2. Use **Agent Arena** to inspect twelve already-scored, forward-only historical contests.
3. Use **Pre-draw Set Lab** to inspect the current dated candidate artifacts.
4. Read the lifecycle panel first. It names the history cutoff, target draw, current state, scheduled
   refresh, and exact condition required for the live profile to advance.
5. For an independent pre-draw check, keep the dated Hugging Face revision URL and download the
   exact files before the draw. A hash proves content identity, not publication time by itself.

## The three profiles

| Profile | Data | Real target draw? | Automatic publication? | Purpose |
| --- | --- | --- | --- | --- |
| EuroMillions Lab | Deterministic synthetic draws | No | No | Tests the interface, scoring contract, and example commitments. |
| EuroMillions | Validated observed history | Yes | Yes | Publishes dated candidates after Tuesday and Friday results are verified. |
| NL Lotto | Validated observed history | Yes | Yes | Publishes dated candidates after the Saturday result is verified. |

The lab can display example frozen sets, but those sets are `DEMO_ONLY`. They never become a real
pending draw and never settle against an operator result.

## Lifecycle states

| State | Meaning | What makes it change |
| --- | --- | --- |
| `DEMONSTRATION_ONLY` | Synthetic control. No operator draw is scheduled. | Nothing. Choose an observed profile. |
| `PUBLICATION_DATA_MISSING` | The observed profile lacks the files needed to identify a current target. | A validated build publishes the missing files. |
| `PRE_DRAW_PUBLISHED` | Frozen candidates target a future draw. | The draw occurs, its official result is validated, and the next publication succeeds. |
| `DRAW_DAY_PENDING` | The target date is today. Date alone does not prove whether publication preceded draw time. | The post-draw result refresh and publication succeed. |
| `AWAITING_VERIFIED_RESULT` | The target date passed but the live profile has not advanced yet. | Official fetch, validation, Space upload, and public health check all succeed. |

`PENDING` is therefore a data-lifecycle label. It does **not** mean:

- waiting for a human to approve the set;
- waiting for the transformer/GARCH evidence gate to pass;
- likely to win;
- purchased;
- profitable; or
- settled against an official result.

## Exact publication schedule and conditions

Scheduled GitHub Actions runs are in UTC:

| Profile | Scheduled run | Result incorporated | New target |
| --- | --- | --- | --- |
| EuroMillions | Wednesday 08:15 | Tuesday result | Friday draw |
| EuroMillions | Friday 23:15 | Friday result | Tuesday draw |
| NL Lotto | Sunday 08:15 | Saturday result | Following Saturday draw |

A scheduled run advances a profile only when all of the following are true:

1. The workflow resolves the correct lottery from its original cron or explicit manual input.
2. The official source returns a history containing the expected completed draw.
3. History schema, ranges, ordering, overlap, and snapshot checks pass.
4. Crowd Escape, transformer + GARCH, exhaustive raster, benchmark, and commitment artifacts build.
5. Artifact hashes and cross-file cutoff/target invariants pass.
6. The complete bundle uploads to Hugging Face.
7. The Space restarts and its public health check succeeds.

GitHub can start scheduled workflows late. If the source or any gate fails, LottoBench keeps the last
verified files and their original target date. It does not rename stale files as a new forecast.

## What happens after the target draw

The current Space bundle is a **current-publication view**, not the canonical settlement ledger.
After a successful post-draw refresh, the live page advances to the next target and the previous
files remain accessible in Hugging Face's dated revision history.

The Space-specific Crowd Escape, transformer + GARCH, and raster files are not currently rewritten
as first-class `SETTLED` rows. Their dated revisions can be compared with the official result, while
the repository's separate prospective outcome tracker is the mechanism that records and settles
predictions with integrity checks. Do not interpret disappearance from the current view as proof of
settlement, purchase, payout, or profit.

## Screen A — Agent Arena

Agent Arena is historical evaluation, not the current forecast. It replays twelve forward-only
holdout contests with the same ticket budget for every provider and a seeded uniform-random null.

Key fields:

- **ROI alpha**: modeled expected-ROI percentage-point difference from the equal-budget null.
- **Latest movement**: change in cumulative mean ROI alpha when the newest holdout contest is added.
- **Consistency**: share of holdout contests in which a provider finished above the null.
- **Pair reach**: diversity/coverage diagnostic, not a win probability.

Small leaderboard movement is expected because every new contest contributes only one observation
to the twelve-contest cumulative view.

## Screen B — Pre-draw Set Lab

All artifacts on this screen use history only through the displayed cutoff and share the displayed
target date.

### Crowd Escape

Publishes twelve sealed tickets ranked by a static model of human number-choice popularity. It aims
to reduce jackpot-sharing pressure *conditional on an otherwise identical jackpot hit*. It does not
forecast the lottery machine and it does not change fair-draw odds.

### Transformer + GARCH branch set

Ranks the exact legal universe by distance to two forecasts of the next historical pair-density
score level. The downloadable file contains one million deterministic candidates and a separate
forward-only containment comparison against a matched random set.

The evidence gate and `RESEARCH_ONLY` label concern retrospective/forward containment evidence.
They are independent of the publication lifecycle. A file can be correctly `PENDING` for a target
draw while its research gate remains unpassed.

### Exhaustive pair-density raster

Scores every legal first-prize combination by historical pair co-occurrence. EuroMillions covers
139,838,160 tickets; NL Lotto covers 8,145,060. The Space publishes the top rows and the exact score
distribution. Pair density describes history; it is not a higher mechanical draw probability.

### Allocation-ranked frozen submissions

The sliders re-rank already-frozen agent submissions between two strategy surfaces. They do not edit
the ticket numbers, retrain a model, create a new public commitment, or change the target date.

## Files and verification

| File | What it contains | Primary integrity field |
| --- | --- | --- |
| `manifest.json` | Profile source, snapshot, game rules, evaluation contract, current target | `history.snapshot_sha256` |
| `summary.json` | Compact leaderboard, ROI evolution, refresh metadata, artifact summaries | Embedded snapshot and target fields |
| `crowd_escape_forecasted_draws.csv` | Twelve sealed crowding candidates | Per-ticket `commitment_sha256` |
| `crowd_escape_summary.json` | Crowd model, economics, target, file hash | `artifacts` SHA-256 map |
| `temporal_hybrid_candidates_1m.csv.gz` | Exact million-ticket branch-ranked set | Hash in temporal summary |
| `temporal_hybrid_backtest.csv` | Forward-only containment evidence | Hash in temporal summary |
| `temporal_hybrid_summary.json` | Model diagnostics, gate, coverage, economics, hashes | `artifacts` SHA-256 map |
| `pair_raster_top.csv` | Top exhaustive pair-density rows | Snapshot/target commitments |
| `pair_raster_distribution.csv` | Exact full-universe score distribution | Cross-checked by build gates |
| `prospective.csv` | Frozen submissions from all agents | Per-ticket `commitment_sha256` |

For the strongest available check:

1. Open **Dated publication history**.
2. Select the revision that existed before the draw.
3. Download the CSV and manifest from that exact revision, not from moving `main`.
4. Verify the file SHA-256 and per-ticket commitments.
5. Compare the target date and publication timestamp with the official draw time.
6. After the result, compare the unchanged file with the official numbers and retain both sources.

The Space's **Check publication record** button automates the current Crowd Escape file/hash/
commitment comparison. It does not independently notarize time and does not verify the separate
million-ticket archive.

## Evidence and economics boundaries

- Every legal combination has the same mechanical probability in a fair draw.
- Candidate-set coverage is not profitability.
- A contained first-prize combination is not a realized win unless the corresponding tickets were
  actually purchased before the draw.
- Profit requires settled official payouts to exceed the full stake after sharing and deductions.
- Modeled ROI alpha is not realized ROI.
- The Space does not buy tickets, pool funds, execute wagers, or provide betting advice.

## Machine-readable reading order

For automated consumers, read files in this order:

1. `data/profiles/index.json`
2. `<profile>/manifest.json`
3. `<profile>/summary.json`
4. The method-specific summary JSON
5. The artifact named and hashed by that summary

Reject a comparison when profile key, history cutoff, target draw date, snapshot digest, file hash,
or commitment reconstruction disagree. Never infer a new target from file modification time.

## Frequently asked questions

**Why does the lab say there is no real draw?**  
Because it is a deterministic control. Its example dates and commitments exercise the same code but
are not operator events.

**Why is a real profile still pending after the draw date?**  
The result-refresh workflow has not yet completed all fetch, validation, upload, and health gates.
The old revision is intentionally preserved.

**Why did the score barely move?**  
The leaderboard is cumulative across twelve forward contests. One new draw normally changes the
average only slightly.

**Does a passed transformer/GARCH gate settle the current draw?**  
No. The evidence gate measures historical holdout containment. Settlement requires the target's
official result and, for realized economics, official payout and purchase evidence.

**Which profile should I use?**  
Use the lab to understand mechanics, EuroMillions or NL Lotto for dated pre-draw research, Agent
Arena for already-scored historical evidence, and Pre-draw Set Lab for current artifacts.
