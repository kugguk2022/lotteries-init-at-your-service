# LottoBench Space Guide

This page is the operational guide for the public LottoBench Space. It explains what each screen
shows, exactly what `PENDING` means, when files change, how to verify a pre-draw publication, and
which conclusions the evidence does and does not support.

The core research question is whether a co-occurrence-based candidate set contains subsequent
winning numbers more effectively than a comparable control. Crowd avoidance is a separate
sharing-cost question. Neither a simulated sharing score nor a shortlist rank is a cash return.

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

- **Actual-match metrics**: measured matches against each held-out draw, with the ticket budget and
  draw count. Main-number matches and full-ticket matches must be reported separately.
- **Modeled sharing uplift / legacy ROI alpha**: jackpot-only modeled expected-value difference
  from the equal-budget null. It is not an observed monetary return.
- **Latest movement**: change in the named cumulative metric when the newest holdout contest is
  added. It is not automatically a change in a purchased portfolio's balance.
- **Legacy consistency / above-null rate**: share of contests with positive modeled sharing uplift,
  not a count of winning prizes. Historical CSV fields named `win_rate_vs_house_pct` use this meaning.
- **Pair reach**: diversity/coverage diagnostic, not a win probability.

Always identify the ranking objective. A leader on crowd-sharing assumptions need not lead on
co-occurrence containment or actual matches. Small movement in a fixed-parameter sharing model can
reflect stable model assumptions, not a failure to learn from another draw. Each point in the
displayed replay is a cumulative mean; a new release also rolls the twelve-draw window forward.

### Why −71% is not the year's return

The legacy EuroMillions economics uses a fixed €100M jackpot, €2.50 ticket price, and 50M other
tickets. It models the jackpot tier only. Even with no sharing,
`100,000,000 / (139,838,160 × 2.50) − 1 = −71.3955%`. Therefore a value near −71.41% does not
measure a historical 71% cash loss, rule out lower-tier winnings, or disprove an earlier profitable
period. NL Lotto has separate fixed assumptions, not live jackpot economics.

**All-tier realized ROI is unavailable until stakes and official payouts are settled.** Cash ROI
is `(total paid prizes − total paid stakes) / total paid stakes`, not the mean of unweighted
per-draw percentages. A paper portfolio's settled hypothetical return is useful but must remain
separate from actual purchases. Missing payout evidence is not a zero prize.

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

A row number is not a calibrated probability. Many tickets share exactly the same score; an
ordering within that tie must not imply the model prefers one of them. Legacy v1 used lexicographic
tie ordering. The inspected September 2026 EuroMillions file's first 3,000 rows consequently all
contained numbers 1 and 2, despite being score-tied.

The v2 repair is identified by schema `2.0.0` and method
`v2_loo_modern_era_neutral_ties`. It removes each training draw's own pair contributions from its
score, adds transformer position embeddings, fits EuroMillions hybrid scores within the 12-star
era, and uses reproducible neutral ties. These are correctness changes, not proven improvements in
prediction. Check the manifest before assuming the downloadable file was rebuilt with v2; never
reuse a legacy containment count as evidence for the new version.

The v2 score sequence still retrospectively scores known historical draws using the matrix at the
training cutoff, excluding each scored draw's own contribution. It is not the original R
prefix-score sequence. A faithful main-only prefix baseline remains a separate open comparison.

The evidence gate and `RESEARCH_ONLY` label concern retrospective/forward containment evidence.
They are independent of the publication lifecycle. A file can be correctly `PENDING` for a target
draw while its research gate remains unpassed.

### Exhaustive pair-density raster

Scores every legal first-prize combination by historical pair co-occurrence. EuroMillions covers
139,838,160 tickets; NL Lotto covers 8,145,060. The Space publishes the top rows and the exact score
distribution. Pair density describes history; it is not a higher mechanical draw probability.

### Main sets versus complete tickets

Under current EuroMillions rules, 2,118,760 five-main sets expand into 139,838,160 complete tickets
when each is paired with 66 possible star pairs. The owner's R reference `G` uses ten main pairs;
the hybrid uses ten main pairs, ten main–star pairs, and one star pair. These are different methods
and different candidate units.

| Candidate count | Coverage if distinct main sets | Coverage if distinct complete tickets |
| ---: | ---: | ---: |
| 3,000 | 0.141592% | 0.002145% |
| 500,000 | 23.598709% | 0.357556% |
| 1,000,000 | 47.197417% | 0.715112% |

These mechanical fractions are not measured predictive advantages. A file with different star
pairs may repeat its main sets. Report both distinct counts, and compare exact main5 with main5,
full5+2 with full5+2. Historical game-rule eras need their own denominators.

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
- The legacy leaderboard's modeled sharing win rate is not prize-winning accuracy.
- A failed or passed retrospective gate is not the result of a completed three-year trial.
- The Space does not buy tickets, pool funds, execute wagers, or provide betting advice.

## What evidence exists so far?

At the 2026-09-19 audit, the current profile shows twelve forward-held-out draws. The separate
published real-history co-occurrence comparison contains 120 draws, not three years. Its
500-ticket shortlist contains exact main5 on 1/120 draws and exact full5+2 on 0/120. At least-four
main containment is 8/120 versus approximately 6.23 expected under the published reference;
at-least-three containment is 82/120 versus approximately 109.06 expected. The exact-main hit is
an exploratory positive result, without adjustment for multiple tested sizes or methods, not proof
of profitable jackpot prediction.

The inspected prospective ledger contains pending predictions but no settled outcome files.
Scheduled publication of a refreshed twelve-draw replay does not itself build the three-year
settlement ledger. Evidence status should advance only with verified new records.

The identifiable repository value near 78% is 41/52 scalar scores inside a nominal 80% GARCH
prediction interval. That is interval coverage, not prize accuracy. Whether it is the result
remembered from an earlier discussion remains unresolved. Original historical wins or candidate
containments need their dated artifacts and metric definitions to be reconciled; they are not
disproved by the absence of those files from this repository.

The source-backed audit and before/after acceptance scorecards are in
[`docs/SPACE_RESULTS_AUDIT.md`](https://github.com/kugguk2022/lotteries-init-at-your-service/blob/main/docs/SPACE_RESULTS_AUDIT.md).

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
First identify the score. A fixed-assumption sharing score can remain stable without saying
anything about prediction. Actual-match metrics also fluctuate within a small twelve-draw replay;
the displayed cumulative averages can mask per-draw movement.

**Does more data guarantee better ROI?**

No. It enables stronger tests and potentially better models of a real signal; it does not create
signal automatically. Judge a change by frozen, equal-budget comparisons, prospective records and
proper payout accounting. Report improvements and regressions together.

**Where is my year's profit/loss?**

It is unavailable without a complete stake-and-payout ledger. The jackpot-only modeled percentage
is not a substitute, and a contained winning combination in an unpurchased file is not a cash win.

**Does a passed transformer/GARCH gate settle the current draw?**  
No. The evidence gate measures historical holdout containment. Settlement requires the target's
official result and, for realized economics, official payout and purchase evidence.

**Which profile should I use?**  
Use the lab to understand mechanics, EuroMillions or NL Lotto for dated pre-draw research, Agent
Arena for already-scored historical evidence, and Pre-draw Set Lab for current artifacts.
