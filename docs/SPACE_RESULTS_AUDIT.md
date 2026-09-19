# Space results audit and repair contract

Audit date: 2026-09-19. Baseline: repository commit
`478a8408038bd629782b524d0822c6cf8b4cdba4`; inspected Space artifacts contain EuroMillions history
through 2026-09-18 and NL Lotto history through 2026-09-12. Those are dated observations, not a
promise about a later deployment.

## Finding

The Space mixed different questions: co-occurrence forecast quality, static crowd-sharing
preferences, shortlist coverage, and cash profit. Its headline modeled ROI could not answer the
owner's question about historical returns. The temporal implementation also had target alignment
and tie-selection problems. Correcting those defects improves the validity of the experiment;
it does not establish that the repaired method predicts better.

The owner's reported historical wins and earlier small candidate sets remain **unresolved**, not
disproved. The original dated candidate files, game rules, selected tickets, stake/payout records,
and metric definition are not present in the inspected evidence.

## Evidence scorecard

| Question | Verified baseline evidence | Conclusion / open requirement |
| --- | --- | --- |
| Is −71% the annual realized ROI? | `lotteries_core/roi.py` computes jackpot-only expected value using fixed assumptions. | No. All-tier realized ROI is unavailable without settlements and payouts. |
| Does the leading agent predict most winners? | `scripts/agent_ledgers.py` ranks by modeled sharing EV; its legacy `win_rate_vs_house_pct` counts positive modeled differences. | That ranking measures a sharing preference, not prize prediction. Show actual-match metrics separately. |
| Is the original main-number method faithfully reproduced? | The R reference uses prefix-time main-pair scores and a six-value trailing mean; the hybrid uses main, star and cross pairs. | No. A separately versioned prefix-main baseline remains open. |
| Is 78% verified prize accuracy? | The identifiable `coverage_80=0.7884615385` is GARCH interval coverage, 41/52. | No such prize-accuracy attribution is established. The earlier conversation's source is unresolved. |
| Has a three-year benchmark completed? | Current Space evaluation is a rolling 12-draw replay; the published real-history comparison has 120 draws. | No. Neither is a completed three-year comparison. |
| Are prospective payouts accumulating? | The inspected `ledger/euromillions` contains only pending predictions. | No settled monetary evidence was found; recording/settlement automation is still open. |

## Why −71.41% was almost fixed by construction

[`lotteries_core/roi.py`](../lotteries_core/roi.py) uses these comparison defaults, not live
operator economics:

| Game | Fixed jackpot / ticket price | Fixed other-ticket count |
| --- | --- | --- |
| EuroMillions | €100,000,000 / €2.50 | 50,000,000 |
| NL Lotto | €1,000,000 / €2.00 | 1,000,000 |

It calculates `P(jackpot) × E(jackpot payout given a hit) / ticket price − 1`. It omits lower
prize tiers, actual draw jackpots, actual ticket purchases, and settled prize payments. For
EuroMillions, even assuming no sharing, the fixed model gives:

```text
100,000,000 / (139,838,160 × 2.50) − 1 = −71.3955046%
```

Therefore the observed Crowd Escape value near −71.41% is not a measurement of losing 71% of money
over a year. It is close to this particular jackpot-only model's ceiling. The main leaderboard's
modeled “wins” were comparisons against the same popularity prior the crowd-avoidance agent
optimizes, not independent verification of number prediction.

The required monetary metric is
`(sum(actual payouts) − sum(actual stakes)) / sum(actual stakes)`, reported with game, currency,
dates, all covered prize tiers, and purchase/settlement status. An unpurchased paper portfolio can
have a separately named **hypothetical settled return**, never a cash win. Missing payouts must be
unavailable, not zero. See [`docs/VERSIONED_ROI_BENCHMARK.md`](VERSIONED_ROI_BENCHMARK.md).

## Main sets are not full tickets

The reference [`euromillions_legacy_check.r`](../experiments/euromillions/euromillions_legacy_check.r)
constructs `G` from ten main-number pair counts. Its separate auxiliary-count calculation does not
enter `G`. The temporal hybrid instead combines ten main pairs, ten main–star pairs, and one star
pair. The prefix-time trajectory, score meaning, forecast window, and search universe all changed.

Under the current 50-main / 12-star rules there are 2,118,760 main sets and 139,838,160 complete
5+2 tickets: 66 star combinations per main set. Historical rule eras require their own counts.

| Distinct candidates | Fraction of current main-set universe | Fraction of current full-ticket universe |
| ---: | ---: | ---: |
| 3,000 | 0.141592% | 0.002145% |
| 500,000 | 23.598709% | 0.357556% |
| 1,000,000 | 47.197417% | 0.715112% |

These are mechanical fractions, not estimated predictive advantages. A full-ticket file can also
repeat the same main set with different stars. Every comparison must report both distinct-main
count and distinct-full-ticket count, and distinguish exact main5 from exact full5+2.

## Temporal and tie defects

In the baseline, `_score_series` in
[`providers/temporal.py`](../lotteries_core/providers/temporal.py) built a matrix using all supplied
training history, then rescored those historical draws under that matrix. Each historical draw
contributed its own pairs to its training score: +21 for EuroMillions cross scoring, +15 for
NL Lotto main scoring. An unseen candidate has no such self-contribution.

An independent check on the packaged 2026-09-08 EuroMillions row gives score 462 with its own row
included and 441 with that row removed. The difference is exactly 21. This is an in-sample versus
unseen-candidate alignment problem, not evidence that the outer held-out result was fetched during
training. It is also not the original R prefix trajectory.

The baseline transformer lacked position embeddings, limiting its ability to distinguish the
ordering of the preceding observations. Cross-pair temporal scores also pooled historical star
availability eras without exposure adjustment.

For the inspected 2026-09-22 candidate file, the first 3,000 rows all had `G=567` and all contained
main numbers **1 and 2**. They represented 2,457 distinct main sets. The complete million rows
contained 770,349 distinct main sets, split between 894,765 rows at `G=567` and 105,235 at `G=534`.
The first-row preference came from lexicographic ties, not a learned preference for 1 and 2.
Likewise, a held-out ticket's precise `actual_rank` included arbitrary within-band ordering and
must not be read as calibrated confidence.

## Repair scorecard

| Area | Baseline defect | v2 repair / remaining acceptance |
| --- | --- | --- |
| Score alignment | Historical target includes its own pairs; candidates do not. | Remove the known self-contribution from training labels. Preserve the outer forward-only cutoff and test the +21/+15 identities. |
| Sequence order | No explicit position information in transformer inputs. | Add position embeddings and regression checks. More capacity is not evidence of better results. |
| EuroMillions rule era | Raw cross counts pool different star universes. | Restrict hybrid fitting to the current 12-star era. Keep the older validated history intact for other analyses. |
| Equal-score order | Lexicographic first rows look like model-preferred numbers. | Use reproducible neutral tie ordering; expose score-band/tie meaning and candidate counts. |
| Artifact provenance | Legacy metrics could be mistaken for results of repaired code. | Hybrid schema `2.0.0`, method `v2_loo_modern_era_neutral_ties`; rebuild and revalidate before claiming deployment. |
| Original R lineage | Changed score universe and temporal definition. | **Open:** implement and evaluate a separately versioned prefix-main baseline. v2 is not that reproduction. |
| Cash return | Modeled jackpot-only EV dominates the presentation. | Separate sharing diagnostics from actual matches; keep all-tier ROI unavailable pending payout evidence. |
| Three-year evidence | Rolling replays are not an accumulating sealed outcome experiment. | **Open:** frozen longer walk-forward evaluation and automatic prospective recording/settlement. |

The v2 repair retains retrospective historical scores conditional on the training cutoff, with
leave-one-out self-contribution removed. It does **not** claim a causal-prefix rewrite. Its first
acceptance criterion is correctness and reproducibility, not a higher score. Do not transplant the
legacy 1/12 containment into v2 summaries, choose versions because they retain one attractive hit,
or infer that a code fix must improve retrospective results. Current deployment and performance
must be verified from the exact artifact version, hashes, cutoff, and rebuilt backtest.

## Positive and negative evidence, together

[`publishing/common/real_history_poi_g_results.csv`](../publishing/common/real_history_poi_g_results.csv)
records a 120-draw evaluation of the cross-pair implementation, with its settings in
[`scripts/evaluate_real_history.py`](../scripts/evaluate_real_history.py): window 26, shortlist sizes
20/100/500, and the current 12-star era.

| Test | Observed result | Matched interpretation |
| --- | --- | --- |
| 500-ticket shortlist, exact main5 | 1/120 draws | Exploratory positive result; 0.0283 expected hits under the distinct-main fair null. |
| Same shortlist, exact full5+2 | 0/120 draws | The main5 containment was not a first-prize full-ticket containment. |
| Same shortlist, at least 4 mains | 8/120 draws | Approximately 6.23 expected under the artifact's uniform-shortlist reference. |
| Same shortlist, at least 3 mains | 82/120 draws | Approximately 109.06 expected under that reference; lower breadth despite the isolated exact-main hit. |
| 20 / 100 shortlist, exact main5 | 0 / 0 hits | No exact-main containment in these smaller tested sets. |

For 500 distinct main sets over 120 independent fair draws,
`1 − (1 − 500 / 2,118,760)^120 = 0.0279245` is the unadjusted chance of at least one exact-main
containment. This is **not adjusted for multiple tested sizes, methods or inspected outcomes** and
must not be promoted to proof of a repeatable edge. The at-least-three/four reference is the
artifact's approximation, not an exact same-portfolio probability. Frequency-matched controls are
also available in the source CSV. This test did not evaluate the owner's original 3,000-,
500,000-, or million-main-set method.

## The remembered 78% and the three-year horizon

The closest identifiable repository value is `coverage_80=0.7884615384615384` in
[`garchx_summary.json`](../outputs/euromillions/garchx/garchx_summary.json): 41 of 52 observed scalar
scores inside a nominal 80% interval. It is interval calibration, not 78% correct tickets or prize
draws. There is no proof this is the metric remembered from the earlier conversation. Preserve
that provenance uncertainty; see [`GINGERM_SCORE_VALIDATION.md`](GINGERM_SCORE_VALIDATION.md).

The live twelve-fold replay and the 120-draw artifact are separate from the proposed three-year
prospective benchmark. At the audited revision, only pending entries existed in
[`ledger/euromillions`](../ledger/euromillions). No completed three-year outcome evidence or settled
cash-return series was found. Do not manufacture historical pre-draw commitments retroactively.

## Follow-through

### Rebuilt EuroMillions v2 check (2026-09-19)

The repaired full-universe build was rerun locally against the same verified source snapshot
`0dc156b633eb025e444af8eae21067f27b5dc6d40d9108600007631e8a39e77d`, through
2026-09-18. Of 1,982 source draws, 1,042 are in the declared 12-star era. The 12 held-out
dates are 2026-08-11 through 2026-09-18; each forecast uses only preceding rows.

| Acceptance check | Rebuilt v2 result | Interpretation |
| --- | --- | --- |
| Full-ticket containment | 0/12 versus 1/12 matched uniform control | No demonstrated improvement or predictive advantage. The old 1/12 is not retained as v2 evidence. |
| Evidence gate | Not passed; one-sided p=1 | Research only. A correctness repair need not increase retrospective hits. |
| Next-draw score band | One million selections from 2,392,633 tied tickets at G=275 | No single row is uniquely most likely. Tie keys choose an exact-size subset reproducibly. |
| Target / branches | 2026-09-22; GARCH 278.606335, transformer 275.115680 | Targets are on the modern-era, self-excluded score scale; do not compare raw G with v1. |
| Local regression suite | 181 passed, 1 skipped | Includes brute-force archive/rank consistency, batch-independent ties and same-ticket match accounting. |

Backtest CSV SHA-256: `e3ceca1a3e50ffd2fdf96d2ee9e76d3fd22fa353f7cfec08fdaba0f196e035d5`.
This records local acceptance, not a claim that the live Space has already refreshed. Required
hosted CI and both observed-profile publication checks still govern deployment.

### Remaining experimental work

1. Publish only artifacts generated by the exact verified repair; retain dated legacy evidence.
2. Reproduce the main-only prefix method and compare locked candidate budgets against controls,
   reporting ties, distinct mains, exact-main/full containment and same-ticket prize tiers.
3. Add official payout tables and pre-draw paper/purchased status before interpreting return.
4. Accumulate sealed prospective outcomes without tuning against those outcomes.
5. Recover the user's original files if available, and reconcile rather than dismiss the historical
   results. Additional data helps evaluate a claim; it does not by itself guarantee better ROI.
