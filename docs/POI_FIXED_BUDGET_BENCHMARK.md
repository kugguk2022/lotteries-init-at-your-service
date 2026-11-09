# PoI at a fixed ticket budget

The primary Space comparison now evaluates the owner's main-pair PoI statistic,
not a GARCH interval containing most of the legal ticket universe.

For a training cutoff t, W(a,b) counts earlier draws containing both main numbers.
G(ticket) sums W over its main-number pairs: ten pairs for EuroMillions, fifteen
for NL Lotto. This preserves the main-pair statistic in the original R approach.
It does not claim the R code itself supplied a pre-draw forecast target.

The existing repository's forecast extension supplies the target: the mean G of
the last 26 training draws, rescored under W at the current training cutoff.
We rank by absolute distance to that target, then descending G. Equal scores
use a versioned seeded SplitMix64 order of the full-ticket ID. EuroMillions stars
are included in the ticket and tie order but do not contribute to this original
main-pair statistic. This is distinct from cross-pool GINGERM, PoI island-GA and
the joint Transformer/GARCH model; their results must not be relabeled as this model.

## Evaluation contract

- Predeclared budgets: 12, 100, 1,000 and 1,000,000 unique full tickets. Every smaller
  set is an exact prefix of the largest ranking. No expanding a range after seeing a result.
- Up to 52 held-out draws, with at least 26 earlier training draws. EuroMillions
  uses only the fixed 12-star era starting 2016-09-27. NL Lotto uses six player
  numbers; its reserve number is not a seventh selectable number.
- Each ranking receives only rows strictly before the target. Outcomes enter only
  settlement. Historical replays are explicitly retrospective, even though fitting
  is forward-only. They are not public pre-draw commitments.
- Full-ticket winner containment, >=3 mains, >=4 mains, and all-main containment
  are reported separately. Main-only matches are not official prize tiers.
  Full matching counts and same-ticket main/star tier counts remain downloadable.
- The random comparator is exact: uniformly selected unique legal full tickets
  at the identical budget. Expected matching-ticket counts are B*M/U; the chance
  of at least one match is 1 - C(U-M,B)/C(U,B). We do not approximate sampling with
  replacement or compare one million tickets to a 12-ticket null.
- Wilson intervals describe variability across draws; they are not adjusted for
  multiple budgets/metrics or proof of a predictive edge. All outcomes, including
  zero wins and underperformance, remain visible. The fixed seed is not optimized
  on these results. Future method changes require new versioned cohorts.
- ROI stays null without verified all-prize payouts, ticket prices and stakes.
  A winning ticket inside a large candidate space is not proof of profitability.

## Reproduce

Use a validated local snapshot; raw operator histories are not republished:

```sh
python -m scripts.build_poi_benchmark --game euromillions \
  --history /path/to/lotteries.db --out /path/to/poi-euromillions
python -m scripts.build_poi_benchmark --game nl-lotto \
  --history /path/to/lotteries.db --out /path/to/poi-nl-lotto
pytest -q tests/test_poi_benchmark.py tests/test_poi_g.py tests/test_poi_g_artifacts.py
```

The manifest records source code hashes, history hash/cutoff, seed, target, exact
size, and download hashes. The replay records a digest of the full ranked candidate
IDs for each budget and draw. Reuse requires matching configuration, training and
outcome digests and intact row hashes; corrected source history forces recomputation.

The next set contains one million rows, with a separate 1,000-row preview. The
first 12/100/1,000 rows are the corresponding fixed-budget selections. Original
public Hugging Face commits retain historical downloads; a generation timestamp
alone never proves that a set was publicly available before its draw. Until an
independent publication receipt is checked, the next-set status explicitly remains
unverified. Same-day or older reconstructions are labeled late.

The scheduled profile builder generates and validates these artifacts before
publication. Existing range/Transformer/Crowd Escape outputs and required gates
remain intact; broad ranges move into a secondary diagnostics accordion.

## Remaining experiments

Island-GA is a separate shortlist optimizer requiring its own matched-budget,
multi-seed evaluation. It is not silently substituted for PoI. Frequency-matched
controls would help distinguish pair structure from marginal-number concentration.
Neither an absent model nor an unevaluated experiment is classified as weak.
