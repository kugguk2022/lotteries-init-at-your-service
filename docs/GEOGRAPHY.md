# Winner / sales geography: what it can and cannot tell us

You asked whether geographic data about winners can help. It can — but only for a specific,
purpose, and only after careful normalization. This document is the data contract for that feature so
it is built correctly rather than as a correlation-fishing trap.

## The trap to avoid

The single most common mistake with lottery geography is treating **where winners live** as if it
told you **which numbers or tickets will win**. It does not. Winner locations are overwhelmingly a
function of **where tickets are sold**, which is a function of **population, retailer density, and
income** — not of the random draw. A raw map of winners is essentially a population map with noise.
Feeding un-normalized winner geography into a model just teaches it to predict "where lots of tickets
are bought", which has zero bearing on the draw.

## What geography is legitimately good for

1. **Calibrating the crowd-popularity model.** Aggregated, anonymised data about *which numbers or
   patterns players in a region pick* is exactly what the `unpopularity` lever needs. If real pick
   data shows a combination is even more over-picked than our priors assume, its expected
   conditional payout is even lower — and we should avoid it more strongly. This feeds
   `PopularityModel.calibrate_from_counts` (a hook that requires **already-normalised** counts).

2. **Instant-game regional inventory.** For scratch games, remaining top-prize inventory can differ
   by region/retailer. That is real, EV-relevant, public information and feeds
   `roi.InstantGamePool` per region.

Neither use predicts the draw. Both are about *players* and *inventory*.

## Mandatory normalization (the data contract)

Any geographic signal MUST be normalised before it enters a model, or it just re-measures sales
volume:

- **Per-capita:** divide counts by population of the region.
- **Per-retailer / per-sales:** divide by number of ticket sales (or retailers) in the region, when
  available. This is the single most important control.
- **Time-aligned:** align to the same draw window; do not mix rule regimes or jackpot sizes.
- **Anonymised & aggregated:** never individual-level. Region-level aggregates only. No personal data
  of winners is ingested, stored, or modelled — ever.

Concretely, the expected input to `PopularityModel.calibrate_from_counts(spec, counts)` is a length
`main_n` vector of **sales-normalised relative pick rates**, not raw winner counts.

## Sourcing

Prefer official lottery operators' published statistics (prize breakdowns, remaining-prize reports,
retailer lists) and official open-data portals. Do not scrape personal information about winners.
When a source cannot be normalised for sales volume, treat it as unusable for modelling and, at most,
as a sanity-check visual.

## Status

The calibration hook exists and validates shape, but ships as a **no-op** until a properly
normalised, vetted dataset is wired in (see `repurpose.md`, Pass 6). This is deliberate: a wrong
geographic signal is worse than none, because it looks authoritative while measuring the wrong thing.
