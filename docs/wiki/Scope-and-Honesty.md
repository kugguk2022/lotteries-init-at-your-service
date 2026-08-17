# Scope and Honesty

Read this before using or extending anything here. This page is the orientation.

## What this project is

A **research framework** for studying lottery ticket portfolios under a fixed budget, in simulation.

## What it is not

A betting system, a prediction system, or a way to make money. It never pools funds, buys tickets, or
executes wagers.

## The three facts everything else follows from

**1. A fair draw is unpredictable by construction.** EuroMillions, Totoloto, and EuroDreams use
certified, audited random draws. Past draws carry no information about future draws. No model changes
this — not GLM, not gradient boosting, not deep learning, not HMMs, not GARCH, not PageRank. A method
that appears to work on historical draws has found sampling noise, and the way to tell the difference
is a null-calibrated test, not a plausible story. [Methods and Findings](Methods-and-Findings.md)
carries a worked example of exactly that check demoting a method.

**2. The game is negative-sum.** Operators return less in prizes than they take in sales. Expected ROI
per ticket is negative and stays negative. Every ROI number in this repository is negative, and the
only honest goal is *less* negative.

**3. There is exactly one real lever, and it is not about winning.** Jackpots are **shared** among
winners. Choosing combinations the crowd avoids does not change your probability of winning by even a
fraction — it changes how many people you split with *if* you win. That conditional payout is a real,
measurable quantity, and it is what `unpopularity_lift` tracks.

## What follows for how the code is written

- Machine learning is pointed at **player behaviour** or **diagnostics**, never at the draw.
- Evaluation is **forward-only** and at **equal budget**, or it does not count.
- Coverage and conditional ROI are the headline metrics; realized hits are reported as high-variance
  noise.
- Negative and inconclusive results are retained, in the repository, with their evidence.

## The standing experiment

Whether the co-occurrence method beats an equal-budget uniform random control is an **empirical claim
under test**, not an assumption. It is being measured prospectively over roughly three years against a
matched control, with a pre-committed rule to **park the method if the evidence is negative**. See
[Outcome Tracking](Outcome-Tracking.md).

## If you are here to find a winning system

There isn't one, and this repository will not become one. What you can take from it instead: a
worked example of how to test a hypothesis honestly — forward-only evaluation, matched controls,
null calibration, pre-committed stopping rules, and documented demotions. Those transfer to problems
where the signal is real.

## Related reading in this repository

- [`docs/GEOGRAPHY.md`](../GEOGRAPHY.md) — why winner-location data is a population map with noise, and
  the data contract that keeps it from becoming a correlation-fishing trap.
- [`docs/OUTCOME_TRACKING.md`](../OUTCOME_TRACKING.md) — the non-negotiable rules of the ledger.
