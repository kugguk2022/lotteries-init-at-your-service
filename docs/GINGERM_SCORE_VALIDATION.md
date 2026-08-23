# GINGERM score validation

Status: **the claimed `0.78` has not yet been reproduced or attributed to GINGERM from repository
evidence. Do not publish it as GINGERM accuracy.**

## Similar values currently present

| Value | Artifact | Actual meaning | GINGERM score? |
|---:|---|---|---|
| `0.7884615385` | `outputs/euromillions/garchx/garchx_summary.json` | Empirical coverage of a nominal 80% GARCH prediction interval over a 52-draw holdout (41/52 observations inside the interval). | No |
| `0.6826347305` | `outputs/euromillions/arithmetic_branch/branch_summary.json` | GCD branch-rule match rate over 1,003 filtered rows. | No |
| `0.1690` | `docs/wiki/Methods-and-Findings.md` | GINGERM/co-occurrence portfolio pair coverage in a five-draw, equal-budget holdout. | Yes, but it is coverage—not prediction accuracy. |
| `0.840` | `docs/wiki/Methods-and-Findings.md` | GINGERM/co-occurrence portfolio number coverage in that same holdout. | Yes, but it measures how much of the number universe the portfolio touches. |

The prospective ledger currently contains only pending entries and reports no settled results.
Therefore it cannot validate a monthly GINGERM result yet.

## Why `0.7885` is not inference accuracy

An 80% interval is designed to contain approximately 80% of observations when it is calibrated.
Observed coverage of 41/52 (`0.7885`) is close to that nominal target. It does not mean that 78.85%
of lottery numbers, tickets, or draws were predicted correctly. The GARCH component and GINGERM are
different methods and answer different questions.

Similarly, a portfolio can cover a large fraction of the 50-number universe without predicting a
draw. Coverage improves diversification across a fixed ticket budget; it does not change the
mechanical probability of any individual fair-draw ticket.

## Evidence needed to restore the earlier `0.78`

Provide the original output, notebook, spreadsheet, screenshot, or enough information to rerun it:

1. exact metric formula and denominator;
2. start and end dates, draw count, and game-rule period;
3. history available at each prediction time;
4. ticket budget per draw;
5. whether `0.78` was per-number recall, portfolio coverage, interval coverage, classification
   accuracy, payout/ROI, or another metric;
6. random seed and GINGERM configuration/target mode;
7. matched random or signal-off control at the same budget; and
8. confirmation that every draw was strictly held out from fitting and parameter selection.

Once recovered, LottoBench should store the result as a structured artifact with these fields rather
than advertising a context-free “score.” A user-facing claim should name the metric explicitly—for
example, “number coverage 0.78 at 20 tickets over N forward-only draws”—and report its matched-control
difference and uncertainty beside it.
