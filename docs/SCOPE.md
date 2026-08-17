# Project Scope

This repository is a research and simulation framework for EuroMillions, Totoloto, and EuroDreams.

> **Note:** We do not play. We supply strategies that may improve outcomes for some people,
> somewhere, in the future.

## Included

- Fetching, normalizing, and validating historical draw data.
- Generating candidate portfolios under a fixed simulated ticket budget.
- Comparing strategies with forward-only, equal-budget benchmarks and matched controls.
- Measuring coverage, diversity, hit tiers, payout, net return, and conditional jackpot-sharing effects.
- Recording reproducible pre-draw predictions and settling them against official results.
- Supporting local, file-based exchange of inference results between providers.

## Excluded

- Pooling or holding user funds.
- Purchasing tickets or executing wagers.
- Operating a betting service.
- Presenting unvalidated results as guaranteed winnings or an established predictive edge.
- Treating repository popularity or contributor activity as experimental evidence.

## Evaluation boundary

Draw prediction and positive net return are research objectives, not established results. Strategies
must be evaluated prospectively or with forward-only holdouts, at equal budget, against matched
controls. Negative and inconclusive results remain part of the record.

## Current delivery boundary

Execution remains outside this repository. The software may calculate simulated payouts and record
prediction hashes, but a cash win requires independent confirmation from the relevant lottery
operator or payment record.

See [`OUTCOME_TRACKING.md`](OUTCOME_TRACKING.md) for the evaluation ledger and
[`GEOGRAPHY.md`](GEOGRAPHY.md) for geographic-data constraints.
