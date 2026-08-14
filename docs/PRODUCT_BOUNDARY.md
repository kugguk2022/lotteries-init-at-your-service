# Product boundary

This repository is a free, open-source research toolkit for generating,
comparing, simulating, and exporting lottery combination portfolios. Its output
is research data, not a registered bet and not a prediction that any individual
combination is more likely to be drawn.

## Allowed workflow

```text
historical/public data + local configuration
                 ↓
candidate generation and portfolio optimization
                 ↓
offline simulation against neutral baselines
                 ↓
local CSV, JSON, text, and reproducibility manifests
                 ↓
independent user decision outside this repository
```

The toolkit may:

- fetch and validate public historical draw data;
- generate deterministic or seeded candidate combinations;
- compare portfolio coverage, overlap, and simulated outcomes against neutral
  random baselines;
- export local research artifacts with the configuration, assumptions, and
  source hashes needed to reproduce them.

## Prohibited transaction boundary

The repository must not:

- purchase, register, transmit, or submit lottery entries;
- integrate with betting operators for transaction execution;
- accept deposits, hold funds, pool stakes, or maintain user balances;
- administer syndicates tied to real money;
- collect prizes, divide winnings, or charge a fee linked to winnings;
- store operator credentials, payment credentials, or custody secrets;
- claim guaranteed returns, winning numbers, or improved draw probability for
  an individual combination based on historical draws.

No CLI, API, background worker, or export format may cross this boundary.

## Evidence and language rules

- Every comparison must include an equal-budget neutral random baseline.
- Historical holdout and simulation results must be labelled as historical or
  simulated; they are not evidence of future prediction.
- “Any prize” and gross return must not be presented as profitability.
- Reproducible runs should record the algorithm version, normalized
  configuration, seed, input hashes, candidate-set hash, and output hash.
- Public copy must avoid “AI winning numbers,” “beat the lottery,” guaranteed
  return, and similar unsupported claims.

## Change review

Any proposed payment, betting-operator, pooled-fund, wallet, custody,
prize-settlement, or entry-submission feature is outside the project contract.
It requires an explicit product decision and a separate legal, security, and
human review before code is added. It must never arrive as an incidental
dependency or hidden integration.

Security issues that indicate the executable code has crossed this boundary
freeze autonomous merge and deployment until reviewed.
