# Prospective outcome-tracking contract

The repository's primary experiment is whether the owner's co-occurrence level-set method, alone or
coordinated with other providers, produces better tracked outcomes than an equal-budget uniform
random control.

The named entrants are `gingerm` (the owner's co-occurrence method), `spectral_contrarian` (the
contrarian Perron-Frobenius method), and `parallax` (replication-guarded inference with coverage-first
selection). They remain individually selectable through the REST API and outcome ledger. The
`unpopularity` provider is a separate crowd-avoidance strategy; it is not GINGERM.

## Non-negotiable rules

- Generate and record every portfolio before the target draw.
- Use only completed draws available at record time; never drop the most recent completed draw.
- Record the matched random-control portfolio at the same time as the method portfolio.
- Hash the history and complete prediction record before settlement.
- Settle against official results after the draw and retain main/star hit tiers for every ticket.
- Use an official per-draw payout table when reporting monetary portfolio outcomes.
- Treat payout as simulated unless a purchase-receipt hash was recorded before the draw. A hash is
  supporting evidence, not independent verification of purchase or payment.
- Keep tuning/backtests separate from the prospective ledger. Never tune against the sealed ledger.
- Continue for the planned three-year window unless a predefined safety or integrity stop applies.
- Report positive, negative, and inconclusive results. Park the method if the final evidence is
  negative; do not replace outcome evidence with repository stars, contributor activity, coverage,
  or popularity metrics.

## Commands

```bash
# Before the draw; history contains completed draws only.
lotto-track record --history data/euromillions.csv --preset euromillions \
  --draw-key 2026-08-18 --ledger ledger/euromillions --n-sets 20 --ticket-price 2.50 \
  --methods gingerm,spectral_contrarian,parallax

# After the draw. The optional payout table maps tiers, for example {"5+2": 100000000}.
lotto-track settle --ledger ledger/euromillions --draw-key 2026-08-18 \
  --actual-main 4,17,23,38,45 --actual-stars 3,9 \
  --payout-table official-payouts/2026-08-18.json --currency EUR

lotto-track report --ledger ledger/euromillions
```

`lotto-likely-sets` exposes the same generator directly. The provider adapter is available as
`CooccurrenceLevelSetProvider` and can be included in the existing benchmark with
`python -m lotteries_core.benchmark ... --with-cooccurrence`.

## Win meanings

- **Portfolio hit:** a preregistered generated ticket matches an official prize tier.
- **Tracked portfolio prize:** the tier hit valued using that draw's official payout table.
- **Purchased-ticket evidence:** the record contains a receipt hash created before the draw.
- **Verified cash win:** requires external operator/payment confirmation and is not claimed by this
  repository.
