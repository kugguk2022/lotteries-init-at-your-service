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
- Use an official per-draw payout table when reporting monetary portfolio outcomes. A draw
  settled without one records NaN prize, net return and ROI -- never a zero prize -- and
  `report` sums money over the draws that have a table, stating how many that is.
- Treat payout as simulated unless a purchase-receipt hash was recorded before the draw. A hash is
  supporting evidence, not independent verification of purchase or payment.
- Keep tuning/backtests separate from the prospective ledger. Never tune against the sealed ledger.
- Continue for the planned three-year window unless a predefined safety or integrity stop applies.
- Report positive, negative, and inconclusive results. Park the method if the final evidence is
  negative; do not replace outcome evidence with repository stars, contributor activity, coverage,
  or popularity metrics.

## Commands

```bash
# Before the draw; history contains completed draws only. Game, history store, ledger path
# and ticket price all default to the tracked EuroMillions setup, so a scheduled run needs
# no flags beyond the draw it is recording.
lotto-track record --draw-key 2026-08-18 --methods gingerm,spectral_contrarian,parallax

# The same run spelled out, for any other game or a one-off:
lotto-track record --history data/euromillions.csv --preset euromillions \
  --draw-key 2026-08-18 --ledger ledger/euromillions --n-sets 20 --ticket-price 2.50 \
  --methods gingerm,spectral_contrarian,parallax

# After the draw. The payout table maps tiers, for example {"5+2": 100000000}; "5_2" keys are
# accepted too, because published breakdowns use both spellings.
lotto-track settle --draw-key 2026-08-18 --actual-main 4,17,23,38,45 --actual-stars 3,9 \
  --payout-table official-payouts/2026-08-18.json

lotto-track report
```

Settlement is safe to repeat: an already-settled draw is skipped, result rows are keyed on
`(draw_key, method)` so a retried or crashed run cannot duplicate them, and the ledger's JSONL
files are rewritten atomically. Use `--force` to re-score a draw whose result was entered
wrongly -- it replaces that draw's rows in place. Pass `--payout-source approximate` when the
table is a static estimate rather than the operator's published breakdown.

`lotto-likely-sets` exposes the same generator directly. The provider adapter is available as
`CooccurrenceLevelSetProvider` and can be included in the existing benchmark with
`python -m lotteries_core.benchmark ... --with-cooccurrence`.

## Win meanings

- **Portfolio hit:** a preregistered generated ticket matches an official prize tier.
- **Tracked portfolio prize:** the tier hit valued using that draw's official payout table.
  Without a table the prize is recorded as missing (NaN), not as zero, so the ROI aggregates
  in `report` cover only the draws that actually have prize data.
- **Purchased-ticket evidence:** the record contains a receipt hash created before the draw.
- **Verified cash win:** requires external operator/payment confirmation and is not claimed by this
  repository.

## Ledger integrity

Each record carries two independent digests:

- `record_sha256` covers the preregistration -- the portfolio, its matched control, the history
  hash and the provider identity, as they stood *before* the draw. Settlement never changes it.
- `settlement_sha256` covers the scoring inputs -- the official numbers, the payout table hash
  and its provenance, and the resulting `result_sha256`.

That split is what lets a settled record still validate: scoring a draw adds a `settlement`
object instead of mutating the preregistered one. Records settled before the split carry no
`settlement` block and validate on `record_sha256` alone.

The ledger directory is committable evidence -- `ledger/**/*.csv` is exempt from the
repository's `*.csv` ignore rule -- and `.gitattributes` gives the append-only ledger files a
union merge, so two settlements landing on the same file do not conflict.
