# Outcome Tracking

The repository's central experiment. Everything else is machinery in service of this.

The binding rules are in [`docs/OUTCOME_TRACKING.md`](../OUTCOME_TRACKING.md); the implementation is
[`lotteries_core/outcome_tracker.py`](../../lotteries_core/outcome_tracker.py). This page explains why
the ledger is built the way it is.

## The claim under test

> Coordinated, coverage-and-co-occurrence-aware set selection produces better tracked outcomes than an
> equal-budget uniform random control.

That is an empirical claim, and the only honest way to judge it is to **record predictions before each
draw and score them after**, over a long window, then read off whether an edge exists — and **park the
project if it does not**.

Against a fair draw the expected edge is zero. The point is to measure, not to assume.

## Why "before the draw" is the whole design

It is trivially easy to find a method that would have done well on draws you have already seen. The
ledger removes that possibility structurally rather than asking for good intentions:

- The portfolio is generated from history that **does not contain** the target draw.
- A **matched uniform-random control** portfolio is generated at the same moment, seeded from the draw
  key, so "did it beat luck?" is always answerable.
- The complete prediction record — tickets, control, config, and a SHA-256 of the training history —
  is hashed and appended to `pending_predictions.jsonl` before the draw.
- On settlement the record's hash is **re-verified**. A record that was altered after the fact fails
  the integrity check and refuses to settle.

Tuning and backtesting are kept strictly separate from the ledger. Never tune against the sealed
ledger — doing so converts a prospective experiment back into a retrospective story.

## The workflow

```bash
# 1. BEFORE the draw. History must contain completed draws only, and must not drop the most recent one.
lotto-track record --history data/euromillions.csv --preset euromillions \
    --draw-key 2026-08-18 --ledger ledger/euromillions --n-sets 20 --ticket-price 2.50

# 2. AFTER the draw, with the official result.
lotto-track settle --ledger ledger/euromillions --draw-key 2026-08-18 \
    --actual-main 4,17,23,38,45 --actual-stars 3,9 \
    --payout-table data/prizes.json

# 3. Any time.
lotto-track report --ledger ledger/euromillions
```

> Refresh the history first. The bundled CSV ends 2025-08-12; recording against stale history means the
> method never sees the draws immediately preceding the one it is predicting.

## Ledger files

| File | Contents |
|---|---|
| `pending_predictions.jsonl` | Recorded, not yet drawn. Each record carries `record_sha256`. |
| `settled_predictions.jsonl` | Settled records, with the actual result attached. |
| `results.csv` | One scored row per settled draw: method vs control, per tier, with prizes and net return. |

## Reading the report

```
settled draws              : 12
method mean best-main hits : 2.417
control mean best-main hits: 2.250
mean per-draw lift (method-control, mean-main): +0.0183
one-sided permutation p (lift>0): 0.3126
verdict: INSUFFICIENT DATA (12 draws) — keep tracking; a 3-year window is the goal.
```

The lift is a **paired** comparison — method minus control on the same draw — and the p-value comes
from a sign-flip permutation test over those paired differences. Pairing matters: an unusually
"coverable" draw helps both portfolios, and pairing cancels that out.

### The verdict ladder

| Condition | Verdict |
|---|---|
| fewer than 30 settled draws | `INSUFFICIENT DATA` — keep tracking |
| mean lift ≤ 0 | `NO EDGE vs random so far` — trending toward PARK |
| lift > 0 and p < 0.05 | `POSITIVE SIGNAL` — keep tracking to confirm stability |
| lift > 0 and p ≥ 0.05 | `WEAK/INCONCLUSIVE` — not yet distinguishable from luck |

The 30-draw floor exists so that an early lucky run cannot be read as a result.

## Money

The tracker handles **none**. `--ticket-price` enables prospective net-return arithmetic so the
negative-sum reality stays visible in the numbers. `--ticket-proof-sha256` records a hash of a purchase
receipt if one exists; the receipt itself is never stored, and a hash is supporting evidence, not
independent verification of purchase or payment. Payouts are treated as **simulated** unless such a
hash was recorded before the draw.

## The pre-committed stopping rule

Report positive, negative, and inconclusive results alike. If the final evidence over the window is
negative, **park the method**. Do not substitute repository stars, contributor activity, test coverage,
or popularity for outcome evidence.

Writing the stopping rule down before the data arrives is what makes the rest of it worth reading.
