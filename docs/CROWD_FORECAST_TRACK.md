# Crowd-forecast track and all-tier payout model

Two additions that move LottoBench from "can anything beat random at hitting numbers?" (no, by
construction) to a target where forecasting skill exists and can be measured against ground truth.

## 1. Crowd-forecast track (`lotteries_core/crowd_forecast.py`)

**Target.** For each draw the operator publishes how many tickets won each prize tier. Given the
drawn numbers and total entries, forecast those counts.

**Null.** `uniform_crowd`: everybody picks uniformly, so expected winners are
`entries x hypergeometric probability`.

**Protocol.** Forward-only. For each evaluated draw the entrant sees only earlier draws, commits a
crowd model, and is scored on the held-out draw. Skill is `1 - loss / null loss` on a
precision-weighted log error, with a paired bootstrap 95% interval over draws.

**Entrants shipped.**

| entrant | what it is |
| --- | --- |
| `uniform_crowd` | the null |
| `static_prior` | the hand-set `PopularityModel` that drives the current Crowd Escape agent |
| `fitted_crowd` | product-form number weights + lucky-dip mixture, fitted to winner counts |

**First real result** (42 EuroMillions draws 2024-06-25..2026-11-06, 22 evaluated forward-only;
`docs/crowd-forecast-evidence-2026-11-09.json`):

| entrant | skill vs null | 95% CI | typical miss on large tiers | calibration slope |
| --- | --- | --- | --- | --- |
| `fitted_crowd` | **+0.54** | +0.24 .. +0.71 | 11% | 0.83 |
| `uniform_crowd` | 0 | - | 17% | - |
| `static_prior` | -10.6 | -20.5 .. -5.1 | 59% | 0.11 |

Reading: a fitted model beats the uniform null out of sample with an interval that excludes zero.
The static prior points the right way (low numbers are over-picked) but overstates the effect by
roughly nine times, so it predicts winner counts far worse than assuming a uniform crowd.

**Controls in the test suite.** A planted crowd is recovered and beats the null; a truly uniform
crowd yields no skill (the model does not manufacture an edge from noise); the loader rejects
implausible rows.

## 2. All-tier payout model (`lotteries_core/tier_payout.py`)

Every EuroMillions tier is pari-mutuel, so prize-per-winner is `tier fund / winners`. The module
prices a ticket across all 13 tiers under a crowd model, the advertised jackpot and the entry count,
and reports alpha in percentage points over uniformly random tickets under the same assumptions.

Median economics of the sample (jackpot EUR 73.6M, 22.8M entries), 12-ticket budget:

| agent | modeled ROI | alpha vs random | alpha after calibration haircut |
| --- | --- | --- | --- |
| `uniform_random` | -62.7% | +0.1 pp | +0.1 pp |
| `unpopularity` (static prior) | -59.3% | +3.5 pp | +2.9 pp |
| `calibrated_escape` (new) | -57.2% | +5.6 pp | +4.7 pp |

The haircut multiplies alpha by the out-of-sample calibration slope (0.83).

## Claims boundary

- No entrant changes the probability that a ticket wins. The edge is payout conditional on winning.
- ROI stays strongly negative. Alpha means "less negative than random", not profit.
- Payout figures are modeled expected values, graded by a crowd model that is itself validated on
  winner counts. They are not settled cash results.
- `fund_per_line = 1.10 EUR` is an assumption (checked against the long-run average 2+0 prize).
  Absolute ROI depends on it; alpha between agents much less.
- 22 evaluation draws from a non-contiguous sample is a first result, not a final one. The
  calibrated agent picks the numbers with the lowest *estimated* popularity, which biases its
  modeled edge upward; the haircut does not remove that selection effect.

## Not done yet

- A raw-HTML adapter for winner counts in `lotteries_core/sources/` and a scheduled refresh. The
  current ledger was transcribed by a page reader (see `ledger/crowd/README.md`).
- Registering `calibrated_escape` in `lotteries_core/registry.py` and the Space bundle. Left out
  on purpose so existing regression fixtures and published leaderboards are untouched.
- NL Lotto: needs its own tier table and the operator's winner counts.
- Settled cash check: compare modeled prize-per-winner with published prize amounts per draw.

## Run it

```bash
python scripts/build_crowd_forecast.py --out outputs/crowd_forecast/euromillions
python -m pytest -q tests/test_crowd_forecast.py
```
