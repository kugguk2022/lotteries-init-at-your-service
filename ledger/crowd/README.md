# Winner-count ledger (crowd-forecast track)

`euromillions_tier_winners.csv` holds, per draw: the drawn numbers, the advertised jackpot, the
total entries across all participating countries, and the total winners in each of the 13 prize
tiers. It is the ground truth for `lotteries_core.crowd_forecast`.

| column | meaning |
| --- | --- |
| `draw_date` | ISO date of the draw |
| `ball_1..ball_5`, `star_1..star_2` | drawn numbers, ascending |
| `jackpot_eur` | advertised 5+2 jackpot in EUR |
| `entries` | total lines entered, all countries |
| `w_A_B` | total winners matching exactly A main numbers and B stars, all countries |

## Provenance and limits

- 42 draws between 2024-06-25 and 2026-11-06, read on 2026-11-09 from the public per-draw result
  pages at `https://www.euro-millions.com/results/DD-MM-YYYY` ("Total Winners" column and
  "Total entries").
- The sample is **not contiguous**: it is the set of draw pages that could be retrieved in one
  session. Forward-only evaluation still only ever trains on earlier dates.
- Values were transcribed by an automated page reader, not parsed from raw HTML. The loader
  rejects rows whose winners/entries ratio is implausible, but individual cells have not been
  checked against the operator. Re-pull through a proper adapter before citing any single figure.
- Whether to keep redistributing this table is a maintainer decision (the Space currently does not
  redistribute raw observed histories).
