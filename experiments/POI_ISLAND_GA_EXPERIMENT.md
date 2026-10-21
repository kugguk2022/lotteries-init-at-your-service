# PoI-G × Superlikelihood island-GA experiment

**Status:** executable experimental harness, **no measured outcome yet**. Does not edit the canonical PoI-G provider or publish the Hugging Face Space. The original source currently documents causal PoI's 26-draw target, score distance ranking, and overlap-based portfolio construction. This experiment consumes a previously frozen Superlikelihood candidate pool and does not recreate its undiscovered generator.

## Input contract

- `--pools`: CSV with `draw_key,rank,main_numbers,auxiliary_numbers,score,created_utc`. Example row: `2026-09-01,1,"[1,2,3,4,5]","[1,2]",0.75,2026-08-31T12:00:00Z`. The exact target date must be ISO YYYY-MM-DD. A million **distinct** combinations must appear for every evaluated date; rank must run from 1 to 1000000. Scores are higher-is-better; to preserve PoI G use `-abs(G - causal_target)`. The CSV must originate from the original frozen model and must not include results of the evaluated draw. Verify source provenance separately.
- `--draws`: CSV with `draw_key,ball_1,ball_2,ball_3,ball_4,ball_5,star_1,star_2` sorted by draw date.
- `--payouts`: JSON mapping date to **official per-winning-ticket** tier payouts, e.g. `{"2026-09-01":{"2+1":12.3,"5+2":1000000}}`. Other tiers default to zero, so supply every paid tier. Currency and ticket-price must match operator and draw. For EuroMillions multiple jackpot winners, use the verified prize per winner, not the advertised pooled jackpot.
- `created_utc` is checked conservatively against midnight UTC on target day; callers must also verify actual ticket cutoff and commit hashes. No source-side integrity checks exist yet.

## Run

```bash
python experiments/poi_island_ga.py \
  --pools path/to/frozen_superlikelihood_1m.csv \
  --draws path/to/historical_draws.csv \
  --payouts path/to/official_payouts.json \
  --budget 100 --pool-size 1000000 \
  --ticket-price YOUR_PRICE \
  --seed 20261011 \
  --max-shared 3 --rare-share 0.10 \
  --islands 4 --population 14 --generations 35 \
  --output outputs/poi_islands_result.json
```

Requires Python 3.10+ and NumPy. No external model service needed.

## Comparisons

1. `poi_g_ranked`: first 100 by frozen ranking, without portfolio repair.
2. `uniform_within_pool`: reproducible random 100 from the same frozen million. **Not** a whole-universe fair-draw null.
3. `poi_ga_islands`: seeded islands with the original causal PoI candidate scores, historical least-frequent-main-pair feature, overlap cap penalty (3 shared main balls), and minimum rare-pair share (10%) penalty.
4. `poi_ga_relaxed`: same optimizer, overlap limit increased to four and rare-pair share set to zero.

All constraints in the GA are implemented as **soft penalties** and are reported diagnostically, not guaranteed feasible. This is not an exact recreation of any unavailable historical hand-tuned PoI constraints; restore those verbatim before calling the variants faithful reproductions.

The script rejects missing result days and insufficient historical draw history. It emits per-draw returns, prize-tier counts, winner-in-pool versus winner-in-shortlist, and aggregate ROI. Selection has no access to target draw results, but score provenance still requires independent audit. Use disjoint tuning and evaluation periods. Every subset/strategy must use the same 100-ticket budget per draw. The recorded results are **paper-settled historical comparisons**, not realized user ROI.

## Known next work

- Recover actual frozen Superlikelihood million-candidate files and hash manifests.
- Add full-universe random 100-ticket null and original PoI custom constraint/optimization solver, if recoverable.
- Run multiple independent GA seeds and holdout periods; report uncertainty and per-tier coverage; compare GARCH 80/95 only on compatible frozen candidate pools and identical dates.
- Optimize memory/performance if many million-row pools are processed simultaneously (current version loads them into RAM).
- Add production prize schedule validation and exact submission deadline semantics.
