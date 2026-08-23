# Residual Model Comparison

Cutoff start date: 2016-09-27

## Training / Shared-target comparison

- Auto prime-pruned shared RMSE: 22.7728
- Mixture prime-pruned shared RMSE: 22.7508
- Delta (mixture - auto): -0.0220
- Auto predicted next POI/score: 280.769 / 281
- Mixture predicted next POI/score: 282.279 / 282

## Shortlist holdout comparison

- Holdout steps: 3  top_n: 25
- Auto branch recall@5: 0.2000
- Mixture branch recall@5: 0.2000
- Diagnostics3 recall@5: 0.1333
- Mixture minus auto recall@5: 0.0000
- Mixture minus diagnostics3 recall@5: 0.0667

## Model-only walk-forward holdout (12 steps)

- Auto classic/prime-pruned RMSE: 23.6414 / 23.9722
- Mixture classic/prime-pruned RMSE: 23.6114 / 23.9635
- Delta classic RMSE (mixture - auto): -0.0300
- Delta prime-pruned RMSE (mixture - auto): -0.0087
