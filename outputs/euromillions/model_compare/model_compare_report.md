# EuroMillions Model Comparison

## Classification-style main-number predictors
- Survivors vs random at 95%: none

## POI / regression family
- Survival rule: directional accuracy above 50% with z>1.96 and RMSE below persistence.
- Survivors on the common trailing window: garchx, garchx_alt_vol

## Notes
- `garchx*` and `garch_glm_diagnostics` are compared on the same trailing window.
- `diagnostics3_next` is cut to the same trailing window for fairness, even though it also has a longer full-history next-step series.
- Exact 5/5 accuracy is zero for every directly comparable forward main-number model in the current outputs.
