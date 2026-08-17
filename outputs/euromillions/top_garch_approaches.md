# Top GARCH Approaches

- Ranking basis: rmse asc, mae asc, interval coverage error asc, rows desc
- Kept outside `legacy`: garchx, garchx_alternative_volatility, garchx_alternative_volatility_v2
- Moved to `legacy`: garch_glm_diagnostics, garch_glm_diagnostics_v2, garchx_alt_vol_smoke_v5, garchx_glm_v2, garchx_smoke_v5, garchx_smoke_v5_short, garchx_v2

## Ranked approaches
- 1. `garchx_alternative_volatility_v2` | rmse=14.8928 | mae=11.9106 | coverage80=0.8077 | coverage95=0.9615
- 2. `garchx_alternative_volatility` | rmse=14.9528 | mae=11.9765 | coverage80=0.7692 | coverage95=0.9808
- 3. `garchx` | rmse=15.0364 | mae=12.0857 | coverage80=0.7885 | coverage95=0.9808
- 4. `garchx_v2` | rmse=15.0433 | mae=12.0693 | coverage80=0.8077 | coverage95=0.9808
