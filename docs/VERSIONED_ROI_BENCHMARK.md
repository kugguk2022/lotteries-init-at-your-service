# Versioned Realized-ROI Benchmark

LottoBench evolves providers by comparing prospectively recorded, settled outcomes—not by replacing
missing evidence with a backtest or modeled expected ROI.

## Evidence lifecycle

1. `lotto-track record` records a portfolio before its draw. It stamps the LottoBench version,
   provider version, provider-configuration hash, history hash, and equal-budget cohort ID.
2. `lotto-track settle` records the official result, payout-table hash, actual stake and payout,
   settlement source, realized ROI, and an integrity hash over the result.
3. `lotto-roi report` groups results by game, currency, provider version, and configuration.
4. `lotto-roi export` creates a deterministic privacy-minimized JSON bundle.
5. `lotto-roi compare` validates and combines bundles without silently mixing model versions.

## Versioning rule

Each canonical provider has an explicit algorithm version in `lotteries_core.registry.ProviderSpec`.
Maintainers must bump it whenever code or defaults can change generated portfolios. Configuration-only
changes remain separated by `provider_config_sha256`. Old CSV ledgers without these fields remain
readable but are labeled `legacy-unversioned`.

## ROI definition

For a provider/version cohort:

```text
realized ROI = (sum(actual payouts) - sum(actual stakes)) / sum(actual stakes)
```

Aggregation uses the ratio of totals, not the unweighted mean of per-draw percentages. Comparisons
are separated by game and currency. `settled_draws` and `settled_records` are always displayed so a
high-variance early result is not presented without its sample size.

Every prediction also carries the repository's deterministic equal-budget random portfolio for the
same draw. Exports therefore report `control_realized_roi` and `realized_roi_lift`, preventing a
provider from receiving credit merely because it happened to be evaluated during unusually generous
or unusually poor draws.

## Export boundary

Bundles include draw ID, game, provider/model/configuration versions, cohort and provenance hashes,
cost, payout, currency, ROI, verification label, and timestamps. They exclude generated tickets,
receipt contents, names, email, location, IP address, and machine identifiers.

LottoBench performs no automatic submission or telemetry. A future central registry can accept these
bundles explicitly and append them to a versioned evidence dataset after schema and integrity checks.
