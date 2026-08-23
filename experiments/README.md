# Experiments

This directory preserves the repository's historical lottery-specific research: EuroMillions HMM,
GARCH, branch, scraper and agent work; Totoloto and EuroDreams prototypes; and their regression
tests. It is intentionally excluded from the `lottobench` wheel and from the default core test run.

From the repository root:

```bash
make setup-experiments
make test-experiments
```

To run an old module whose imports use the historical top-level package names, enter this directory
first, for example:

```bash
cd experiments
python -m euromillions.get_draws --help
```

These programs are reproducibility assets, not supported LottoBench APIs. Promote reusable work into
`lotteries_core/` behind the provider protocol, with forward-only tests and an ablation control.
