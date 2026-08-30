# Community benchmark publishing

LottoBench has two deliberately separate public release tracks:

- The Hugging Face and Kaggle dataset mirrors contain deterministic synthetic draws, POI-G
  diagnostics, derived benchmark metrics, and the reproducibility contract.
- The Hugging Face Space contains three derived profiles: a deterministic EuroMillions-format
  control, observed EuroMillions results, and observed Nederlandse Lotto results. Raw operator
  histories are never redistributed.

Published mirrors:

- Dataset: https://huggingface.co/datasets/kugguk/lottobench-community-benchmark
- Agent arena: https://huggingface.co/spaces/kugguk/lottobench-community-leaderboard

Rebuild the synthetic dataset mirrors from the repository root:

```bash
python -m scripts.build_platform_bundles
```

Build the additive extensive POI-G candidate set and its separate fixed-budget ROI selection for
the Hugging Face dataset only:

```bash
python -m scripts.build_hf_poi_g_artifacts
```

This second command writes only under `publishing/huggingface/data`; it does not rebuild or deploy
the Space and does not change any scheduled workflow.

Build all three Space profiles from a validated local multi-game database:

```bash
python -m scripts.build_space_profiles --db data/lotteries.db
```

## Scheduled Space refresh

`.github/workflows/publish-space.yml` runs at 08:15 UTC every Wednesday and Sunday, after the
Tuesday EuroMillions and Friday EuroMillions/Saturday Nederlandse Lotto draw windows. It fetches
fresh histories, rebuilds every profile, runs the output contracts, uploads the Space only after
validation, waits for startup, and checks the public endpoint.

The workflow requires a GitHub Actions repository secret named `HF_TOKEN` with write access to
`kugguk/lottobench-community-leaderboard`. It also supports a manual `workflow_dispatch` run.

Every profile records its source class, date range, retrieval timestamp, normalized history digest,
game shape, agent cohort, budget, holdout, and seed. Pending submissions remain explicitly
unscored and carry snapshot-bound commitment hashes.

## Dataset release

The synthetic Hugging Face and Kaggle bundles should still be uploaded together after running
`scripts.build_platform_bundles`; their `benchmark_version`, `dataset_sha256`, and POI-G artifacts
must match. The observed-lottery Space refresh is independent because it advances after official
draws and publishes derived outputs only.

Do not add raw operator history, payout tables, or user ledger records to public bundles without a
separate redistribution-rights and privacy review.
