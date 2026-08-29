# Community benchmark publishing

These bundles publish LottoBench without redistributing third-party lottery histories. They contain
only deterministic synthetic draws, derived benchmark metrics, and the evaluation contract.

Published mirrors:

- Dataset: https://huggingface.co/datasets/kugguk/lottobench-community-benchmark
- Leaderboard: https://huggingface.co/spaces/kugguk/lottobench-community-leaderboard

Rebuild every mirrored artifact from the repository root:

```bash
python scripts/build_platform_bundles.py
```

The Hugging Face dataset, Hugging Face Space, and Kaggle dataset/notebook must use the same
`benchmark_version` and `dataset_sha256`. A platform upload is a distribution channel, not an
independent validation or endorsement.

## Publishing to Hugging Face

Rebuilding only refreshes the local bundles; the mirrors are uploaded by hand and will silently
drift until they are pushed. Upload both, always together — a Space rendering a different snapshot
than the dataset it cites is worse than a stale Space.

```bash
huggingface-cli upload kugguk/lottobench-community-benchmark publishing/huggingface . \
  --repo-type=dataset
huggingface-cli upload kugguk/lottobench-community-leaderboard publishing/huggingface-space . \
  --repo-type=space
```

Then confirm the mirrors match this working tree:

- both repos list `data/poi_g_subset_results.csv`,
- `dataset_sha256` in each `data/benchmark_manifest.json` matches `common/benchmark_manifest.json`,
- the dataset card exposes the `history`, `results`, and `poi_g_subsets` configs in the viewer.

Do not add operator-sourced history, payout tables, prospective tickets, or user ledger records to
these folders without a separate redistribution-rights and privacy review.
