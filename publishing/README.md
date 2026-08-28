# Community benchmark publishing

These bundles publish LottoBench without redistributing third-party lottery histories. They contain
only deterministic synthetic draws, derived benchmark metrics, and the evaluation contract.

Rebuild every mirrored artifact from the repository root:

```bash
python scripts/build_platform_bundles.py
```

The Hugging Face dataset, Hugging Face Space, and Kaggle dataset/notebook must use the same
`benchmark_version` and `dataset_sha256`. A platform upload is a distribution channel, not an
independent validation or endorsement.

Do not add operator-sourced history, payout tables, prospective tickets, or user ledger records to
these folders without a separate redistribution-rights and privacy review.
