# Repository Map

## Shipped to PyPI

| Path | Purpose |
|---|---|
| [`lottobench/`](../../lottobench/) | Stable public namespace, country game definitions, and local data CLI. |
| [`lotteries_core/`](../../lotteries_core/) | Provider protocol, canonical registry, aggregation, forward evaluation, ROI, provenance, storage, ledger, and optional API. |
| [`README.pypi.md`](../../README.pypi.md) | PyPI project description. |
| [`pyproject.toml`](../../pyproject.toml) | Package metadata, dependencies, entry points, and tool configuration. |

Only `lottobench*` and `lotteries_core*` are selected by the build configuration.

## Core validation and maintenance

| Path | Purpose |
|---|---|
| [`tests/`](../../tests/) | Tests for the shipped package and API. |
| [`scripts/doctor.py`](../../scripts/doctor.py) | Actionable environment preflight. |
| [`scripts/validate_user_journey.py`](../../scripts/validate_user_journey.py) | Offline registry → benchmark/ROI → SQLite/provenance journey. |
| [`scripts/check_distribution.py`](../../scripts/check_distribution.py) | Rejects missing package files and leaked data/experiments. |
| [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) | Python 3.10–3.14 core matrix, separate experiment tests, and wheel smoke test. |

## Preserved research

[`experiments/`](../../experiments/) contains the lottery-specific scrapers, EuroMillions models,
HMM/GARCH/branch work, agents, R files, launch scripts, and their tests. These files remain available
for reproduction but are excluded from the wheel and the default core test discovery.

Reusable work graduates from `experiments/` into `lotteries_core/` only after it implements the
provider protocol, has forward-only evaluation, includes an appropriate ablation, and carries tests.

## Runtime and evidence

| Path | Purpose |
|---|---|
| `data/` | Local histories and migration inputs; database/CSV artifacts are not packaged. |
| `ledger/` | Prospective predictions and settled ROI evidence. |
| `outputs/` | Historical benchmark/research artifacts; excluded from distributions. |
| `docs/` | Scope, publishing, evidence, and project wiki documentation. |
