# LottoBench

**Auditable benchmarking for lottery strategies.**

LottoBench is an experimental, local Python toolkit for comparing lottery ticket-selection
strategies at equal budget. It provides reproducible game definitions, strategy providers,
fixed-seed portfolios, forward-only evaluation, coverage metrics, a multi-game SQLite store, and an
optional read-only HTTP API.

It does not predict winning numbers, improve the mechanical odds of a fair draw, place wagers, sell
tickets, or provide financial or gambling advice. Lottery participation is ordinarily negative-sum.
See the bundled `NOTICE.md` and `LICENSE` before relying on or redistributing the software.

## Supported scope

**EuroMillions and the primary Netherlands Lotto series are supported end to end.** Retrieval,
normalization, game-separated storage, forward benchmarking, and realized-ROI settlement use the
same public commands and are covered by tests.

For Netherlands Lotto, a playable line is six numbers from 1–45. The reserve number and the
jackpot-machine colour are stored as result metadata; they are not player-selected pools. Lotto XL
and the additional Super Saturday draw are separate series and are not mixed into the primary
`nl-lotto` training history.

Other national games are defined but **not** supported: they have no retrieval adapter, so they
cannot complete the journey below. `lottobench games` labels them explicitly and
`lottobench.game(...)` refuses them with the reason. They move into the supported list only once
each passes the same contract. See `docs/wiki/Backlog.md`.

## Install

```bash
pip install lottobench==0.1.0a5
```

The base install carries the small numerical and retrieval stack needed for everything below.

## The whole journey: two commands

```bash
lottobench fetch --game euromillions --db data/lotteries.db
lottobench benchmark --game euromillions --db data/lotteries.db

lottobench fetch --game nl-lotto --db data/lotteries.db
lottobench benchmark --game nl-lotto --db data/lotteries.db
```

`fetch` retrieves published EuroMillions history, validates every draw against the game's declared
shape, and writes it into the local SQLite database together with its provenance digest.
`benchmark` reads that database and runs every available provider forward-only at equal budget
against the `uniform_random` control.

Nothing else is required — no CSV to supply, no separate scraper setup, no manual data step.

```bash
lottobench games        # what is supported, and what is only defined
lottobench providers    # registered strategies and local availability
```

## Library

```python
import numpy as np
import lottobench
from lotteries_core import storage

definition = lottobench.game("euromillions")
history = storage.read_history("data/lotteries.db", game="euromillions")

provider = lottobench.create("frequency")
provider.fit(history, definition.spec)
result = provider.propose(definition.spec, budget=5, rng=np.random.default_rng(7))
print(result.tickets)
```

The stable public identity is `lottobench`. The lower-level `lotteries_core` namespace remains
available for compatibility and research extensions.

## Optional extras

| Extra | Adds | For |
|---|---|---|
| `api` | FastAPI + uvicorn | `lotto-serve`, the read-only local HTTP API |
| `ml` | scikit-learn, xgboost | the `ml_ensemble` provider |
| `transformer` | PyTorch | the `sequence_transformer` provider |

```bash
pip install "lottobench[api]==0.1.0a5"
```

The optional **LottoBench Analytics API** serves read-only portfolio metrics, validated realized-ROI
summaries, and cumulative ROI evolution:

```bash
lotto-serve
# interactive schema: http://127.0.0.1:8007/docs
```

When deployed on a public HTTPS host, its `/openapi.json` can be imported as a GPT Action. PyPI
installs the server code; it does not host the API or ledger data.

Provider discovery reports honestly whether an optional provider is actually runnable; no fallback
model is ever substituted under the same provider name, and a provider whose dependency is missing
is reported as unavailable rather than failing mid-benchmark.

## Legacy CSV import and export

```bash
lottobench import-csv history.csv --game euromillions --db data/lotteries.db
lottobench export-csv exported.csv --game euromillions --db data/lotteries.db
```

SQLite databases and exported CSV histories are runtime data and are not included in the package.

## Versioned realized ROI

Prospective ledgers can be compared by provider/model version using actual recorded stake and payout:

```bash
lotto-roi report --ledger ledger/euromillions
lotto-roi export --ledger ledger/euromillions --out roi-benchmark.json
lotto-roi validate roi-benchmark.json
lotto-roi compare roi-benchmark.json another-benchmark.json
```

## Causal POI-G candidate subsets

POI-G is exposed separately from the equal-ticket provider protocol because a shortlist is not a
purchased portfolio:

```python
import pandas as pd
from lotteries_core import GameSpec, generate_poi_g_subset
from lotteries_core.roi import JackpotModel

history = pd.read_csv("history.csv")
spec = GameSpec.euromillions()
subset = generate_poi_g_subset(history, spec, subset_size=3_000)

print(subset.size, subset.universe_fraction, subset.reduction_factor)
portfolio = subset.select(budget=20)
modeled = subset.modeled_portfolio_roi(spec, 20, JackpotModel())
```

The subset is ranked from history available at invocation time. For forward tests, call it before
appending the target draw. `modeled_portfolio_roi` is jackpot-tier model output for the bounded
selection only; realized ROI exists only after a preregistered portfolio is settled with actual
cost and payout. Exact enumeration of very large game universes can be computationally expensive.

The same boundary is available as a verifiable CLI bundle:

```bash
lottobench poi-export --game euromillions --db data/lotteries.db \
  --draw-key 2026-09-01 --subset-size 3000 --budget 20 --out poi-run

lottobench poi-settle poi-run --actual-main 1,2,3,4,5 --actual-auxiliary 6,7 \
  --ticket-price 2.50 --payout-table official-payouts.json
```

`poi-export` writes the extensive ranked candidates separately from the fixed-budget inference
envelope. `poi-settle` validates those hashes and prices only the selected budget against an
identically sized deterministic random control. Without a payout table, money and realized ROI
remain missing rather than silently becoming zero.

Candidate bundles intentionally contain generated tickets so another evaluator can inspect them.
Privacy-minimized realized-ROI exports remain separate and contain hashes plus aggregate financial
outcomes, but no tickets, receipt contents, machine identifiers, or user identity.

## Experimental status

Version `0.1.0a5` is an alpha. APIs, data schemas, strategies, and game support may change. A
passing benchmark or a high metric value is not evidence of future draw prediction unless the exact
metric, holdout, data cutoff, ticket budget, baseline, and leakage controls are supplied and
reproducible. Always verify game rules and official results with the relevant operator.
