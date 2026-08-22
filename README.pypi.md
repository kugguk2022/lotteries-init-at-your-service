# lotteries-core

`lotteries-core` is an Alpha research package for comparing 12 registered lottery inference
providers under the same ticket budget. It records reproducible inference envelopes, forward-only
evaluation results, and auditable jackpot-sharing ROI assumptions.

It does not predict guaranteed winners, buy tickets, move money, or claim positive expected returns.
Licensed lottery draws are designed to be random and the baseline game is negative-sum.

## Install

```bash
python -m pip install lotteries-core
```

Optional HTTP API and ML ensemble dependencies are deliberately separate:

```bash
python -m pip install "lotteries-core[api]"
python -m pip install "lotteries-core[ml]"
python -m pip install "lotteries-core[ml-full]"  # optional XGBoost + Torch members
```

The base installation requires only NumPy and pandas. PyArrow, statsmodels, matplotlib, seaborn,
scrapers, bundled draw histories, generated ledgers, and research outputs are not shipped.

## Twelve-provider registry

```python
from lotteries_core import registry

print(registry.names())
provider = registry.create("frequency")
```

The registry includes named public entrants, baselines, spectral orientations, guarded/ablation
controls, the co-occurrence level-set strategy, and the optional ML popularity ensemble. Some names
are public competition identities over shared implementations; ablations are explicit controls.

## Traceable ROI

```python
from lotteries_core.protocol import GameSpec
from lotteries_core.roi import JackpotModel, portfolio_roi_trace

spec = GameSpec.euromillions()
tickets = [((1, 8, 19, 34, 47), (3, 11))]
trace = portfolio_roi_trace(spec, tickets, JackpotModel())
print(trace["assumptions"])
print(trace["tickets"][0])
print(trace["summary"])
```

The trace reports the model/schema version, game shape, jackpot assumptions, per-ticket popularity
share, expected co-winners, payout conditional on winning, jackpot probability, and expected ROI.
Use `InferenceEnvelope` alongside it to bind a proposal to its provider, seed, configuration,
framework version, Git commit, and SHA-256 of the training data.

## Competing inference

```bash
python -m pip install "lotteries-core[ml]"
lotto-benchmark --history history.csv --all-providers --budget 25 --holdout 40
```

Evaluation is chronological and equal-budget. Results should be interpreted as research evidence,
not betting advice.

Source, documentation, and issue tracker:
https://github.com/kugguk2022/lotteries-init-at-your-service
