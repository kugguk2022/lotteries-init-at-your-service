# Contributing a Provider

How to enter a new strategy into the competition. This is the supported extension point: the
framework is built so that a new method is **one file plus two registry lines**, and everything else —
forward-only evaluation, equal budget, envelopes, aggregation, the prospective ledger — comes free.

`lotteries_core/providers/parallax.py` is the worked example of an externally contributed provider.
Read it alongside this page.

## The contract

Implement [`InferenceProvider`](../../lotteries_core/protocol.py):

```python
class MyProvider(InferenceProvider):
    name = "my_provider"          # stable, unique; used in envelope provenance and ledger labels
    description = "..."           # mechanism plus its operating scope

    def fit(self, history: pd.DataFrame, spec: GameSpec | None = None) -> "MyProvider":
        """Fit from training history only. Returns self."""
        return self

    def propose(self, spec: GameSpec, budget: int, rng: np.random.Generator) -> ProviderResult:
        """Return exactly `budget` distinct legal tickets, plus a preference score per ticket."""
```

Five rules the framework relies on:

1. **Deterministic** given the same history and the same seeded `rng`. This is what makes envelopes
   reproducible. If your method needs no randomness, `del rng` and say so.
2. **Exactly `budget` tickets**, distinct and legal. Equal budget is what makes the comparison fair;
   a provider that quietly returns fewer is not competing on the same terms. Validate with
   `spec.validate_ticket(t)`.
3. **Forward-only.** `fit` receives only draws before the evaluated one. Never reach for data outside
   the `history` you are given.
4. **Scores are preferences, not probabilities.** Higher means more preferred. They are not comparable
   across providers until the aggregator normalizes them.
5. **Put your provenance in `diagnostics`** — a free-form JSON-serializable dict. Anything a reader
   would need to reproduce or challenge your result belongs there.

## Ship an ablation

The strongest convention in this repository, and the one that makes a contribution trustworthy:
**ship a mode that turns your signal off and changes nothing else.**

Both existing contributed providers do this:

- `PerronFrobeniusProvider(orientation="uniform")` — identical sampler, stationary vector discarded.
- `ParallaxGuardProvider(mode="ablation")` — identical candidate pool and portfolio objective, learned
  residual set to zero.

The reason is that on a fair draw, most apparent gains come from portfolio construction rather than
from any signal. Without an ablation you cannot tell which you built, and neither can a reviewer. With
one, the difference between the two modes *is* your signal's contribution, measured rather than
argued. Both providers' ablations currently match their live modes almost exactly — which is a real
finding, and one neither would have produced without the control.

## Calibrate against the null

If your method claims to have found structure, show it is not noise. `spectral.py` demonstrates the
pattern with `null_tv_band`: simulate fair histories of the same length, compute your own statistic on
each, and report where the observed value falls in that distribution. `parallax.py` demonstrates the
other pattern: refuse to admit a residual unless it replicates across two disjoint halves of history
*and* clears a family-wise threshold.

A statistic without a null is a story.

## Register it

Two edits.

`lotteries_core/providers/__init__.py`:

```python
from .my_module import MyProvider
__all__ = [..., "MyProvider"]
```

`lotteries_core/benchmark.py` — add a `--with-*` flag and append it in `build_providers`.

To enter it into the tracked ledger competition, add one line to `_make_provider` in
`lotteries_core/outcome_tracker.py` and its name to `PROVIDER_METHODS`.

## Test it

Add tests to `tests/test_core_inference.py` (CI-gated). At minimum:

- budget met, tickets legal and distinct;
- determinism — two different seeds give identical output, or the same seed reproduces;
- your signal is detected on a **planted** signal;
- your signal is **rejected** on a fair synthetic history.

That last one matters most. A method that finds something in fair random data is broken, and it is the
cheapest bug to catch.

## Run the gate

```bash
ruff check .
pytest -q
```

Both must pass; CI blocks on both across the whole repository.

## Compare

```bash
python -m lotteries_core.benchmark --history data/euromillions.csv \
    --game euromillions --budget 25 --holdout 40 --all-providers \
    --out outputs/euromillions/competition_benchmark.json
```

Read the results against [Methods and Findings](Methods-and-Findings.md). `unpopularity` is
the one to beat on `unpopularity_lift`, and the sampler-only ablations are the ones to beat on
coverage.

## What will get a contribution rejected

- It claims to predict the draw. See [Project Scope](Scope.md).
- It has no ablation and no null calibration, so its gain cannot be attributed.
- It peeks at data after the evaluated draw, in any form.
- It returns fewer than `budget` tickets, making its numbers incomparable.
- It deletes an inconvenient negative result rather than recording it.

## Entering the tracked competition

```bash
python -m lotteries_core.outcome_tracker record \
    --history data/euromillions.csv --preset euromillions \
    --draw-key YYYY-MM-DD --ledger ledger/euromillions \
    --methods all --n-sets 20 --ticket-price 2.50
```

`--methods all` records every registered entrant against **one shared random control** for that draw,
so the comparison is paired. See [Outcome Tracking](Outcome-Tracking.md) for the rules that keep the
ledger valid — the most important being that predictions are recorded before the draw and never tuned
against afterwards.
