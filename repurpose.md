# Repurpose: from lottery playground to a distributed-inference research framework

This document is the **architectural contract** for the repository's pivot. It is intentionally
conservative: it preserves the existing, already-validated experiments while introducing a common
substrate for the new research question.

## The research question (stated honestly)

> Can *coordinating many independent inference strategies* — run in a distributed way — improve
> **combinatorial coverage** and **expected return-per-ticket** under a **fixed ticket budget**,
> relative to any single strategy spending the same budget?

That is the whole claim. In particular, this project does **not** claim, and its code does not try,
to predict the outcome of a fair random draw. See [`docs/SCOPE_AND_ETHICS.md`](docs/SCOPE_AND_ETHICS.md).

## What is preserved

Nothing validated is thrown away. The following remain the source of truth and are *wrapped*, not
rewritten:

- the typed EuroMillions data API (`euromillions/get_draws.py`, `schema.py`, `infer.py`);
- the forward-only walk-forward baseline (`run_all.py`);
- the arithmetic-branch strategy and its holdout benchmarks (`euromillions/arithmetic_branch.py`,
  `branch_shortlist_benchmark.py`);
- the GARCH/GLM diagnostics and the POI/Sobol lab (`euromillions_agent/`).

Regression fixtures pin current behaviour *before* any physical reorganization, so a refactor that
changes a validated number fails loudly.

## What is added (`lotteries_core/`)

A small, dependency-light package that is the common substrate:

| Module | Responsibility |
| --- | --- |
| `protocol.py` | `GameSpec` (combinatorial shape) and `InferenceProvider` (the interface every strategy implements). |
| `envelope.py` | `InferenceEnvelope` — a reproducible, provenance-carrying container of proposed tickets. The unit of exchange between nodes. |
| `popularity.py` | A model of how *other players* pick numbers (the basis of the only valid draw-game lever). |
| `roi.py` | Expected return-per-ticket under pari-mutuel jackpot sharing, plus the instant-game remaining-prize EV model. Simulation-only. |
| `coverage.py` | Combinatorial coverage and diversity metrics (first-class objectives). |
| `aggregation.py` | Deterministic, diversity-aware, equal-budget aggregation across envelopes — the coordination step. |
| `evaluation.py` | Forward-only, equal-budget benchmark. |
| `providers/` | `frequency`, `unpopularity`, and a `ml_ensemble` (GLM + gradient-boosting + optional deep MLP). |

## Key decisions

1. **Preserve validated experiments before physical reorganization.**
2. **Forward-only, equal-budget evaluation** is the only accepted way to claim "better".
3. **Provenance and reproducibility are mandatory** — every envelope records seed, data hash, git SHA,
   config, and framework version, and reproducibility is *checked*, not assumed.
4. **Diversity / coverage are first-class metrics**, not afterthoughts.
5. **Distributed operation starts as local / file exchange.** Nodes write envelopes to a shared
   directory; a coordinator reads and aggregates them.
6. **Networking is added only after deterministic aggregation over file envelopes is validated.**
7. **Financial outcomes are simulation-only.** No pooled funds, no ticket purchasing, no wager
   execution, and no claim that distributed inference guarantees increased winnings.

## Product boundary (non-negotiable)

- ❌ No pooling of user funds.
- ❌ No ticket purchasing or wager execution.
- ❌ No claim of guaranteed or positive expected winnings.
- ✅ Reproducible, forward-only simulation and measurement.
- ✅ Coverage, diversity, and expected-conditional-ROI as the reported quantities.

## Implementation passes

- [x] **Pass 0 — contract & scope.** This document, `docs/SCOPE_AND_ETHICS.md`, `docs/GEOGRAPHY.md`,
      README scope banner.
- [x] **Pass 1 — protocol & envelopes.** `GameSpec`, `InferenceProvider`, `InferenceEnvelope`.
- [x] **Pass 2 — the honest ROI lever.** Popularity model, expected-conditional-ROI, coverage,
      diversity-aware equal-budget aggregation, instant-game remaining-prize EV.
- [x] **Pass 3 — wrap strategies as providers.** Frequency baseline, unpopularity, GLM+GBM(+DL) ensemble.
- [x] **Pass 4 — forward-only equal-budget benchmark + regression fixtures + causal POI fix.**
- [ ] **Pass 5 — file-based distributed runner** (multiple nodes → shared envelope directory → coordinator).
- [ ] **Pass 6 — real, sales-normalised popularity calibration data** (see `docs/GEOGRAPHY.md`).
- [ ] **Pass 7 — networking layer**, only after Pass 5 aggregation is proven deterministic.
