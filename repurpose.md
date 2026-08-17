# Repurpose: from lottery playground to a distributed-inference research framework

This document is the **architectural contract** for the repository's pivot. It is intentionally
conservative: it preserves the existing, already-validated experiments while introducing a common
substrate for the new research question.

## The research question

> Can the owner's co-occurrence level-set method, alone or coordinated with independent inference
> strategies, produce **better preregistered draw outcomes and net user returns** than matched random
> and strongest-single-strategy controls under the same fixed ticket budget?

Draw prediction and a positive user outcome are research objectives, not established claims. The
null hypothesis is that a fair draw is independent of history and the method has no edge. The
repository therefore records portfolios before draws, settles them after official results, and
keeps negative or inconclusive evidence. See
[`docs/OUTCOME_TRACKING.md`](docs/OUTCOME_TRACKING.md) and
[`docs/SCOPE.md`](docs/SCOPE.md).

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
| `likely_set_generator.py` | Format-agnostic co-occurrence level-set generator and provider adapter. |
| `outcome_tracker.py` | Immutable prospective record → official settlement → long-horizon verdict ledger. |
| `popularity.py` | A secondary model of how other players pick numbers for jackpot-sharing analysis. |
| `roi.py` | Expected return-per-ticket under pari-mutuel jackpot sharing, plus the instant-game remaining-prize EV model. Simulation-only. |
| `coverage.py` | Combinatorial coverage and diversity metrics (first-class objectives). |
| `aggregation.py` | Deterministic, diversity-aware, equal-budget aggregation across envelopes — the coordination step. |
| `evaluation.py` | Forward-only, equal-budget benchmark. |
| `providers/` | Co-occurrence level set, frequency, unpopularity, and an optional ML ensemble. |

## Key decisions

1. **Preserve validated experiments before physical reorganization.**
2. **Forward-only, equal-budget evaluation** is the only accepted way to claim "better".
3. **Provenance and reproducibility are mandatory** — every envelope records seed, data hash, git SHA,
   config, and framework version, and reproducibility is *checked*, not assumed.
4. **Official hit tiers, tracked payout, and net return are primary outcomes.** Coverage, diversity,
   and unpopularity are explanatory or selection metrics, not substitutes for wins.
5. **Distributed operation starts as local / file exchange.** Nodes write envelopes to a shared
   directory; a coordinator reads and aggregates them.
6. **Networking is added only after deterministic aggregation over file envelopes is validated.**
7. **Execution remains outside the repository.** No pooled funds, ticket purchasing, custody, or
   wager execution. The repository may track simulated portfolio payout and a pre-draw receipt hash,
   but it does not claim a verified cash win without external operator/payment confirmation.
8. **Repository popularity is irrelevant evidence.** Stars, contributor activity, or self-awarded
   endorsements never count as experimental success.

## Product boundary (non-negotiable)

- ❌ No pooling of user funds.
- ❌ No ticket purchasing or wager execution.
- ❌ No guaranteed-winnings claim and no presentation of an unvalidated edge as established.
- ✅ Explicitly test draw-prediction performance and the possibility of positive net user outcomes.
- ✅ Reproducible forward-only recording, official settlement, matched controls, and evidence-based verdicts.
- ✅ Report hit tiers, payout and net return before secondary coverage/popularity metrics.

## Implementation passes

- [x] **Pass 0 — contract & scope.** This document, `docs/SCOPE.md`, `docs/GEOGRAPHY.md`,
      README scope banner.
- [x] **Pass 1 — protocol & envelopes.** `GameSpec`, `InferenceProvider`, `InferenceEnvelope`.
- [x] **Pass 2 — the conditional ROI lever.** Popularity model, expected-conditional-ROI, coverage,
      diversity-aware equal-budget aggregation, instant-game remaining-prize EV.
- [ ] **Pass 3 — wrap strategies as providers.** Co-occurrence, frequency, unpopularity and ML exist;
      branch-classic, Sobol and uniform adapters remain required.
- [x] **Pass 4 — prospective outcome instrument.** Pre-draw record, matched control, integrity hash,
      official settlement, tier/payout tracking and cumulative verdict CLI.
- [ ] **Pass 4 evidence gate — three-year prospective ledger.** Infrastructure completion is not a
      positive result; the verdict remains unknown until evidence accumulates.
- [ ] **Pass 5 — file-based distributed runner** (multiple nodes → shared envelope directory → coordinator).
- [ ] **Pass 6 — real, sales-normalised popularity calibration data** (see `docs/GEOGRAPHY.md`).
- [ ] **Pass 7 — networking layer**, only after Pass 5 aggregation is proven deterministic.
