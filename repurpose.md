# Repository Repurpose: Distributed Lottery Inference Research

## Status

This document defines the intended purpose, scope, repository skeleton, migration rules, and anti-regression boundaries for the next major direction of `lotteries-init-at-your-service`.

The existing repository is a useful lottery-data and modelling playground. The repurpose should **preserve validated data, benchmarks, and research code**, while changing the product/research framing from isolated ticket prediction toward a **distributed inference and coverage research system**.

This document is the architectural contract for that transition. Existing modules should not be deleted or moved until the relevant migration step is implemented and tested.

---

## 1. New Intended Purpose

Build an open-source research framework that lets independent users or nodes:

1. ingest and validate public lottery draw data;
2. run multiple independent inference, statistical, and search strategies;
3. publish compact, reproducible inference summaries rather than raw private state;
4. aggregate independent model outputs into consensus or diversity-aware candidate sets;
5. measure whether distributed inference improves **coverage, diversity, calibration, or expected hit distribution** compared with single-model and uniform-random baselines;
6. simulate portfolio-style allocation across candidate combinations under explicit user-defined budgets;
7. keep actual ticket purchasing and financial custody outside the repository.

The project is therefore a **distributed inference / combinatorial coverage research platform**, not a pooled betting service and not a promise that lottery draws are predictable.

---

## 2. Explicit Non-Goals and Boundaries

The repository must not become:

- a system that pools or holds user money;
- a service that buys tickets on behalf of users;
- a central coordinator for syndicate purchases;
- a mechanism for automatic wager execution;
- a system that guarantees profit, positive EV, or increased winnings;
- a system that presents historical overfitting as predictive evidence;
- a hidden centralized betting operation disguised as distributed inference.

Users may independently consume generated research outputs. Any real-world participation remains outside the codebase and under the user's own responsibility and jurisdiction.

The main scientific claim we are allowed to investigate is narrower:

> Can multiple independent search/inference processes produce better diversity, coverage, calibration, robustness, or risk distribution than a single heuristic under the same candidate budget?

That question is testable without claiming future lottery draws are predictable.

---

## 3. Core Design Principles

### 3.1 Reproducibility first

Every published inference result must be reproducible from:

- dataset fingerprint;
- algorithm/version identifier;
- parameters;
- seed where relevant;
- timestamp or draw cut-off;
- candidate budget.

### 3.2 Equal-budget comparisons

All algorithm comparisons must use the same candidate/ticket budget unless the experiment explicitly studies budget scaling.

### 3.3 Forward-only validation

No model may train or tune on a future draw that is later scored as holdout evidence.

### 3.4 Diversity is a first-class metric

A distributed system should not merely average identical predictors. It should measure and reward useful disagreement, orthogonality, coverage, and robustness.

### 3.5 No prediction theater

Outputs should separate:

- descriptive statistics;
- candidate-generation heuristics;
- calibrated probabilities where defensible;
- experimental rankings;
- actual holdout evidence.

### 3.6 User-side autonomy

The repository generates data and recommendations. It does not execute wagers, custody funds, or require users to coordinate purchases.

---

## 4. Proposed Repository Skeleton

```text
lotteries-init-at-your-service/
├── README.md
├── repurpose.md
├── pyproject.toml
├── Makefile
│
├── docs/
│   ├── architecture.md
│   ├── research-methodology.md
│   ├── metrics.md
│   ├── threat-model.md
│   ├── legal-boundaries.md
│   └── experiments/
│
├── lotteries/
│   ├── common/
│   │   ├── schema.py
│   │   ├── dataset.py
│   │   ├── validation.py
│   │   └── fingerprints.py
│   ├── euromillions/
│   ├── eurodreams/
│   └── totoloto/
│
├── inference/
│   ├── protocol.py
│   ├── registry.py
│   ├── baselines/
│   │   ├── uniform.py
│   │   └── frequency.py
│   ├── statistical/
│   ├── combinatorial/
│   ├── ml/
│   └── legacy_adapters/
│
├── distributed/
│   ├── node.py
│   ├── manifest.py
│   ├── envelopes.py
│   ├── aggregation.py
│   ├── consensus.py
│   ├── diversity.py
│   └── transport/
│       ├── local.py
│       └── file_exchange.py
│
├── portfolio/
│   ├── candidate_pool.py
│   ├── dedupe.py
│   ├── coverage.py
│   ├── allocation.py
│   └── constraints.py
│
├── evaluation/
│   ├── walk_forward.py
│   ├── same_budget.py
│   ├── calibration.py
│   ├── diversity_metrics.py
│   ├── robustness.py
│   └── reports.py
│
├── simulations/
│   ├── payout_models.py
│   ├── bankroll.py
│   ├── synthetic_draws.py
│   └── scenarios.py
│
├── cli/
│   ├── fetch.py
│   ├── infer.py
│   ├── node.py
│   ├── aggregate.py
│   ├── evaluate.py
│   └── simulate.py
│
├── data/
│   ├── raw/
│   ├── normalized/
│   ├── fixtures/
│   └── manifests/
│
├── runs/
│   └── .gitkeep
│
├── outputs/
│   └── .gitkeep
│
├── legacy/
│   ├── README.md
│   ├── r/
│   └── experiments/
│
└── tests/
    ├── unit/
    ├── integration/
    ├── regression/
    ├── evaluation/
    └── fixtures/
```

---

## 5. What Each New Layer Owns

### `lotteries/`

Canonical lottery-specific data contracts and ingestion adapters.

Responsibilities:

- draw schemas;
- source normalization;
- range validation;
- deduplication;
- deterministic dataset fingerprints.

No inference logic should live here.

### `inference/`

Independent candidate-generating strategies.

Every strategy should implement one common protocol, conceptually:

```python
class InferenceStrategy:
    name: str
    version: str

    def fit(self, history): ...
    def rank(self, candidate_budget: int): ...
    def manifest(self): ...
```

The important output is not simply a list of numbers. It is a reproducible inference package with provenance.

### `distributed/`

Exchange and aggregation of independent inference packages.

Phase 1 should use local/file transport only. Networking can be added later.

A node should be able to emit a signed or hash-addressed manifest such as:

```json
{
  "schema": "inference-envelope/v1",
  "lottery": "euromillions",
  "dataset_fingerprint": "sha256:...",
  "strategy": "branch-classic",
  "strategy_version": "...",
  "candidate_budget": 25,
  "seed": 42,
  "candidates": [],
  "metrics": {},
  "created_at": "..."
}
```

### `portfolio/`

Combines candidate sets without pretending that consensus equals truth.

Functions should include:

- duplicate suppression;
- overlap penalties;
- combination-space coverage;
- diversity-aware selection;
- configurable allocation under a candidate-count or hypothetical budget constraint.

### `evaluation/`

This becomes the most important layer of the project.

Every proposed improvement should be benchmarked against:

1. uniform random;
2. current simple frequency baseline;
3. strongest validated existing single strategy;
4. distributed aggregate using the same candidate budget.

### `simulations/`

Financial and payout calculations stay simulation-only.

No purchase integration belongs in this directory or elsewhere in the repository.

---

## 6. Existing Code Migration Map

The current code should be treated as research assets, not discarded.

### Keep and promote

Existing functionality equivalent to:

- draw fetchers;
- schema validation;
- normalization;
- walk-forward testing;
- same-budget benchmarks;
- reusable baseline inference;

should migrate toward `lotteries/`, `inference/`, and `evaluation/`.

### Adapt behind inference adapters

Existing EuroMillions experimental modules such as arithmetic branches, diagnostics, GARCH-derived experiments, HMM variants, Grok/transformer experiments, and Sobol/combinadic generation should initially be wrapped behind `inference/legacy_adapters/`.

Do not rewrite them before we can reproduce their current benchmark outputs.

### Quarantine as legacy

Deprecated R scripts, one-off notebooks/scripts, historical scratch experiments, and obsolete generated artifacts should move only after provenance is captured into `legacy/`.

### Outputs

Generated experiment outputs should not define architecture. Where practical, large or regenerable artifacts should be ignored by Git and regenerated from manifests.

---

## 7. Distributed Inference Model

The first distributed version should be deliberately simple.

### Node

A node is an independent inference runner. It may use:

- a different model family;
- different hyperparameters;
- different seeds;
- different feature families;
- different combination-space search methods.

### Envelope

A node exports an immutable inference envelope containing provenance and candidate rankings.

### Aggregator

The aggregator accepts multiple valid envelopes that reference the same lottery and compatible dataset cut-off.

It then builds an aggregate candidate pool using configurable policies:

- rank voting;
- reciprocal-rank fusion;
- diversity-weighted fusion;
- coverage maximization;
- model-family quotas;
- disagreement-aware selection.

### No central prediction authority

The aggregator must preserve enough provenance to explain why each candidate entered the final pool.

---

## 8. Success Metrics

The project should not use raw "wins" as its only metric.

Minimum metrics:

### Predictive / holdout

- main-number recall@K;
- secondary-number/star recall@K;
- exact-tier hit counts;
- Brier/log score where real probabilities exist;
- rank of realized numbers/candidates.

### Coverage

- unique combinations represented;
- pair coverage;
- triple coverage;
- number-frequency coverage;
- combination-space dispersion.

### Diversity

- inter-model candidate overlap;
- Jaccard similarity;
- rank correlation;
- model-family concentration;
- entropy of aggregate selections.

### Robustness

- performance across rolling holdout windows;
- degradation under changed seeds;
- degradation under missing nodes;
- sensitivity to one dominant model.

### Financial simulation

- simulated stake;
- simulated payout distribution;
- simulated return distribution;
- drawdown;
- tail risk;
- comparison against equal-budget random portfolios.

Financial metrics must always be labeled as historical or simulated, never guaranteed future returns.

---

## 9. Anti-Regression Gates

A repurpose is only useful if we avoid destroying what already works.

Before moving an existing strategy, capture a regression fixture containing:

- dataset fingerprint;
- command;
- parameters;
- seed;
- candidate budget;
- summary metrics;
- expected artifact schema.

Minimum CI gates after migration begins:

```text
lint
unit tests
schema tests
fixture/data validation
legacy strategy reproduction smoke tests
same-budget evaluation tests
no-lookahead / chronological split tests
inference-envelope schema tests
aggregation determinism tests
```

Any PR changing inference or aggregation logic should show the benchmark delta against the current baseline.

---

## 10. Implementation Phases

### Phase 0 — Contract and reproducibility

- adopt this `repurpose.md`;
- fix repository metadata and README language;
- inventory existing models and commands;
- identify generated artifacts that should leave Git;
- capture reproducible benchmark fixtures for the strongest current strategies.

**Exit gate:** current documented benchmarks can be reproduced from a clean environment.

### Phase 1 — Common inference protocol

- create `InferenceStrategy` protocol;
- wrap uniform and frequency baselines;
- wrap `branch_classic` or the best reproducible current strategy;
- emit inference envelopes;
- add envelope schema tests.

**Exit gate:** three heterogeneous strategies emit interchangeable envelopes.

### Phase 2 — Local distributed aggregation

- accept multiple envelopes;
- validate provenance compatibility;
- implement rank fusion;
- implement diversity-aware fusion;
- implement candidate deduplication and coverage reports.

**Exit gate:** aggregate can be benchmarked under exactly the same total candidate budget as every single-model baseline.

### Phase 3 — Evaluation framework

- generalized walk-forward runner;
- larger holdout windows;
- uncertainty intervals;
- diversity and robustness metrics;
- machine-readable experiment manifests.

**Exit gate:** one command produces an apples-to-apples report for random, single-model, and distributed strategies.

### Phase 4 — Multi-node exchange

Start with safe, inspectable mechanisms:

- directories;
- JSONL bundles;
- signed/hash-addressed files;
- optional Git-based exchange.

Only after that should we consider network transports.

**Exit gate:** independently produced envelopes can be exchanged without sharing execution state or funds.

### Phase 5 — User-facing research tooling

Potential interfaces:

- CLI summary;
- static HTML experiment report;
- local dashboard;
- API for candidate/metric consumption.

No wagering execution API.

---

## 11. Immediate File-Level Migration Proposal

Do not execute all of these moves in one PR.

First migration sequence:

```text
1. add repurpose.md
2. update README.md scope and disclaimers
3. add docs/research-methodology.md
4. add inference/protocol.py
5. add inference/baselines/uniform.py
6. adapt existing euromillions/infer.py as frequency baseline
7. add distributed/envelopes.py
8. add evaluation/same_budget.py
9. wrap current branch_classic implementation
10. benchmark single vs aggregate
```

Only after the baseline equivalence gates pass should we reorganize the older package layout physically.

---

## 12. Proposed MVP

The smallest meaningful proof of the new idea is not a networked betting application.

It is this experiment:

```text
Historical draw dataset
        |
        +--> uniform baseline -------+
        +--> frequency model --------+
        +--> branch-classic ---------+--> envelope validator
        +--> Sobol/coverage model ----+         |
                                                 v
                                         diversity-aware
                                           aggregator
                                                 |
                                                 v
                                      same-budget evaluator
                                                 |
                      +--------------------------+-------------------+
                      |                          |                   |
                 holdout recall              coverage           robustness
```

Compare, over a substantial forward-only holdout:

- each individual strategy with `N` candidates;
- an aggregate constrained to the same `N` candidates;
- uniform random with the same `N` candidates.

If the aggregate does not beat or robustly complement the baselines, keep the result. A negative result is still useful research.

---

## 13. Definition of Done for the Repurpose

The repository can be considered successfully repurposed when:

- the README describes distributed inference/coverage research rather than implied lottery prediction;
- existing strongest experiments remain reproducible;
- all active inference implementations expose a common interface;
- every inference run produces a provenance manifest;
- independent manifests can be aggregated deterministically;
- distributed aggregation is evaluated under equal budgets;
- reports include random baselines and forward-only holdouts;
- simulated financial outcomes are clearly labeled;
- there is no code for pooled funds or automatic ticket purchasing;
- CI prevents lookahead, schema drift, and silent benchmark regressions.

---

## 14. Working Project Statement

Suggested concise description:

> **Open-source distributed inference and combinatorial-coverage research for lottery datasets. Multiple independent models produce reproducible candidate rankings that can be diversity-aware aggregated and evaluated against equal-budget random and single-model baselines. Research only; no pooled funds or wager execution.**

This statement should guide future README, package metadata, issue labels, and implementation decisions.
