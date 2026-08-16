# Methods and Findings

Every method in the repository, what it actually computes, and how it has scored. Results here are
copied from runs, not from intent. Where a method has been **demoted**, the evidence for the demotion
is given — a documented negative result is worth as much as a positive one and is much easier to lose.

## How "better" is decided

[`lotteries_core/evaluation.py`](../../lotteries_core/evaluation.py) is the only arbiter. Its rules:

- **Forward-only.** At holdout draw `t`, every provider is fit on draws `< t` only, then scored
  against draw `t`. No holdout draw ever informs the state that scores it.
- **Equal budget.** Every provider spends the same number of tickets at every step.
- **Coverage and ROI, not hits.** Hitting a fair draw in any realistic window is astronomically
  unlikely, so realized `hit_recall` is reported but treated as noise.

### The metrics

| Metric | Meaning | Direction |
|---|---|---|
| `pair_coverage` | Fraction of all number *pairs* the portfolio touches | higher |
| `number_coverage` | Fraction of the 50 numbers the portfolio touches | higher |
| `mean_jaccard_diversity` | Mean pairwise dissimilarity of tickets | higher |
| `unpopularity_lift` | Expected payout multiple from avoiding crowded combinations | higher |
| `expected_roi_per_ticket` | Expected return per ticket. **Always negative** — the game is negative-sum. The goal is *less* negative, never positive. | higher (less negative) |
| `hit_recall` | Realized fraction of drawn numbers covered | high variance; ignore small differences |

## The providers

### `frequency` — baseline

Smoothed historical-frequency weighted sampling. Deliberately kept as the reference every other
method must beat. On a fair draw it has **no** predictive edge over uniform sampling; the repository's
own walk-forward test only shows it beating uniform on a deliberately *biased* synthetic dataset.

### `unpopularity` — the honest lever

Oversamples candidate tickets, scores each by expected jackpot payout under the crowd-popularity
model, and keeps the least-crowded ones. It does **not** improve the odds of winning. It improves the
expected payout *conditional* on winning, because fewer people share the jackpot. This is the only
mathematically valid lever in a fair draw game, and it is the current champion on that metric.

### `cooccurrence_level_set` — the owner's method

From draw history it builds pairwise co-occurrence counts `W`, scores every candidate ticket by
`G(ticket) = Σ W` over all pairs inside it, and returns tickets whose `G` is closest to a target. In
`predicted` mode the target is a causal trailing-mean forecast built only from prior draws.

Under the fair-draw null, past co-occurrence cannot change a ticket's mechanical odds. `G` is an
experimental ranking signal under test, not a claim. See [Outcome Tracking](Outcome-Tracking.md).

### `perron_frobenius` — PageRank on the co-occurrence graph

Treats numbers as nodes of the co-occurrence graph, column-normalizes into a transition matrix (with
dangling numbers teleporting uniformly), and damps it exactly as Google's PageRank does:
`M = d·P + (1−d)/n·11ᵀ`. `M` is strictly positive, hence primitive, so Perron–Frobenius guarantees a
unique positive stationary vector `π` with `M π = π`, reached by power iteration at rate `|λ₂| ≤ d`.

Three orientations: `affinity` (prefer high π), `contrarian` (prefer low π), and `uniform` — an
**ablation** that runs the identical sampler with π discarded, so any advantage can be attributed to
the spectral signal or denied it.

**Verdict: demoted, with evidence.** The provider computes its own distance from worthlessness,
`tv_from_uniform = ½Σ|πᵢ − 1/n|`, calibrated against simulated fair histories of identical length via
`null_tv_band`:

```
observed TV, real EuroMillions 2016-2025 (1004 draws) : 0.03024
fair-draw null, 2000 replicates                       : mean 0.03225, sd 0.00351
one-sided p (null ≥ observed)                         : 0.714
observed sits at the 29th percentile of the null
```

The real co-occurrence graph is *flatter* than a typical fair simulation. π spans 0.0148–0.0230
against a uniform 0.0200, and all of that is sampling noise.

There is also a structural reason PageRank cannot express the intuition it is usually reached for. A
**closed regular component receives stationary mass exactly proportional to its size** — i.e. exactly
uniform. So "these numbers always come up together" cannot, on its own, lift a number in this ranking.
This is captured as a regression test (`test_isolated_clique_gets_no_pagerank_boost`).

### `parallax_guard` — replicated residuals, coverage-first portfolio

An externally contributed provider ([`providers/parallax.py`](../../lotteries_core/providers/parallax.py)).
Two ideas, deliberately separated.

**The admission rule.** History is split into two disjoint, interleaved halves. A number or pair is
admitted only when both views show the **same signed deviation** from the exact fair-draw expectation
*and* the weaker view clears a family-wise (Bonferroni) normal threshold. A one-window hot streak
therefore contributes exactly zero. It also restricts to draws under the **current game matrix**
(EuroMillions moved to a 12-star pool on 2016-09-27); older rows are valid records but are not
observations of today's null.

**The portfolio objective.** Candidates are selected as one set by greedy marginal utility over new
main pairs, balanced main and star usage, new star pairs, low crowd popularity, and the admitted
residual — with pair reach weighted to dominate, so no signal can outvote gross duplication.

`mode="ablation"` runs the identical generator and objective with the residual forced to zero.

**Result on real data.** Fitted on 1,972 EuroMillions draws (927 of them under current rules), the
guard admits **zero** residuals — `evidence_nonzero = 0`, `evidence_max_abs = 0.0`. Guarded and
ablation are therefore the *same estimator*, and produce identical portfolios. Independent of the
PageRank null test above, and by a completely different statistic, this is the same verdict: there is
no replicable structure in the draw history.

Its portfolio construction, however, is the strongest in the repository — a single 25-ticket portfolio
reaches `pair_coverage` 0.2041 and `number_coverage` 1.000, with star usage spread across the pool
within one (4–5 uses each), fixing the pinned-star concentration the level-set generator exhibits.
That value is entirely attributable to the optimizer, which is exactly what the ablation was built to
establish.

## Head-to-head results

Forward-only, EuroMillions, budget 25, 40-draw holdout, seed 1234, on `data/euromillions.csv`
(1,972 draws through 2026-08-14). Artifact:
[`outputs/euromillions/competition_benchmark.json`](../../outputs/euromillions/competition_benchmark.json).

| provider | pair_cov | number_cov | unpop_lift | ROI/ticket | hit_recall |
|---|---|---|---|---|---|
| `parallax_guard` | **0.2039** | **1.000** | 1.0925 | −0.7373 | 0.1026 |
| `parallax_guard_ablation` | **0.2039** | **1.000** | 1.0925 | −0.7373 | 0.1026 |
| `frequency` | 0.1850 | 0.933 | 1.0448 | −0.7487 | 0.1052 |
| *aggregated* | 0.1774 | 0.952 | 1.1415 | −0.7255 | 0.1020 |
| `perron_frobenius_affinity` | 0.1610 | **1.000** | 1.1283 | −0.7287 | 0.0982 |
| `perron_frobenius_uniform` *(ablation)* | 0.1600 | **1.000** | 1.1264 | −0.7291 | 0.0948 |
| `perron_frobenius_contrarian` | 0.1598 | **1.000** | 1.1323 | −0.7277 | 0.0970 |
| `unpopularity` | 0.0745 | 0.319 | **1.1890** | **−0.7141** | 0.1028 |

`cooccurrence_level_set`, measured separately on a 5-draw holdout because of its enumeration cost:
0.1690 / 0.840 / 1.0749 / −0.7415.

Reproduce with the command in [Getting Started](Getting-Started.md).

### What to take from this

1. **Both ablations matched their live modes.** `parallax_guard` and `parallax_guard_ablation` agree
   to four decimals on *every* metric, and the three `perron_frobenius` orientations — including the
   sampler-only control — sit within 0.0012 of each other on pair coverage. Two contributed methods,
   two unrelated statistics, one conclusion: **the measured value is portfolio construction, not
   signal.** Neither would have shown this without an ablation, which is why the repository asks for
   one.
2. **`unpopularity` wins the only lever that is real** — best `unpopularity_lift` (1.1890) and best
   expected ROI (−0.7141) — while being *worst* on coverage (0.0745 pair, 0.319 number). It buys
   conditional payout by concentrating on a narrow band of unpopular numbers. That trade is the
   honest one, but it is a trade.
3. **`parallax_guard` wins coverage** by a clear margin (0.2039 vs 0.1850 for the next best) with
   perfect number coverage and balanced star usage.
4. **Aggregation underperformed.** Aggregated pair coverage (0.1774) came in **0.0265 below** the best
   single provider, contrary to the framework's headline claim that coordination should not reduce
   coverage. It does lift `unpopularity_lift` above every entrant except `unpopularity` itself, so it
   is trading coverage for the ROI lever rather than failing outright — but the claim as stated is not
   supported by this run. Open question, not a settled result.
5. **`hit_recall` says nothing.** All eight rows land between 0.095 and 0.106, which is what a fair
   draw looks like. Do not read a ranking into it.

## Forecasting-mode results (EuroMillions lab)

From `euromillions/arithmetic_branch.py`, one-step walk-forward over the last 52 draws:

| Mode | Internal composite fit RMSE | True holdout RMSE |
|---|---|---|
| `classic` | 22.807 | **26.915** |
| `prime-pruned` | **22.773** | 26.972 |

`prime-pruned` looked better on internal fit and lost on true holdout — a clean demonstration of why
only holdout counts. `classic` remains the default; `prime-pruned` is a diagnostic view.

Same-budget shortlist comparison over the last 3 draws (27 tickets each): main-ball recall tied at
0.1333; `diagnostics3_super_likely` took stars 0.6667 vs 0.0000; exact 5+2 accuracy 0.0000 for both.
The window is far too small to conclude anything beyond "not yet materially better".

Artifacts under [`outputs/euromillions/arithmetic_branch/`](../../outputs/euromillions/arithmetic_branch/).

## Methods that cannot work, and are not attempted

GLM, gradient boosting, deep learning, HMMs, and GARCH cannot extract a signal from a certified random
draw, because there is none to extract. Where this repository uses them, they are pointed at **player
behaviour** (crowd popularity, for the jackpot-sharing lever) or at **diagnostics** — never at the
draw. The `ml_ensemble` provider is explicit about this in its own docstring. Full argument in
[`docs/SCOPE_AND_ETHICS.md`](../SCOPE_AND_ETHICS.md).
