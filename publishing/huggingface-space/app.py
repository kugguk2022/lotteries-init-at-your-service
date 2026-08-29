from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATASET_URL = "https://huggingface.co/datasets/kugguk/lottobench-community-benchmark"
SOURCE_URL = "https://github.com/kugguk2022/lotteries-init-at-your-service"

results = pd.read_csv(ROOT / "data" / "benchmark_results.csv")
poi_results = pd.read_csv(ROOT / "data" / "poi_g_subset_results.csv")
manifest = json.loads((ROOT / "data" / "benchmark_manifest.json").read_text(encoding="utf-8"))

PROVIDER_LABELS = {
    "provider": "Provider",
    "pair_coverage": "Pair coverage (primary)",
    "number_coverage": "Number coverage",
    "mean_jaccard_diversity": "Diversity",
    "unpopularity_lift": "Unpopularity lift",
    "expected_roi_per_ticket": "Modelled ROI/ticket",
    "hit_recall": "Hit recall",
}

provider_view = (
    results[list(PROVIDER_LABELS)]
    .sort_values("pair_coverage", ascending=False)
    .round(4)
    .rename(columns=PROVIDER_LABELS)
)

# The shortlist is scored on containment, so the reader needs the random baseline next to it and
# an explicit sense of how little a four-draw holdout can resolve.
poi_view = poi_results.assign(
    shortlist=poi_results["subset_size"].map("{:,} tickets".format),
    universe=poi_results["universe_fraction"].map("{:.2%}".format),
    reduction=poi_results["reduction_factor"].map("{:g}x".format),
    contained=poi_results.apply(
        lambda row: f"{int(row['contained_draws'])} of {int(row['holdout'])}", axis=1
    ),
    random_baseline=poi_results["random_expected_containment_rate"].map("{:.2%}".format),
    null_probability=(
        1.0 - poi_results["random_expected_containment_rate"]
    ).pow(poi_results["holdout"]).map("{:.0%}".format),
    lift=poi_results["containment_lift"].round(2),
    roi=poi_results["modeled_expected_roi_per_selected_ticket"].round(4),
)[
    ["shortlist", "universe", "reduction", "contained", "random_baseline", "lift", "roi", "null_probability"]
].rename(
    columns={
        "shortlist": "Shortlist",
        "universe": "Share of universe",
        "reduction": "Search reduction",
        "contained": "Contained the draw",
        "random_baseline": "Random baseline",
        "lift": "Containment lift (primary)",
        "roi": "Modelled ROI (top 5)",
        "null_probability": "P(0 hits | random)",
    }
)

game = manifest["game"]
best = provider_view.iloc[0]
singles = provider_view[provider_view["Provider"] != "coordinated_aggregation"]
best_single = singles.iloc[0]
# Stated as a comparison only when aggregation actually won; a losing run must not read as a win.
aggregation_leads = best["Provider"] == "coordinated_aggregation"

if aggregation_leads:
    headline = (
        f"Coordinated aggregation leads at **{best['Pair coverage (primary)']:.4f}** against "
        f"**{best_single['Pair coverage (primary)']:.4f}** for the best single provider "
        f"(`{best_single['Provider']}`) — a "
        f"{best['Pair coverage (primary)'] / best_single['Pair coverage (primary)'] - 1:+.1%} difference at "
        "identical budget. That is the claim under test: recombining already-proposed tickets "
        "reaches more of the combinatorial space than any one contributor spending the same "
        f"{manifest['evaluation']['budget']} tickets."
    )
else:
    headline = (
        f"`{best['Provider']}` leads at **{best['Pair coverage (primary)']:.4f}**, ahead of coordinated "
        "aggregation. On this run, recombining proposals did not beat the best single provider at "
        "equal budget — the claim under test is not supported here."
    )

header = f"""
# LottoBench Community Leaderboard

Forward-only, equal-budget results on deterministic synthetic data. Every figure below is read from
a frozen, seeded run — this Space computes nothing at request time.

**Benchmark** `{manifest['benchmark_version']}` · **LottoBench** `{manifest['lottobench_version']}` ·
**seed** `{manifest['seed']}` · **snapshot** `{manifest['dataset_sha256'][:16]}…`

[Dataset]({DATASET_URL}) · [Source]({SOURCE_URL})
"""

provider_note = f"""
## Provider track

Four strategies, {manifest['evaluation']['budget']} tickets per draw, scored on the last
{manifest['evaluation']['holdout']} draws of a {game['main_k']}-of-{game['main_n']} pool plus a
{game['auxiliary_k']}-of-{game['auxiliary_n']} auxiliary pool.

Primary metric is **pair coverage**. {headline}

Hit recall is *not* the primary metric and its spread here is noise. Modelled ROI is negative for
every row and always will be — a fair lottery is negative-sum.
"""

poi_note = """
## POI-G candidate subsets

POI-G is a search-space reducer, not a five-ticket strategy. Containment is scored on the entire
shortlist; modelled ROI is scored only on the five top-ranked tickets, because those are the only
ones anyone would buy.

**The result is null: the shortlist never contained the true draw, at any size.** The last column is
why that settles nothing — a shortlist with exactly random containment would also show zero hits
this often on a four-draw holdout. The honest reading is *no measurable lift, and not enough holdout
to measure one*. It is published rather than dropped because a leaderboard that hides null results
is not a leaderboard.
"""

glossary = """
| Metric | Meaning |
|---|---|
| **Pair coverage** | Fraction of all number pairs covered by at least one ticket in the portfolio. Primary provider metric. |
| **Number coverage** | Fraction of the main number pool touched by the portfolio. |
| **Diversity** | 1 − mean pairwise Jaccard similarity between tickets. 1 means fully disjoint. |
| **Unpopularity lift** | Portfolio expected payout ÷ that of an equally sized uniformly popular portfolio. Above 1 leans unpopular, which raises conditional payout by reducing jackpot sharing. |
| **Modelled ROI/ticket** | Jackpot-tier `E[payout] · P(win) / price − 1`. A comparative diagnostic from a single-tier model, not a payout forecast, and not realised returns. |
| **Hit recall** | Mean fraction of a ticket's numbers appearing in the actual draw. Secondary; unstable at this holdout size. |
| **Containment lift** | Shortlist containment rate ÷ the rate an equally sized random shortlist would achieve. 1.0 is no better than random. |
"""

protocol = f"""
- **Forward-only** — at each holdout step every provider is refitted on strictly earlier rows, so
  there is no leakage path.
- **Equal budget** — {manifest['evaluation']['budget']} tickets per provider per draw; coverage
  comparisons are meaningless otherwise.
- **Frozen game** — `main_n={game['main_n']}, main_k={game['main_k']},
  auxiliary_n={game['auxiliary_n']}, auxiliary_k={game['auxiliary_k']}` →
  {poi_results['universe_size'].iloc[0]:,} legal tickets.
- **Deterministic** — seed `{manifest['seed']}`; the aggregator is a greedy submodular selection,
  identical given identical inputs.
- **Snapshot** — `{manifest['dataset_sha256']}`. Results are comparable only when this digest,
  the provider set, budget, holdout, seed, and scoring code all match.

```bash
python scripts/build_platform_bundles.py
```
"""

disclaimer = """
---

Synthetic software benchmark only. These results are not evidence that a fair lottery is
predictable, are not performance on any operated lottery, and are not financial or gambling advice.
LottoBench places no wagers and does not improve mechanical odds. The safest financial baseline is
not to play. If gambling is causing harm, contact a local support service such as
[BeGambleAware](https://www.begambleaware.org/).
"""

with gr.Blocks(title="LottoBench Community Leaderboard") as demo:
    gr.Markdown(header)
    gr.Markdown(provider_note)
    gr.Dataframe(provider_view, interactive=False, wrap=True)
    gr.Markdown(poi_note)
    gr.Dataframe(poi_view, interactive=False, wrap=True)
    with gr.Accordion("Metric definitions", open=False):
        gr.Markdown(glossary)
    with gr.Accordion("Protocol and reproduction", open=False):
        gr.Markdown(protocol)
    gr.Markdown(disclaimer)

if __name__ == "__main__":
    demo.launch()
