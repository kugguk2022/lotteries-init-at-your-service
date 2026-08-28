from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import pandas as pd

ROOT = Path(__file__).resolve().parent
results = pd.read_csv(ROOT / "data" / "benchmark_results.csv")
manifest = json.loads((ROOT / "data" / "benchmark_manifest.json").read_text(encoding="utf-8"))

display = results[
    [
        "provider",
        "pair_coverage",
        "number_coverage",
        "mean_jaccard_diversity",
        "unpopularity_lift",
        "expected_roi_per_ticket",
        "hit_recall",
    ]
].sort_values("pair_coverage", ascending=False)

description = f"""
# LottoBench Community Leaderboard

Forward-only, equal-budget results on deterministic synthetic data.

**Benchmark:** {manifest['benchmark_version']} · **seed:** {manifest['seed']} ·
**snapshot:** `{manifest['dataset_sha256'][:16]}…`

Primary metric: **pair coverage**. Expected ROI is modeled and remains negative. These results are
not evidence that a fair lottery is predictable and are not financial or gambling advice.
"""

with gr.Blocks(title="LottoBench Community Leaderboard") as demo:
    gr.Markdown(description)
    gr.Dataframe(display, interactive=False)

if __name__ == "__main__":
    demo.launch()
