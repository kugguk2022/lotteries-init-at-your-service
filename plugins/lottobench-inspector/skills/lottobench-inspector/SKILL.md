---
name: lottobench-inspector
description: Inspect and compare the user's LottoBench EuroMillions and NL Lotto provider registry, stored benchmark leaderboards, dataset provenance, freshness, and prospective evidence. Use for LottoBench plugin requests, lottery benchmark comparisons, or questions about which strategies have measured evidence. Execute the bundled inspector; distinguish modeled ROI from settled returns.
---

# LottoBench Inspector

Run the bundled Python 3 inspector using the absolute skill directory supplied by the runtime:

```bash
python3 <skill-dir>/scripts/inspector.py summary
python3 <skill-dir>/scripts/inspector.py providers
python3 <skill-dir>/scripts/inspector.py status --game nl-lotto
python3 <skill-dir>/scripts/inspector.py leaderboard --game euromillions --metric mean_roi_alpha_pp
python3 <skill-dir>/scripts/inspector.py compare --game nl-lotto --a gingerm --b uniform_random
```

Use the `lottobench_*` MCP tools instead when this skill is installed through the full local plugin and those tools are available. Both interfaces use the same verified snapshot.

Report the source commit, dataset cutoff and age, holdout, budget, seed, and metric unit with comparisons. Show provider names exactly; preserve GINGERM, ablations, and optional entrants. Describe optional runtime availability as unknown: inspecting a registry does not instantiate its algorithms.

Treat `mean_roi_alpha_pp` as a **modeled** difference in percentage points from the equal-budget uniform baseline. Treat `mean_expected_roi_pct` as a model output, not actual cash return. Do not infer realized ROI from pending tickets, hit recall, or backtests. Report unavailable settlement evidence as unknown, rather than as zero payout or a global statement that no one has won.

The bundled snapshot is pinned to a verified GitHub commit. It deliberately does not claim live results. When asked for current evidence, use the connected GitHub plugin to fetch the current branch head of `kugguk2022/lotteries-init-at-your-service`, then retrieve every path listed in `references/snapshot.json` at that same commit. Validate every Git blob SHA and SHA-256 before interpreting the data. Retain the earlier snapshot; create a separately named candidate. Do not automatically fetch draw outcomes, train models, update forecasts, settle ledgers, alter schedules, or modify the user's board. Requests to run inference must preserve training/inference before outcome retrieval and the existing forward-only protocol.

Explain source problems returned by the inspector. Keep synthetic community benchmarks separate from the two observed-history game profiles. A new capture date does not make old underlying history fresh. Do not combine leaderboard rows from different budgets, windows, seeds, data snapshots, or metric definitions.

For local MCP use, launch `scripts/mcp_server.py` over stdio. The server exposes four read-only tools and performs no network or filesystem mutations. Remote ChatGPT connector use requires a separately hosted HTTPS MCP endpoint; do not claim that the local server is a registered or hosted ChatGPT app.
