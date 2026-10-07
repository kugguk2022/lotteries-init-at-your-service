# LottoBench Inspector 0.1.0

A small local plugin for inspecting existing EuroMillions and NL Lotto evidence. It packages one reusable skill and four read-only stdio MCP tools. Python 3.10+ is the only runtime requirement; no API key, paid model, database, HTTP server, or Python dependency installation is needed.

| Tool | Result |
| --- | --- |
| `lottobench_providers` | Registry identities, versions, implementation families, optional extras and ablations |
| `lottobench_dataset_status` | History cutoff and age, source class, snapshot hashes and prospective status counts |
| `lottobench_benchmark_results` | Metric-specific leaderboard for one stored game cohort |
| `lottobench_compare` | Difference between two agents under the same game, budget, holdout, seed and metric |

## Use

Open this repository in a compatible local Codex/ChatGPT desktop host, then select **LottoBench Inspector** from the **Kugguk LottoBench** marketplace. The root `.agents/plugins/marketplace.json` makes it available; it does not enable it automatically. `mcp.json` launches Python over stdio. On Windows, use a Python installation exposing `python3`, or configure the host's local command as `python`.

Without a plugin host, run from the repository root:

```bash
python3 plugins/lottobench-inspector/skills/lottobench-inspector/scripts/inspector.py summary
python3 plugins/lottobench-inspector/skills/lottobench-inspector/scripts/inspector.py compare --game nl-lotto --a gingerm --b uniform_random
python3 -B -m unittest discover -s plugins/lottobench-inspector/tests -v
```

Example natural-language prompt: **“Compare GINGERM with uniform_random in NL Lotto and show the dataset cutoff.”**

## Evidence scope

The bundled sources are pinned to commit `1de1ce1b40093c761634ba49a5967f7794b4bc2e`. Every read checks both the Git blob SHA and SHA-256 of all embedded source files. These are content checks, not a signed authenticity guarantee or an independent audit of upstream calculations.

Both profile cohorts use 12 contests and 12 tickets per agent with a uniform-random reference. The EuroMillions history ends on **2026-08-28**; NL Lotto ends on **2026-08-22**. These are stored snapshots, not current operator results. The capture timestamp does not update the underlying history.

ROI alpha is a **modeled percentage-point difference**, not cash profit. There is no validated cost/payout settlement in the inspected profiles; realized ROI is returned as unavailable. The inspector does not reproduce algorithm inference or claim that providers outperform the draw process. It does not expose or write forecasts, train models, refresh histories, settle ledgers, or modify existing schedules or boards.

The small server implements the MCP **2025-06-18** handshake and newline-delimited stdio transport. Newer clients must negotiate that compatible revision. It has no HTTP transport, authentication, registered connector, or hosted deployment. ChatGPT web remote-MCP use needs a separate HTTPS deployment and connector registration.

To refresh the snapshot, retain its old version and collect the listed source files at one verified commit. Update `references/snapshot.json` with exact text, blob hashes, SHA-256 hashes and capture time; run the integrity and cohort checks. Never mix files from different commits. Do not fetch draw outcomes as part of model training/inference; preserve the project's forward-only sequencing.
