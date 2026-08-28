# Connect LottoBench Analytics to a GPT

LottoBench exposes a read-only OpenAPI interface that a GPT can use as an Action. This is the
current integration path; it does not use the retired ChatGPT plugin manifest format.

## 1. Install and expose the API

```bash
python -m pip install "lottobench[api]"
LOTTERIES_HISTORY=data/lotteries.db lotto-serve --host 0.0.0.0 --port 8007
```

For local testing, open `http://127.0.0.1:8007/docs`. A GPT Action needs a publicly reachable HTTPS
deployment. Keep it read-only, mount the ledger data read-only, and add authentication at the proxy
or hosting layer before exposing non-public evidence.

## 2. Import the Action

In the GPT editor, add an Action and import:

```text
https://YOUR_PUBLIC_HOST/openapi.json
```

The two analytics operations have stable operation IDs:

- `getRealizedRoiAnalytics` summarizes validated observations by provider and version.
- `getRealizedRoiEvolution` returns cumulative ROI after each settled draw.

Suggested GPT instruction:

> Use LottoBench Analytics only for descriptive analysis of validated prospective evidence. Always
> distinguish modeled expected ROI from realized ROI, report the number of analyzed and excluded
> records, keep provider versions separate, and repeat the API disclaimer. Never describe a lottery
> strategy as improving the mechanical odds of a fair draw or as financial or gambling advice.

## 3. Test prompts

- “Summarize realized ROI for the `euromillions` ledger and state the sample size.”
- “Show how cumulative ROI evolved by provider version.”
- “Compare each provider with its matched random control; do not infer an edge from a short sample.”

The Action reads existing ledger results. It does not fetch PyPI analytics, place wagers, accept
payments, or create/settle ledger records. PyPI distributes the code; the deployed API must have
access to the ledger database or files you want to analyze.
