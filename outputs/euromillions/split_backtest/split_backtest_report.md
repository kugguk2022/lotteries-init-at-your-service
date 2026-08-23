# EuroMillions Split Backtest

## Cleaning
- Raw rows: 1943.
- Cleaned rows: 1943.
- Duplicate draw dates removed: 0.
- Cleaned history range: 2004-02-13 to 2026-05-05.

## Rule Change Verification
- Before 11 May 2011: Lucky Stars maxed at 9.
- 11 May 2011 to 26 September 2016: Lucky Stars maxed at 11.
- From 27 September 2016: Lucky Stars maxed at 12.
- First observed star 12 in the cleaned local history: 2016-09-27.
- Main balls stayed at 1-50 throughout.

## Test Window (2024-2026)
- agent_logistic | full_history: recall@5=0.1086, any-hit=0.4490, exact5=0.0000
- agent_logistic | post_2016: recall@5=0.0971, any-hit=0.4000, exact5=0.0000
- frequency_top5 | full_history: recall@5=0.1053, any-hit=0.4367, exact5=0.0000
- frequency_top5 | post_2016: recall@5=0.0971, any-hit=0.4082, exact5=0.0000

## Inference Snapshot
- agent_logistic | full_history: next main pick [7, 14, 29, 35, 44] using history through 2026-05-05
- agent_logistic | post_2016: next main pick [5, 17, 27, 34, 47] using history through 2026-05-05
- frequency_top5 | full_history: next main pick [19, 23, 29, 42, 44] using history through 2026-05-05
- frequency_top5 | post_2016: next main pick [17, 20, 21, 29, 42] using history through 2026-05-05

## Takeaway
- Frequency baseline is better on the test split with `full_history` training.
- Logistic agent is better on the test split with `full_history` training.
- Because the 2016 change affected Lucky Stars, not the 1-50 main-ball pool, any improvement here is due to regime stability/recency rather than a changed main-number support.
