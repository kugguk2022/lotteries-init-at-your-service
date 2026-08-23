# GARCH Return Benchmark

- Benchmark mode: walk_forward_n_plus_1_with_realized_prize_per_winner
- Evaluation window: 2026-04-28 to 2026-05-05
- Holdout draws: 3
- Ticket budget per draw: common top-10 clipped to each step's shared shortlist size
- Candidate pool size: 1024
- Ticket cost: 2.5
- Prize alignment: Prize tables were fetched by exact draw date from lottery.co.uk and scored on the Prize Per Winner column, then cached locally.

## Ranking by ROI
- 1. `garchx` | roi=-0.9627 | net_profit=-72.20 | total_return=2.80 | total_cost=75.00 | mean_best_ball_hits=1.3333 | any_payout_draw_rate=0.3333
- 2. `garchx_alternative_volatility` | roi=-0.9627 | net_profit=-72.20 | total_return=2.80 | total_cost=75.00 | mean_best_ball_hits=1.3333 | any_payout_draw_rate=0.3333
- 3. `garchx_alternative_volatility_v2` | roi=-0.9627 | net_profit=-72.20 | total_return=2.80 | total_cost=75.00 | mean_best_ball_hits=1.3333 | any_payout_draw_rate=0.3333
