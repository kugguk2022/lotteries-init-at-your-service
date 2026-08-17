# GARCH Return Benchmark

- Benchmark mode: saved_tail_predictions_with_forward_ticket_scoring
- Holdout draws: 4
- Ticket budget per draw: common top-10 clipped to each step's shared shortlist size
- Candidate pool size: 66
- Ticket cost: 2.5
- Prize alignment: Per-draw prize tables were aligned to the last available history draws by tail count because prizes_range.json has draw ids but no draw dates.

## Ranking by ROI
- 1. `garchx` | roi=-1.0000 | net_profit=-10.00 | total_return=0.00 | total_cost=10.00 | mean_best_ball_hits=0.2500
- 2. `garchx_alternative_volatility` | roi=-1.0000 | net_profit=-10.00 | total_return=0.00 | total_cost=10.00 | mean_best_ball_hits=0.2500
- 3. `garchx_alternative_volatility_v2` | roi=-1.0000 | net_profit=-10.00 | total_return=0.00 | total_cost=10.00 | mean_best_ball_hits=0.2500
