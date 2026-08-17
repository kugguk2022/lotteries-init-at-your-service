# GARCH Return Benchmark

- Benchmark mode: saved_tail_predictions_with_forward_ticket_scoring
- Holdout draws: 4
- Ticket budget per draw: common top-10 clipped to each step's shared shortlist size
- Candidate pool size: 2048
- Ticket cost: 2.5
- Prize alignment: Per-draw prize tables were aligned to the last available history draws by tail count because prizes_range.json has draw ids but no draw dates.

## Ranking by ROI
- 1. `garchx` | roi=23522.9240 | net_profit=2352292.40 | total_return=2352392.40 | total_cost=100.00 | mean_best_ball_hits=1.5000
- 2. `garchx_alternative_volatility` | roi=23522.9240 | net_profit=2352292.40 | total_return=2352392.40 | total_cost=100.00 | mean_best_ball_hits=1.5000
- 3. `garchx_alternative_volatility_v2` | roi=23522.9240 | net_profit=2352292.40 | total_return=2352392.40 | total_cost=100.00 | mean_best_ball_hits=1.5000
