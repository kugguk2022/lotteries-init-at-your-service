"""Build the crowd-forecast track evidence: forward-only leaderboard, fitted model, payout report.

Usage::

    python scripts/build_crowd_forecast.py \
        --winners ledger/crowd/euromillions_tier_winners.csv \
        --out outputs/crowd_forecast/euromillions

Writes ``leaderboard.csv``, ``summary.json`` (leaderboard + per-draw losses), ``crowd_model.json``
(the model fitted on all supplied draws, i.e. the commitment for the next draw) and
``payout_report.json`` (all-tier modeled return of calibrated vs static Crowd Escape vs random
tickets). Deterministic for a fixed input file and seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lotteries_core.crowd_forecast import (  # noqa: E402
    fit_crowd_model,
    forward_benchmark,
    load_winner_history,
)
from lotteries_core.protocol import GameSpec  # noqa: E402
from lotteries_core.providers.calibrated_escape import CalibratedEscapeProvider  # noqa: E402
from lotteries_core.providers.uniform import UniformRandomProvider  # noqa: E402
from lotteries_core.providers.unpopularity import UnpopularityProvider  # noqa: E402
from lotteries_core.tier_payout import (  # noqa: E402
    TierEconomics,
    portfolio_report,
    uniform_ticket_baseline,
)


def _ticket_text(ticket) -> str:
    main, star = ticket
    return " ".join(map(str, main)) + (" + " + " ".join(map(str, star)) if star else "")


def _space_artifact(bench: dict, model: dict, report: dict, digest: str) -> dict:
    """Derived, self-contained evidence for the Space's crowd-forecast screen.

    Only derived figures are published (ratios to the uniform-crowd expectation, losses, fitted
    weights, modeled returns). Raw winner counts stay in the repository ledger.
    """
    tiers = bench["tiers"]
    actual = np.array(bench["actual_winners"])
    null = np.array(bench["predicted_winners"]["uniform_crowd"])
    large = null.mean(axis=0) >= 1000.0
    replay = []
    for name, rows in bench["predicted_winners"].items():
        pred = np.array(rows)
        miss = np.sqrt(np.mean(np.log((actual + 0.5) / (pred + 0.5))[:, large] ** 2, axis=1))
        for date, value, loss in zip(
            bench["per_draw"]["draw_date"], miss, bench["per_draw"][f"loss_{name}"]
        ):
            replay.append({"draw_date": date, "forecaster": name,
                           "typical_miss_pct": round(100.0 * float(np.expm1(value)), 3),
                           "weighted_log_error": round(float(loss), 4)})
    deviation = []
    for d, date in enumerate(bench["per_draw"]["draw_date"]):
        for t, tier in enumerate(tiers):
            if large[t]:
                deviation.append({
                    "draw_date": date, "tier": tier,
                    "actual_vs_uniform_pct": round(100.0 * (actual[d, t] / null[d, t] - 1), 2),
                    **{f"{name}_vs_uniform_pct": round(
                        100.0 * (np.array(rows)[d, t] / null[d, t] - 1), 2)
                       for name, rows in bench["predicted_winners"].items()
                       if name != "uniform_crowd"},
                })

    def relative(theta: list[float]) -> list[float]:
        weights = np.exp(np.array(theta))
        return [round(100.0 * float(w / weights.mean() - 1.0), 2) for w in weights]

    return {
        "schema_version": "1.0.0",
        "track": "crowd_forecast",
        "game": bench["game"],
        "evidence_kind": "historical_forward_replay_not_predraw_publication",
        "winners_ledger_sha256": digest,
        "history_draws": bench["history_draws"],
        "history_cutoff": model["history_cutoff"],
        "first_evaluated_draw": bench["first_evaluated_draw"],
        "last_evaluated_draw": bench["last_evaluated_draw"],
        "min_train": bench["min_train"],
        "metric": bench["metric"],
        "leaderboard": bench["leaderboard"],
        "replay": replay,
        "large_tier_deviation": deviation,
        "popularity": {
            "main_vs_average_pct": relative(model["theta_main"]),
            "stars_vs_average_pct": relative(model["theta_star"]),
            "feature_effects": model["feature_effects"],
        },
        "payout": report,
        "data_caveats": [
            "Non-contiguous sample of draws; forward-only evaluation still trains on earlier "
            "dates only.",
            "Winner counts were transcribed from public result pages by an automated reader and "
            "have not been cell-checked against the operator.",
            "Prize-fund money per line (1.10 EUR) is an assumption; absolute ROI depends on it.",
            "The calibrated agent selects the lowest estimated popularity, which biases its "
            "modeled edge upward.",
        ],
        "claims_boundary": bench["claims_boundary"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--winners", default="ledger/crowd/euromillions_tier_winners.csv")
    parser.add_argument("--out", default="outputs/crowd_forecast/euromillions")
    parser.add_argument("--min-train", type=int, default=20)
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--jackpot", type=float, default=None,
                        help="Advertised jackpot for the payout report (default: median of history)")
    parser.add_argument("--entries", type=float, default=None,
                        help="Expected entries for the payout report (default: median of history)")
    parser.add_argument("--space-out", default=None,
                        help="Also write the Space screen artifact (crowd_forecast.json) here, "
                             "e.g. publishing/huggingface-space/data/crowd/euromillions")
    args = parser.parse_args(argv)

    spec = GameSpec.euromillions()
    history = load_winner_history(args.winners, spec)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(Path(args.winners).read_bytes()).hexdigest()

    bench = forward_benchmark(history, min_train=args.min_train)
    bench["winners_sha256"] = digest
    pd.DataFrame(bench["leaderboard"]).to_csv(out / "leaderboard.csv", index=False)
    (out / "summary.json").write_text(json.dumps(bench, indent=2, sort_keys=True) + "\n")

    model = fit_crowd_model(history)
    payload = model.to_dict()
    payload.update(
        history_cutoff=str(history.dates[-1]),
        training_draws=len(history),
        winners_sha256=digest,
        feature_effects=model.diagnostics["feature_effects"],
        least_crowded_main=[int(i) + 1 for i in np.argsort(model.theta_main)[:10]],
        most_crowded_main=[int(i) + 1 for i in np.argsort(-model.theta_main)[:10]],
        least_crowded_stars=[int(i) + 1 for i in np.argsort(model.theta_star)[:4]],
    )
    (out / "crowd_model.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    frame = pd.read_csv(args.winners)
    jackpot = args.jackpot or float(frame["jackpot_eur"].median())
    entries = args.entries or float(frame["entries"].median())
    economics = TierEconomics.euromillions(entries=entries, jackpot=jackpot)
    baseline = uniform_ticket_baseline(model, economics, seed=args.seed)
    slope = next(
        row["calibration_slope"] for row in bench["leaderboard"]
        if row["forecaster"] == "fitted_crowd"
    )
    agents = {
        "uniform_random": UniformRandomProvider(),
        "unpopularity_static_prior": UnpopularityProvider(),
        "calibrated_escape": CalibratedEscapeProvider(model),
    }
    report = {
        "economics": {
            "jackpot_eur": jackpot, "entries": entries,
            "ticket_price": economics.ticket_price, "fund_per_line": economics.fund_per_line,
        },
        "crowd_model_history_cutoff": str(history.dates[-1]),
        "out_of_sample_calibration_slope": slope,
        "null": baseline,
        "agents": {},
    }
    for name, provider in agents.items():
        tickets = provider.propose(spec, args.budget, np.random.default_rng(args.seed)).tickets
        row = portfolio_report(model, economics, tickets, baseline=baseline, seed=args.seed)
        # Haircut: scale the modeled edge by the out-of-sample calibration slope, so an
        # overstated crowd model cannot publish an overstated alpha.
        row["alpha_pp_calibration_haircut"] = row["alpha_pp"] * min(1.0, max(0.0, slope))
        row["tickets_committed"] = [_ticket_text(t) for t in tickets]
        report["agents"][name] = row
    (out / "payout_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    if args.space_out:
        space = Path(args.space_out)
        space.mkdir(parents=True, exist_ok=True)
        (space / "crowd_forecast.json").write_text(
            json.dumps(_space_artifact(bench, payload, report, digest), indent=2, sort_keys=True)
            + "\n"
        )

    print(pd.DataFrame(bench["leaderboard"]).to_string(index=False))
    for name, row in report["agents"].items():
        print(f"{name:28} alpha {row['alpha_pp']:+.2f} pp "
              f"(haircut {row['alpha_pp_calibration_haircut']:+.2f} pp), ROI {row['mean_roi']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
