"""Build redistribution-safe public Space profiles from normalized local histories.

Raw operator histories stay in the local database. The Space receives only provenance metadata,
forward-only aggregate metrics, scored agent submissions, and pending next submissions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lotteries_core import registry, storage
from lotteries_core.aggregation import aggregate
from lotteries_core.dataset import content_digest
from lotteries_core.envelope import InferenceEnvelope
from lotteries_core.evaluation import evaluate_forward
from lotteries_core.protocol import GameSpec
from lotteries_core.roi import JackpotModel, default_jackpot_model
from lottobench.games import GAMES
from scripts.agent_ledgers import build_agent_ledgers

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "publishing" / "huggingface-space" / "data" / "profiles"
SEED = 20260829
BUDGET = 12
HOLDOUT = 12
PROVIDER_NAMES = (
    "uniform_random",
    "frequency",
    "unpopularity",
    "gingerm",
    "spectral_contrarian",
    "parallax",
)

AGENT_LABELS = {
    "uniform_random": "Uniform null",
    "frequency": "Frequency",
    "unpopularity": "Crowd escape",
    "gingerm": "Co-occurrence",
    "spectral_contrarian": "Spectral contrarian",
    "parallax": "Parallax guard",
    "coordinated_aggregation": "Coordinator",
}


def _synthetic_profile_history(rows: int = 48) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    values = []
    for index in range(rows):
        main = sorted(int(value) for value in rng.choice(np.arange(1, 51), 5, replace=False))
        auxiliary = sorted(
            int(value) for value in rng.choice(np.arange(1, 13), 2, replace=False)
        )
        values.append(
            {
                "draw_date": (
                    pd.Timestamp("2025-01-01") + pd.Timedelta(days=index * 7)
                ).date().isoformat(),
                **{f"ball_{position + 1}": value for position, value in enumerate(main)},
                **{
                    f"star_{position + 1}": value
                    for position, value in enumerate(auxiliary)
                },
            }
        )
    return pd.DataFrame(values)


def _profile_history(key: str, database: Path) -> tuple[pd.DataFrame, GameSpec, dict]:
    if key == "synthetic":
        history = _synthetic_profile_history()
        spec = GameSpec(
            "euromillions-synthetic-control",
            main_n=50,
            main_k=5,
            star_n=12,
            star_k=2,
        )
        metadata = {
            "rows": len(history),
            "first_draw": str(history["draw_date"].min()),
            "last_draw": str(history["draw_date"].max()),
            "content_sha256": content_digest(history),
            "fetched_utc": "2026-08-28T00:00:00+00:00",
            "source": "deterministic generator",
        }
        return history, spec, metadata

    definition = GAMES[key]
    history = storage.read_history(database, game=key)
    metadata = storage.read_metadata(database, game=key)
    if history.empty or metadata is None:
        raise ValueError(f"{key} is missing from {database}; run lottobench fetch first")
    return history, definition.spec, metadata


def _source_manifest(key: str, metadata: dict) -> dict:
    if key == "synthetic":
        return {
            "class": "DETERMINISTIC CONTROL",
            "label": "Generated EuroMillions-shaped history",
            "url": None,
            "raw_history_published": False,
        }
    if key == "euromillions":
        return {
            "class": "PUBLIC ARCHIVE",
            "label": "Validated EuroMillions archive fallback",
            "url": "https://www.euro-millions.com/results-history-2026",
            "raw_history_published": False,
            "retrieval_record": metadata["source"],
        }
    return {
        "class": "OPERATOR API",
        "label": "Nederlandse Loterij primary Lotto results",
        "url": GAMES[key].source_url,
        "raw_history_published": False,
        "retrieval_record": metadata["source"],
    }


def _jackpot_model(key: str, spec: GameSpec) -> JackpotModel:
    if key == "synthetic":
        return JackpotModel(
            jackpot=50_000_000.0,
            ticket_price=2.5,
            n_other_tickets=10_000_000.0,
        )
    return default_jackpot_model(spec)


def _leaderboard(contests: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for agent, frame in contests.groupby("agent", sort=False):
        above_null = frame["beats_house"].astype(bool)
        rows.append(
            {
                "agent": agent,
                "agent_label": AGENT_LABELS[agent],
                "contests": len(frame),
                "contests_above_null": int(above_null.sum()),
                "consistency_pct": float(above_null.mean() * 100),
                "mean_roi_alpha_pp": float(frame["roi_alpha_vs_house_pp"].mean()),
                "median_roi_alpha_pp": float(frame["roi_alpha_vs_house_pp"].median()),
                "min_roi_alpha_pp": float(frame["roi_alpha_vs_house_pp"].min()),
                "max_roi_alpha_pp": float(frame["roi_alpha_vs_house_pp"].max()),
                "mean_expected_roi_pct": float(frame["expected_roi_pct"].mean()),
                "mean_pair_coverage_pct": float(frame["pair_coverage_pct"].mean()),
                "mean_hit_recall_pct": float(frame["hit_recall_pct"].mean()),
                "mean_unpopularity_lift_x": float(frame["unpopularity_lift_x"].mean()),
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["mean_roi_alpha_pp", "consistency_pct", "mean_pair_coverage_pct"], ascending=False
    )
    frame.insert(0, "rank", range(1, len(frame) + 1))
    return frame.reset_index(drop=True)


def _prospective_submissions(
    history: pd.DataFrame,
    spec: GameSpec,
    leaderboard: pd.DataFrame,
    snapshot_sha256: str,
    last_draw: str,
    seed: int,
    jackpot: JackpotModel,
) -> pd.DataFrame:
    providers = [registry.create(name) for name in PROVIDER_NAMES]
    envelopes = []
    submissions: dict[str, list[tuple[tuple[int, ...], tuple[int, ...]]]] = {}
    for provider in providers:
        try:
            provider.fit(history, spec)  # type: ignore[call-arg]
        except TypeError:
            provider.fit(history)
        result = provider.propose(spec, BUDGET, np.random.default_rng(seed))
        envelope = InferenceEnvelope.build(
            provider=provider.name,
            game=spec,
            result=result,
            seed=seed,
            training_data=history,
            created_utc="",
        )
        envelopes.append(envelope)
        order = sorted(
            range(len(result.tickets)),
            key=lambda index: (-float(result.scores[index]), result.tickets[index]),
        )
        submissions[provider.name] = [result.tickets[index] for index in order]

    submissions["coordinated_aggregation"] = aggregate(
        envelopes, spec, BUDGET, jackpot=jackpot
    )
    evidence = leaderboard.set_index("agent").to_dict("index")
    rows = []
    for agent, tickets in submissions.items():
        for rank, (main, auxiliary) in enumerate(tickets, start=1):
            commitment_source = (
                f"{snapshot_sha256}|after:{last_draw}|{agent}|{rank}|{main}|{auxiliary}"
            )
            rows.append(
                {
                    "agent": agent,
                    "agent_label": AGENT_LABELS[agent],
                    "rank_within_agent": rank,
                    "main_draw": " ".join(str(value) for value in main),
                    "auxiliary_draw": " ".join(str(value) for value in auxiliary),
                    "backtest_consistency_pct": evidence[agent]["consistency_pct"],
                    "backtest_mean_roi_alpha_pp": evidence[agent]["mean_roi_alpha_pp"],
                    "history_cutoff": last_draw,
                    "score_status": "PENDING",
                    "commitment_sha256": hashlib.sha256(
                        commitment_source.encode("utf-8")
                    ).hexdigest(),
                }
            )
    frame = pd.DataFrame(rows).sort_values(
        ["backtest_mean_roi_alpha_pp", "backtest_consistency_pct", "rank_within_agent"],
        ascending=[False, False, True],
    )
    frame.insert(0, "evidence_rank", range(1, len(frame) + 1))
    return frame.reset_index(drop=True)


def _build_profile(key: str, database: Path, output: Path) -> dict:
    history, spec, metadata = _profile_history(key, database)
    snapshot_sha256 = str(metadata["content_sha256"])
    profile_seed = SEED + {"synthetic": 0, "euromillions": 1000, "nl-lotto": 2000}[key]
    jackpot = _jackpot_model(key, spec)
    providers = [registry.create(name) for name in PROVIDER_NAMES]
    summary = evaluate_forward(
        history,
        spec,
        providers,
        budget=BUDGET,
        holdout=HOLDOUT,
        seed=profile_seed,
        jackpot=jackpot,
        include_steps=True,
    )
    scope = "synthetic_forward_replay" if key == "synthetic" else "observed_history_forward_replay"
    contests, tickets = build_agent_ledgers(
        history,
        summary,
        spec,
        snapshot_sha256,
        evaluation_scope=scope,
    )
    leaderboard = _leaderboard(contests)
    prospective = _prospective_submissions(
        history,
        spec,
        leaderboard,
        snapshot_sha256,
        str(metadata["last_draw"]),
        profile_seed + 100_000,
        jackpot,
    )

    profile_dir = output / key
    profile_dir.mkdir(parents=True, exist_ok=True)
    contests.to_csv(profile_dir / "contests.csv", index=False, lineterminator="\n")
    tickets.to_csv(profile_dir / "tickets.csv", index=False, lineterminator="\n")
    leaderboard.to_csv(profile_dir / "leaderboard.csv", index=False, lineterminator="\n")
    prospective.to_csv(profile_dir / "prospective.csv", index=False, lineterminator="\n")

    display_name = {
        "synthetic": "EuroMillions lab control",
        "euromillions": "EuroMillions",
        "nl-lotto": "Nederlandse Lotto",
    }[key]
    manifest = {
        "schema_version": "2.0.0",
        "profile_key": key,
        "display_name": display_name,
        "data_kind": "deterministic_synthetic" if key == "synthetic" else "observed_public_history",
        "source": _source_manifest(key, metadata),
        "history": {
            "rows": int(metadata["rows"]),
            "first_draw": str(metadata["first_draw"]),
            "last_draw": str(metadata["last_draw"]),
            "snapshot_sha256": snapshot_sha256,
            "retrieved_utc": str(metadata["fetched_utc"]),
        },
        "game": {
            "name": spec.name,
            "main_n": spec.main_n,
            "main_k": spec.main_k,
            "auxiliary_n": spec.star_n,
            "auxiliary_k": spec.star_k,
        },
        "evaluation": {
            "protocol": "forward_only",
            "budget_per_agent": BUDGET,
            "holdout_contests": HOLDOUT,
            "seed": profile_seed,
            "house_reference": "uniform_random",
            "providers": list(PROVIDER_NAMES),
            "coordinator": "coordinated_aggregation",
        },
        "claims_boundary": (
            "ROI alpha is modeled expected-ROI difference from the equal-budget uniform null. "
            "It is not realized profit, improved draw probability, or betting advice."
        ),
    }
    (profile_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


PROFILE_KEYS = ("synthetic", "euromillions", "nl-lotto")


def build(
    database: Path,
    output: Path,
    profile_keys: tuple[str, ...] = PROFILE_KEYS,
) -> None:
    for key in profile_keys:
        _build_profile(key, database, output)

    manifests = [
        json.loads((output / key / "manifest.json").read_text(encoding="utf-8"))
        for key in PROFILE_KEYS
    ]
    index = {
        "schema_version": "2.0.0",
        "profiles": [
            {
                "key": manifest["profile_key"],
                "display_name": manifest["display_name"],
                "data_kind": manifest["data_kind"],
                "last_draw": manifest["history"]["last_draw"],
                "snapshot_sha256": manifest["history"]["snapshot_sha256"],
            }
            for manifest in manifests
        ],
    }
    (output / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True, help="local multi-game LottoBench DB")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=PROFILE_KEYS,
        default=PROFILE_KEYS,
        help="profiles to rebuild; the index is refreshed from every existing manifest",
    )
    args = parser.parse_args()
    build(args.db, args.out, tuple(args.profiles))


if __name__ == "__main__":
    main()
