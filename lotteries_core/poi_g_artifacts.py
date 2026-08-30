"""Portable POI-G candidate, selection, and settlement artifacts.

The extensive POI-G shortlist is a search-space artifact.  It is never charged as if every
candidate were purchased.  A separate fixed-budget inference envelope identifies the only tickets
eligible for ROI settlement.  This separation lets a dataset host expose the whole ranked subset
while an evaluator compares equal-budget portfolios without inventing retrospective evidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .envelope import InferenceEnvelope, data_sha256
from .poi_g import PoiGSubset, generate_poi_g_subset
from .popularity import PopularityModel
from .protocol import GameSpec, ProviderResult, Ticket
from .realized_roi import canonical_sha256
from .roi import JackpotModel, default_jackpot_model, expected_roi_per_ticket

ARTIFACT_SCHEMA_VERSION = 1
CANDIDATES_FILE = "poi_g_candidates.csv"
SELECTION_FILE = "poi_g_fixed_budget.csv"
PREDICTION_FILE = "poi_g_prediction.json"
MANIFEST_FILE = "poi_g_candidate_manifest.json"
SETTLEMENT_FILE = "poi_g_settlement.json"


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_numbers(values: tuple[int, ...]) -> str:
    return json.dumps(list(values), separators=(",", ":"))


def _preference_scores(subset: PoiGSubset) -> np.ndarray:
    """Turn distance from target into a conventional higher-is-better provider score."""
    return -np.abs(np.asarray(subset.scores, dtype=float) - float(subset.target_g))


@dataclass(frozen=True)
class PoiGArtifactBundle:
    """In-memory representation of the files published for one draw-keyed POI-G run."""

    candidates: pd.DataFrame
    selection: pd.DataFrame
    prediction: InferenceEnvelope
    manifest: dict[str, Any]

    def write(self, directory: str | Path) -> dict[str, Path]:
        """Write the bundle atomically enough for a fresh dataset staging directory.

        The manifest is written last and hashes every other artifact, so readers never have to
        trust a partially replaced candidate or prediction file.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "candidates": directory / CANDIDATES_FILE,
            "selection": directory / SELECTION_FILE,
            "prediction": directory / PREDICTION_FILE,
            "manifest": directory / MANIFEST_FILE,
        }
        self.candidates.to_csv(paths["candidates"], index=False, lineterminator="\n")
        self.selection.to_csv(paths["selection"], index=False, lineterminator="\n")
        paths["prediction"].write_text(self.prediction.to_json() + "\n", encoding="utf-8")
        manifest = {
            **self.manifest,
            "artifacts": {
                CANDIDATES_FILE: _sha256_path(paths["candidates"]),
                SELECTION_FILE: _sha256_path(paths["selection"]),
                PREDICTION_FILE: _sha256_path(paths["prediction"]),
            },
        }
        paths["manifest"].write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return paths


def build_poi_g_artifacts(
    history: pd.DataFrame,
    spec: GameSpec,
    *,
    draw_key: str,
    subset_size: int,
    budget: int,
    window: int = 26,
    pairing: str = "cross",
    seed: int = 0,
    created_utc: str = "",
    jackpot: JackpotModel | None = None,
    benchmark_version: str = "",
    evidence_kind: str = "prospective",
    repo_dir: str | Path | None = None,
) -> PoiGArtifactBundle:
    """Build a ranked shortlist plus the fixed-budget envelope eligible for ROI.

    ``history`` must contain only rows known before ``draw_key``.  Prospective callers should pass
    an ISO-8601 ``created_utc``.  Deterministic synthetic/replay publishers may leave it empty, but
    must label ``evidence_kind`` accordingly; an empty timestamp is never represented as a sealed
    real-world prediction.
    """
    if history.empty:
        raise ValueError("history must contain at least one known draw")
    if not str(draw_key).strip():
        raise ValueError("draw_key must be non-empty")
    if subset_size < 1:
        raise ValueError("subset_size must be positive")
    if budget < 1:
        raise ValueError("budget must be positive")
    if evidence_kind == "prospective" and not created_utc:
        raise ValueError("prospective artifacts require created_utc")

    subset = generate_poi_g_subset(
        history, spec, subset_size=subset_size, window=window, pairing=pairing
    )
    selection_budget = min(int(budget), subset.size)
    if selection_budget < 1:
        raise ValueError("POI-G produced no candidates")

    jackpot = jackpot or default_jackpot_model(spec)
    popularity = PopularityModel()
    reference = popularity.reference_mean_popularity(spec, seed=seed)
    shares = popularity.absolute_shares(spec, subset.tickets, reference=reference)
    preferences = _preference_scores(subset)
    history_hash = data_sha256(history)
    identity = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "provider": "poi_g_causal",
        "game": asdict(spec),
        "draw_key": str(draw_key),
        "history_sha256": history_hash,
        "history_rows": len(history),
        "subset_size": subset.size,
        "budget": selection_budget,
        "window": int(window),
        "pairing": pairing,
        "seed": int(seed),
        "target_g": subset.target_g,
        "benchmark_version": benchmark_version,
        "evidence_kind": evidence_kind,
        "created_utc": created_utc,
    }
    prediction_id = canonical_sha256(identity)

    rows: list[dict[str, Any]] = []
    for index, (ticket, g_score, crowd_share, preference) in enumerate(
        zip(subset.tickets, subset.scores, shares, preferences), start=1
    ):
        main, auxiliary = ticket
        candidate = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "benchmark_version": benchmark_version,
            "evidence_kind": evidence_kind,
            "prediction_id": prediction_id,
            "draw_key": str(draw_key),
            "game": spec.name,
            "history_sha256": history_hash,
            "history_rows": len(history),
            "created_utc": created_utc,
            "rank": index,
            "main_numbers": _json_numbers(main),
            "auxiliary_numbers": _json_numbers(auxiliary),
            "g_score": int(g_score),
            "target_g": float(subset.target_g),
            "distance_to_target": float(abs(float(g_score) - float(subset.target_g))),
            "preference_score": float(preference),
            "window": int(window),
            "pairing": pairing,
            "subset_size": subset.size,
            "universe_size": subset.universe_size,
            "universe_fraction": subset.universe_fraction,
            "reduction_factor": subset.reduction_factor,
            "selected_for_roi": int(index <= selection_budget),
            "selection_budget": selection_budget,
            "crowd_popularity_share": float(crowd_share),
            "modeled_expected_roi_per_ticket": float(
                expected_roi_per_ticket(spec, jackpot, float(crowd_share))
            ),
        }
        candidate["candidate_sha256"] = canonical_sha256(candidate)
        rows.append(candidate)

    candidates = pd.DataFrame(rows)
    selection = candidates[candidates["selected_for_roi"] == 1].copy()
    selection.insert(0, "selection_rank", range(1, len(selection) + 1))
    selected_tickets = subset.select(selection_budget)
    prediction = InferenceEnvelope.build(
        provider="poi_g_causal",
        game=spec,
        result=ProviderResult(
            tickets=selected_tickets,
            scores=preferences[:selection_budget],
            diagnostics={
                "prediction_id": prediction_id,
                "candidate_subset_size": subset.size,
                "target_g": subset.target_g,
                "universe_size": subset.universe_size,
                "reduction_factor": subset.reduction_factor,
                "evidence_kind": evidence_kind,
            },
        ),
        seed=seed,
        training_data=history,
        config={
            "draw_key": str(draw_key),
            "window": int(window),
            "pairing": pairing,
            "candidate_subset_size": subset.size,
            "selection_budget": selection_budget,
            "benchmark_version": benchmark_version,
            "evidence_kind": evidence_kind,
        },
        created_utc=created_utc,
        repo_dir=repo_dir,
    )
    prediction.validate()

    modeled = subset.modeled_portfolio_roi(spec, selection_budget, jackpot)
    manifest = {
        **identity,
        "prediction_id": prediction_id,
        "candidate_rows": len(candidates),
        "selection_rows": len(selection),
        "universe_size": subset.universe_size,
        "universe_fraction": subset.universe_fraction,
        "reduction_factor": subset.reduction_factor,
        "candidate_rows_sha256": canonical_sha256(
            [row["candidate_sha256"] for row in rows]
        ),
        "selected_candidate_sha256": list(selection["candidate_sha256"]),
        "modeled_roi": modeled,
        "roi_boundary": (
            "ROI applies only to the fixed-budget selection. The extensive candidate subset is "
            "a ranked search space and is never treated as a purchased portfolio."
        ),
    }
    return PoiGArtifactBundle(candidates, selection, prediction, manifest)


def validate_poi_g_artifacts(directory: str | Path) -> dict[str, Any]:
    """Validate file hashes, candidate collection digest, and the inference envelope."""
    directory = Path(directory)
    manifest_path = directory / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest.get("artifacts", {}).items():
        path = directory / name
        if not path.exists() or _sha256_path(path) != expected:
            raise ValueError(f"artifact integrity check failed: {name}")

    candidates = pd.read_csv(directory / CANDIDATES_FILE, keep_default_na=False)
    if len(candidates) != int(manifest["candidate_rows"]):
        raise ValueError("candidate row count does not match manifest")
    hashes = list(candidates["candidate_sha256"].astype(str))
    if canonical_sha256(hashes) != manifest["candidate_rows_sha256"]:
        raise ValueError("candidate collection does not match manifest")

    selection = pd.read_csv(directory / SELECTION_FILE, keep_default_na=False)
    if len(selection) != int(manifest["selection_rows"]):
        raise ValueError("selection row count does not match manifest")
    if list(selection["candidate_sha256"].astype(str)) != list(
        manifest["selected_candidate_sha256"]
    ):
        raise ValueError("fixed-budget selection does not match manifest")

    prediction = InferenceEnvelope.read(directory / PREDICTION_FILE)
    prediction.validate()
    if prediction.data_sha256 != manifest["history_sha256"]:
        raise ValueError("prediction history hash does not match manifest")
    if prediction.diagnostics.get("prediction_id") != manifest["prediction_id"]:
        raise ValueError("prediction identity does not match manifest")
    if prediction.budget != int(manifest["selection_rows"]):
        raise ValueError("prediction budget does not match selected rows")
    return manifest


def _parse_numbers(value: str | list[int] | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(value, str):
        values = [int(item) for item in value.replace(" ", "").split(",") if item]
    else:
        values = [int(item) for item in value]
    return tuple(sorted(set(values)))


def _score(tickets: list[Ticket], actual_main: tuple[int, ...], actual_aux: tuple[int, ...]) -> dict:
    main_set, aux_set = set(actual_main), set(actual_aux)
    tier_counts: dict[str, int] = {}
    best_main = 0
    best_aux = 0
    for main, auxiliary in tickets:
        main_hits = len(set(main) & main_set)
        aux_hits = len(set(auxiliary) & aux_set)
        best_main = max(best_main, main_hits)
        best_aux = max(best_aux, aux_hits)
        key = f"{main_hits}+{aux_hits}"
        tier_counts[key] = tier_counts.get(key, 0) + 1
    return {"best_main": best_main, "best_auxiliary": best_aux, "tier_counts": tier_counts}


def _matched_random(spec: GameSpec, budget: int, prediction_id: str) -> list[Ticket]:
    seed = int.from_bytes(
        hashlib.sha256(f"poi-control:{prediction_id}".encode()).digest()[:8], "big"
    )
    rng = np.random.default_rng(seed)
    seen: set[Ticket] = set()
    tickets: list[Ticket] = []
    while len(tickets) < min(budget, spec.n_tickets()):
        main_values = rng.choice(
            np.arange(1, spec.main_n + 1), spec.main_k, replace=False
        )
        main = tuple(sorted(int(value) for value in main_values))
        auxiliary = (
            tuple(
                sorted(
                    int(value)
                    for value in rng.choice(
                        np.arange(1, spec.star_n + 1), spec.star_k, replace=False
                    )
                )
            )
            if spec.star_k
            else ()
        )
        ticket = (main, auxiliary)
        if ticket not in seen:
            seen.add(ticket)
            tickets.append(ticket)
    return tickets


def settle_poi_g_artifacts(
    directory: str | Path,
    *,
    actual_main: str | list[int] | tuple[int, ...],
    actual_auxiliary: str | list[int] | tuple[int, ...] = (),
    payout_table: dict[str, float] | None = None,
    ticket_price: float | None = None,
    currency: str = "EUR",
    outcome_source: str = "official",
    payout_source: str = "none",
    settled_utc: str | None = None,
) -> dict[str, Any]:
    """Settle the sealed fixed-budget selection, never the extensive candidate subset."""
    directory = Path(directory)
    manifest = validate_poi_g_artifacts(directory)
    prediction = InferenceEnvelope.read(directory / PREDICTION_FILE)
    main = _parse_numbers(actual_main)
    auxiliary = _parse_numbers(actual_auxiliary)
    prediction.game.validate_ticket((main, auxiliary))
    control_tickets = _matched_random(
        prediction.game, prediction.budget, str(manifest["prediction_id"])
    )
    method = _score(prediction.tickets, main, auxiliary)
    control = _score(control_tickets, main, auxiliary)

    payouts = {str(key).replace("_", "+"): float(value) for key, value in (payout_table or {}).items()}
    payout_present = payout_table is not None
    method_prize = (
        float(sum(count * payouts.get(tier, 0.0) for tier, count in method["tier_counts"].items()))
        if payout_present
        else None
    )
    control_prize = (
        float(sum(count * payouts.get(tier, 0.0) for tier, count in control["tier_counts"].items()))
        if payout_present
        else None
    )
    stake = float(ticket_price) * prediction.budget if ticket_price is not None else None
    monetary = payout_present and stake is not None and stake > 0
    settled_utc = settled_utc or datetime.now(timezone.utc).isoformat()
    result: dict[str, Any] = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "prediction_id": manifest["prediction_id"],
        "draw_key": manifest["draw_key"],
        "game": prediction.game.name,
        "provider": prediction.provider,
        "budget": prediction.budget,
        "candidate_subset_size": manifest["candidate_rows"],
        "history_sha256": prediction.data_sha256,
        "prediction_created_utc": prediction.created_utc,
        "settled_utc": settled_utc,
        "actual_main": list(main),
        "actual_auxiliary": list(auxiliary),
        "method": method,
        "control": control,
        "stake": stake,
        "method_prize": method_prize,
        "control_prize": control_prize,
        "realized_roi": (method_prize - stake) / stake if monetary else None,
        "control_realized_roi": (control_prize - stake) / stake if monetary else None,
        "realized_roi_lift": (method_prize - control_prize) / stake if monetary else None,
        "currency": currency,
        "outcome_source": outcome_source,
        "payout_source": payout_source if payout_present else "none",
        "payout_table_present": payout_present,
        "payout_table_sha256": canonical_sha256(payouts) if payout_present else "",
    }
    result["result_sha256"] = canonical_sha256(result)
    (directory / SETTLEMENT_FILE).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
