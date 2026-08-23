"""Version-aware realized-ROI evidence built from prospective LottoBench ledgers.

The export format deliberately excludes tickets, receipt contents, machine identifiers, and user
identity. It is suitable for append-only aggregation in a separate benchmark registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

ROI_SCHEMA_VERSION = 1
LEGACY_VERSION = "legacy-unversioned"
RESULTS = "results.csv"

EXPORT_FIELDS = (
    "schema_version",
    "lottobench_version",
    "provider_name",
    "provider_version",
    "provider_config_sha256",
    "benchmark_id",
    "game",
    "draw_key",
    "currency",
    "n_sets",
    "stake",
    "m_portfolio_prize",
    "m_net_return",
    "realized_roi",
    "c_portfolio_prize",
    "c_net_return",
    "control_realized_roi",
    "realized_roi_lift",
    "outcome_source",
    "payout_table_sha256",
    "purchase_proof_hash_present",
    "record_sha256",
    "settled_utc",
    "result_sha256",
)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _clean(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if hasattr(value, "item"):
        value = value.item()
    return value


def normalize_record(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize current and legacy result rows into the public ROI schema."""
    out = {key: _clean(row.get(key)) for key in EXPORT_FIELDS}
    out["schema_version"] = int(out["schema_version"] or ROI_SCHEMA_VERSION)
    out["lottobench_version"] = str(out["lottobench_version"] or "unknown")
    out["provider_name"] = str(out["provider_name"] or row.get("method") or "unknown")
    out["provider_version"] = str(out["provider_version"] or LEGACY_VERSION)
    out["provider_config_sha256"] = str(out["provider_config_sha256"] or "")
    out["benchmark_id"] = str(out["benchmark_id"] or "")
    out["game"] = str(out["game"] or "unknown")
    out["draw_key"] = str(out["draw_key"] or "")
    out["currency"] = str(out["currency"] or "")
    for key in ("n_sets", "purchase_proof_hash_present"):
        out[key] = int(out[key] or 0)
    for key in (
        "stake",
        "m_portfolio_prize",
        "m_net_return",
        "realized_roi",
        "c_portfolio_prize",
        "c_net_return",
        "control_realized_roi",
        "realized_roi_lift",
    ):
        out[key] = None if out[key] is None else float(out[key])
    if out["realized_roi"] is None and out["stake"] not in (None, 0):
        out["realized_roi"] = float(out["m_net_return"] or 0.0) / float(out["stake"])
    if out["control_realized_roi"] is None and out["stake"] not in (None, 0):
        if out["c_net_return"] is not None:
            out["control_realized_roi"] = float(out["c_net_return"]) / float(out["stake"])
    if out["realized_roi_lift"] is None and out["control_realized_roi"] is not None:
        out["realized_roi_lift"] = out["realized_roi"] - out["control_realized_roi"]
    out["outcome_source"] = str(out["outcome_source"] or "self_reported")
    for key in (
        "payout_table_sha256",
        "record_sha256",
        "settled_utc",
        "result_sha256",
    ):
        out[key] = str(out[key] or "")
    return out


def result_digest(record: dict[str, Any]) -> str:
    unsigned = normalize_record(record)
    unsigned["result_sha256"] = ""
    return canonical_sha256(unsigned)


def validate_record(record: dict[str, Any], *, require_versioned: bool = False) -> dict[str, Any]:
    normalized = normalize_record(record)
    stake = normalized["stake"]
    prize = normalized["m_portfolio_prize"]
    net = normalized["m_net_return"]
    roi = normalized["realized_roi"]
    if stake is None or stake <= 0:
        raise ValueError("realized ROI requires a positive recorded stake")
    if prize is None or prize < 0 or net is None or roi is None:
        raise ValueError("realized ROI requires payout, net return, and ROI")
    if not math.isclose(net, prize - stake, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("net return does not equal payout minus stake")
    if not math.isclose(roi, net / stake, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("realized ROI does not equal net return divided by stake")
    control_payout = normalized["c_portfolio_prize"]
    control_net = normalized["c_net_return"]
    control_roi = normalized["control_realized_roi"]
    if control_payout is not None or control_net is not None or control_roi is not None:
        if (
            control_payout is None
            or control_payout < 0
            or control_net is None
            or control_roi is None
            or normalized["realized_roi_lift"] is None
        ):
            raise ValueError("matched-control ROI fields must be complete")
        if not math.isclose(control_net, control_payout - stake, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("control net return does not equal payout minus stake")
        if not math.isclose(control_roi, control_net / stake, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("control ROI does not equal net return divided by stake")
        if not math.isclose(
            normalized["realized_roi_lift"], roi - control_roi, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError("ROI lift does not equal provider ROI minus control ROI")
    if not normalized["draw_key"] or not normalized["provider_name"]:
        raise ValueError("draw_key and provider_name are required")
    if require_versioned and normalized["provider_version"] == LEGACY_VERSION:
        raise ValueError("legacy row has no provider version")
    digest = normalized["result_sha256"]
    if digest and digest != result_digest(normalized):
        raise ValueError(f"result integrity check failed for {normalized['draw_key']}")
    return normalized


def load_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.is_dir():
        source = source / RESULTS
    if source.suffix.lower() == ".csv":
        if not source.exists():
            return []
        return [normalize_record(row) for row in pd.read_csv(source).to_dict(orient="records")]
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "bundle_sha256" in payload:
        validate_bundle(payload)
    records = payload.get("records", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("ROI bundle must contain a records list")
    return [normalize_record(row) for row in records]


def comparison(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate realized ROI by game/currency/provider/model version and configuration."""
    valid = [validate_record(row) for row in records]
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in valid:
        key = (
            row["game"],
            row["currency"],
            row["provider_name"],
            row["provider_version"],
            row["provider_config_sha256"],
        )
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, rows in groups.items():
        stake = sum(float(row["stake"]) for row in rows)
        payout = sum(float(row["m_portfolio_prize"]) for row in rows)
        net = payout - stake
        control_rows = [row for row in rows if row["c_portfolio_prize"] is not None]
        control_payout = (
            sum(float(row["c_portfolio_prize"]) for row in control_rows)
            if len(control_rows) == len(rows)
            else None
        )
        control_roi = None if control_payout is None else (control_payout - stake) / stake
        realized_roi = net / stake
        summaries.append(
            {
                "game": key[0],
                "currency": key[1],
                "provider_name": key[2],
                "provider_version": key[3],
                "provider_config_sha256": key[4],
                "settled_records": len(rows),
                "settled_draws": len({row["draw_key"] for row in rows}),
                "total_stake": stake,
                "total_payout": payout,
                "net_return": net,
                "realized_roi": realized_roi,
                "control_payout": control_payout,
                "control_realized_roi": control_roi,
                "realized_roi_lift": None if control_roi is None else realized_roi - control_roi,
                "benchmark_cohorts": len({row["benchmark_id"] for row in rows if row["benchmark_id"]}),
                "proof_hash_records": sum(row["purchase_proof_hash_present"] for row in rows),
            }
        )
    return sorted(
        summaries,
        key=lambda row: (
            row["game"],
            row["currency"],
            -row["realized_roi"],
            row["provider_name"],
            row["provider_version"],
        ),
    )


def export_bundle(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    normalized = [validate_record(row) for row in records]
    normalized.sort(
        key=lambda row: (
            row["game"], row["draw_key"], row["provider_name"], row["provider_version"]
        )
    )
    bundle = {"roi_schema_version": ROI_SCHEMA_VERSION, "records": normalized}
    bundle["bundle_sha256"] = canonical_sha256(bundle)
    return bundle


def validate_bundle(bundle: dict[str, Any]) -> None:
    expected = bundle.get("bundle_sha256")
    unsigned = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    if not expected or expected != canonical_sha256(unsigned):
        raise ValueError("ROI bundle integrity check failed")
    for row in bundle.get("records", []):
        validate_record(row)


def _print_comparison(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("[roi] no settled records with cost and payout data")
        return
    print(
        f"{'provider@version':38} {'draws':>7} {'stake':>12} {'payout':>12} "
        f"{'ROI':>10} {'vs random':>11}"
    )
    for row in rows:
        label = f"{row['provider_name']}@{row['provider_version']}"
        print(
            f"{label:38} {row['settled_draws']:>7} {row['total_stake']:>12.2f} "
            f"{row['total_payout']:>12.2f} {row['realized_roi']:>+10.4f} "
            f"{row['realized_roi_lift'] if row['realized_roi_lift'] is not None else float('nan'):>+11.4f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Version-aware realized-ROI benchmark evidence")
    commands = parser.add_subparsers(dest="command", required=True)
    report = commands.add_parser("report", help="compare provider/model versions in a ledger")
    report.add_argument("--ledger", default="ledger/euromillions")
    export = commands.add_parser("export", help="create a deterministic privacy-minimized bundle")
    export.add_argument("--ledger", default="ledger/euromillions")
    export.add_argument("--out", required=True, type=Path)
    validate = commands.add_parser("validate", help="validate an exported ROI bundle")
    validate.add_argument("bundle", type=Path)
    compare = commands.add_parser("compare", help="compare one or more CSV ledgers or bundles")
    compare.add_argument("inputs", nargs="+", type=Path)
    compare.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "validate":
        validate_bundle(json.loads(args.bundle.read_text(encoding="utf-8")))
        print(f"{args.bundle}: valid ROI bundle")
        return 0
    paths = args.inputs if args.command == "compare" else [Path(args.ledger)]
    records = [row for path in paths for row in load_records(path)]
    if args.command == "export":
        bundle = export_bundle(records)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"{args.out}: {len(bundle['records'])} ROI record(s)")
        return 0
    rows = comparison(records)
    if args.command == "compare" and args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        _print_comparison(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
