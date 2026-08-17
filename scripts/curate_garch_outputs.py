from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = REPO_ROOT / "outputs" / "euromillions"
LEGACY_DIR = OUTPUTS_ROOT / "legacy"
TOP_K = 3

PRIMARY_APPROACHES = [
    "garchx",
    "garchx_v2",
    "garchx_alternative_volatility",
    "garchx_alternative_volatility_v2",
]

LEGACY_ONLY_APPROACHES = [
    "garchx_alt_vol_smoke_v5",
    "garchx_glm_v2",
    "garchx_smoke_v5",
    "garchx_smoke_v5_short",
    "garch_glm_diagnostics",
    "garch_glm_diagnostics_v2",
]


@dataclass
class RankedApproach:
    name: str
    source: str
    start_date: str | None
    end_date: str | None
    rows: int | None
    rmse: float
    mae: float
    coverage_80: float | None
    coverage_95: float | None
    coverage_error_sum: float | None
    kept_outside_legacy: bool = False


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_ranked_approach(name: str) -> RankedApproach:
    summary = load_summary(OUTPUTS_ROOT / name / "garchx_summary.json")
    coverage_80 = summary.get("coverage_80")
    coverage_95 = summary.get("coverage_95")
    coverage_error_sum = None
    if coverage_80 is not None and coverage_95 is not None:
        coverage_error_sum = abs(float(coverage_80) - 0.80) + abs(float(coverage_95) - 0.95)

    return RankedApproach(
        name=name,
        source="garchx_summary.json",
        start_date=summary.get("start_date"),
        end_date=summary.get("end_date"),
        rows=int(summary["rows"]) if summary.get("rows") is not None else None,
        rmse=float(summary["rmse"]),
        mae=float(summary["mae"]),
        coverage_80=float(coverage_80) if coverage_80 is not None else None,
        coverage_95=float(coverage_95) if coverage_95 is not None else None,
        coverage_error_sum=float(coverage_error_sum) if coverage_error_sum is not None else None,
    )


def rank_primary_approaches() -> list[RankedApproach]:
    ranked = [build_ranked_approach(name) for name in PRIMARY_APPROACHES]
    ranked.sort(
        key=lambda item: (
            item.rmse,
            item.mae,
            item.coverage_error_sum if item.coverage_error_sum is not None else float("inf"),
            -item.rows if item.rows is not None else 0,
        )
    )
    for index, item in enumerate(ranked):
        item.kept_outside_legacy = index < TOP_K
    return ranked


def move_directory(name: str, destination_root: Path) -> None:
    src = OUTPUTS_ROOT / name
    dst = destination_root / name
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    shutil.move(str(src), str(dst))


def restore_directory(name: str) -> None:
    src = LEGACY_DIR / name
    dst = OUTPUTS_ROOT / name
    if not src.exists():
        return
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    shutil.move(str(src), str(dst))


def main() -> None:
    ranked = rank_primary_approaches()
    keep = {item.name for item in ranked if item.kept_outside_legacy}
    move_to_legacy = sorted((set(PRIMARY_APPROACHES) - keep) | set(LEGACY_ONLY_APPROACHES))

    LEGACY_DIR.mkdir(parents=True, exist_ok=True)
    for name in sorted(keep):
        restore_directory(name)
    for name in move_to_legacy:
        move_directory(name, LEGACY_DIR)

    manifest = {
        "ranking_basis": "rmse asc, mae asc, interval coverage error asc, rows desc",
        "top_k": TOP_K,
        "kept_outside_legacy": sorted(keep),
        "moved_to_legacy": move_to_legacy,
        "ranked_approaches": [asdict(item) for item in ranked],
    }
    (OUTPUTS_ROOT / "top_garch_approaches.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Top GARCH Approaches",
        "",
        f"- Ranking basis: {manifest['ranking_basis']}",
        f"- Kept outside `legacy`: {', '.join(manifest['kept_outside_legacy'])}",
        f"- Moved to `legacy`: {', '.join(move_to_legacy)}",
        "",
        "## Ranked approaches",
    ]
    for index, item in enumerate(ranked, start=1):
        coverage80 = f"{item.coverage_80:.4f}" if item.coverage_80 is not None else "na"
        coverage95 = f"{item.coverage_95:.4f}" if item.coverage_95 is not None else "na"
        lines.append(
            f"- {index}. `{item.name}` | rmse={item.rmse:.4f} | mae={item.mae:.4f} | "
            f"coverage80={coverage80} | coverage95={coverage95}"
        )
    (OUTPUTS_ROOT / "top_garch_approaches.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("Kept outside legacy:")
    for name in sorted(keep):
        print(f"- {name}")
    print("Moved to legacy:")
    for name in move_to_legacy:
        print(f"- {name}")
    print(f"Wrote: {OUTPUTS_ROOT / 'top_garch_approaches.json'}")
    print(f"Wrote: {OUTPUTS_ROOT / 'top_garch_approaches.md'}")


if __name__ == "__main__":
    main()
