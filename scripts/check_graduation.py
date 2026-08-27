"""Evaluate the LottoBench alpha-to-stable gates and the provider-evidence gate.

Two milestones are worth interrupting a maintainer for, and neither should be a judgement call:

* **Graduation** -- the published package installs on every supported Python, its public API and
  its ROI ledger schema have survived two or more prereleases without a removal, its benchmark
  output is byte-reproducible, enough settled ROI observations exist to say anything at all, and no
  critical packaging or data-integrity problem is open.
* **Provider evidence** -- a provider has beaten the ``uniform_random`` control on *prospective,
  settled* draws by a margin that a single lucky payout cannot explain. A positive cumulative ROI
  on its own is explicitly not enough: lottery payouts are heavy-tailed, so one jackpot in a thin
  ledger produces a positive number with no predictive content behind it.

Every check here is deterministic and offline. Facts this process cannot observe -- a live PyPI
install matrix, the repository's open issues -- arrive as explicit inputs, and the gate they feed
reports ``unknown`` when they are absent. "Not measured" must never render as "passed".

Usage::

    python scripts/check_graduation.py --out outputs/graduation/status.json
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

from lotteries_core import __version__
from lotteries_core.realized_roi import ROI_SCHEMA_VERSION, load_records, validate_record

ROOT = Path(__file__).resolve().parents[1]

#: The matched control every entrant is measured against. Beating this is the whole claim.
CONTROL_PROVIDER = "uniform_random"

#: Kept in step with the ``requires-python`` floor and the classifiers in ``pyproject.toml``.
SUPPORTED_PYTHONS = ("3.10", "3.11", "3.12", "3.13", "3.14")

RELEASE_TAG = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)(?:(a|b|rc)(\d+))?$")
SCHEMA_VERSION_LINE = re.compile(r"^ROI_SCHEMA_VERSION\s*=\s*(\d+)", re.MULTILINE)
SAFE_KEY = re.compile(r"[^A-Za-z0-9._-]+")

MET = "met"
UNMET = "unmet"
UNKNOWN = "unknown"

#: game, currency, provider name, provider version, provider configuration digest.
ProviderKey = tuple[str, str, str, str, str]


@dataclass(frozen=True)
class Thresholds:
    """Evidence bars, kept explicit so a notification can quote exactly what it cleared."""

    #: Settled, control-matched draws a provider needs before its ROI is worth reading.
    min_draws: int = 30
    #: Draws where provider and control did not return the same amount. Lottery ledgers are mostly
    #: ties -- nobody won anything -- and a sign test only learns from the rest.
    min_decisive_draws: int = 10
    #: One-sided exact sign-test level. 0.01 rather than 0.05: this claim is extraordinary.
    alpha: float = 0.01
    #: Total settled matched draws in the ledger before the graduation gate is satisfied.
    min_settled_draws: int = 30
    #: Prereleases required before a stable release is defensible.
    min_prereleases: int = 2

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_draws": self.min_draws,
            "min_decisive_draws": self.min_decisive_draws,
            "alpha": self.alpha,
            "min_settled_draws": self.min_settled_draws,
            "min_prereleases": self.min_prereleases,
        }


@dataclass
class Gate:
    """One graduation criterion, and why it is or is not satisfied."""

    name: str
    status: str
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def met(self) -> bool:
        return self.status == MET

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class DrawPair:
    """One settled draw where the provider and its matched control both have a net return."""

    draw_key: str
    stake: float
    provider_net: float
    control_net: float

    @property
    def lift(self) -> float:
        return self.provider_net - self.control_net


def sign_test_p_value(successes: int, trials: int) -> float:
    """Exact one-sided binomial tail ``P(X >= successes)`` under ``p = 0.5``.

    Computed as an exact rational and converted once, so a long ledger cannot overflow the
    intermediate integers into a misleadingly small float.
    """
    if trials <= 0:
        return 1.0
    successes = max(0, min(successes, trials))
    tail = sum(math.comb(trials, k) for k in range(successes, trials + 1))
    return float(Fraction(tail, 1 << trials))


def _run_git(args: Sequence[str], cwd: Path) -> str | None:
    try:
        done = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    if done.returncode != 0:
        return None
    return done.stdout


def release_tags(cwd: Path = ROOT) -> list[tuple[tuple[int, int, int, int, int], str]]:
    """Release tags in ascending version order, prereleases sorting before their final release."""
    listing = _run_git(["tag", "--list"], cwd)
    if listing is None:
        return []
    order = {"a": 0, "b": 1, "rc": 2}
    found = []
    for line in listing.splitlines():
        match = RELEASE_TAG.match(line.strip())
        if not match:
            continue
        major, minor, patch, phase, serial = match.groups()
        key = (int(major), int(minor), int(patch), order.get(phase or "", 3), int(serial or 0))
        found.append((key, line.strip()))
    return sorted(found)


def is_prerelease(tag: str) -> bool:
    match = RELEASE_TAG.match(tag)
    return bool(match and match.group(4))


def exported_names(source: str) -> set[str] | None:
    """The ``__all__`` entries of a module, parsed rather than imported.

    Historical revisions are read straight out of git, so they must never be executed.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if not any(isinstance(item, ast.Name) and item.id == "__all__" for item in targets):
            continue
        value = node.value
        if isinstance(value, (ast.List, ast.Tuple)):
            return {
                element.value
                for element in value.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            }
    return None


def read_ledger(ledger: Path) -> list[dict[str, Any]]:
    """Load ROI records, treating an absent ledger as no evidence rather than an error."""
    try:
        return load_records(ledger)
    except FileNotFoundError:
        return []


def paired_draws(
    records: Iterable[dict[str, Any]],
) -> tuple[dict[ProviderKey, list[DrawPair]], list[str]]:
    """Group control-matched settled draws by provider identity, reporting unusable rows.

    A row without a matched control cannot answer "better than random?", so it is dropped rather
    than being silently compared against nothing.
    """
    grouped: dict[ProviderKey, dict[str, list[dict[str, Any]]]] = {}
    problems: list[str] = []
    for raw in records:
        if raw.get("m_portfolio_prize") is None or raw.get("stake") is None:
            # Settled, but with no official payout table or no recorded stake. That is evidence
            # for hit rates and none at all for ROI -- not an integrity failure, so it is skipped
            # rather than reported as a broken record.
            continue
        try:
            row = validate_record(raw)
        except ValueError as error:
            problems.append(f"{raw.get('draw_key', '?')}: {error}")
            continue
        if row["provider_name"] == CONTROL_PROVIDER or row["c_net_return"] is None:
            continue
        key = (
            row["game"],
            row["currency"],
            row["provider_name"],
            row["provider_version"],
            row["provider_config_sha256"],
        )
        grouped.setdefault(key, {}).setdefault(row["draw_key"], []).append(row)

    pairs: dict[ProviderKey, list[DrawPair]] = {}
    for key, draws in grouped.items():
        pairs[key] = [
            DrawPair(
                draw_key=draw_key,
                stake=sum(float(row["stake"]) for row in rows),
                provider_net=sum(float(row["m_net_return"]) for row in rows),
                control_net=sum(float(row["c_net_return"]) for row in rows),
            )
            for draw_key, rows in sorted(draws.items())
        ]
    return pairs, problems


def evaluate_provider(
    key: Sequence[str], draws: Sequence[DrawPair], thresholds: Thresholds
) -> dict[str, Any]:
    """Score one provider identity against the matched control on its settled draws."""
    lifts = [pair.lift for pair in draws]
    total_stake = sum(pair.stake for pair in draws)
    net_lift = sum(lifts)
    decisive = [value for value in lifts if value != 0.0]
    wins = sum(1 for value in decisive if value > 0.0)
    p_value = sign_test_p_value(wins, len(decisive))

    # Drop the single best draw. A method whose entire advantage is one payout has demonstrated a
    # lucky ticket, not a repeatable edge, and that is the exact failure mode this gate exists for.
    net_lift_excluding_best = net_lift - max(lifts) if lifts else 0.0

    reasons = []
    if len(draws) < thresholds.min_draws:
        reasons.append(f"{len(draws)} settled matched draws < {thresholds.min_draws}")
    if len(decisive) < thresholds.min_decisive_draws:
        reasons.append(f"{len(decisive)} decisive draws < {thresholds.min_decisive_draws}")
    if net_lift <= 0.0:
        reasons.append("cumulative net lift over the control is not positive")
    if net_lift_excluding_best <= 0.0:
        reasons.append("cumulative net lift collapses without its single best draw")
    if p_value > thresholds.alpha:
        reasons.append(f"sign-test p {p_value:.4g} > {thresholds.alpha}")

    return {
        "game": key[0],
        "currency": key[1],
        "provider_name": key[2],
        "provider_version": key[3],
        "provider_config_sha256": key[4],
        "settled_draws": len(draws),
        "decisive_draws": len(decisive),
        "draws_won": wins,
        "total_stake": total_stake,
        "net_lift": net_lift,
        "net_lift_excluding_best_draw": net_lift_excluding_best,
        "roi_lift": (net_lift / total_stake) if total_stake else None,
        "sign_test_p_value": p_value,
        "qualifies": not reasons,
        "blocking_reasons": reasons,
    }


def gate_pypi_install(report: dict[str, Any] | None, pythons: Sequence[str]) -> Gate:
    if not report:
        return Gate(
            "pypi_install",
            UNKNOWN,
            "No PyPI install matrix result was supplied.",
            {"required_pythons": list(pythons)},
        )
    results = report.get("results", report)
    missing = [version for version in pythons if version not in results]
    failed = [version for version in pythons if results.get(version) is False]
    evidence = {"required_pythons": list(pythons), "results": results}
    if missing:
        return Gate("pypi_install", UNKNOWN, f"Not attempted on {', '.join(missing)}.", evidence)
    if failed:
        detail = f"Install from PyPI failed on {', '.join(failed)}."
        return Gate("pypi_install", UNMET, detail, evidence)
    return Gate("pypi_install", MET, f"Installed from PyPI on {', '.join(pythons)}.", evidence)


def gate_release_history(tags: Sequence[tuple[Any, str]], thresholds: Thresholds) -> Gate:
    names = [name for _, name in tags]
    prereleases = [name for name in names if is_prerelease(name)]
    evidence = {"tags": names, "prereleases": prereleases}
    if not names:
        detail = "No release tags are visible in this clone."
        return Gate("release_history", UNKNOWN, detail, evidence)
    if len(prereleases) < thresholds.min_prereleases:
        detail = (
            f"{len(prereleases)} alpha/beta release(s), {thresholds.min_prereleases} required."
        )
        return Gate("release_history", UNMET, detail, evidence)
    detail = f"{len(prereleases)} alpha/beta releases: {', '.join(prereleases)}."
    return Gate("release_history", MET, detail, evidence)


def gate_public_api(tags: Sequence[tuple[Any, str]], cwd: Path = ROOT) -> Gate:
    path = "lotteries_core/__init__.py"
    current = exported_names((cwd / "lotteries_core" / "__init__.py").read_text(encoding="utf-8"))
    if current is None:
        return Gate("public_api_stable", UNKNOWN, f"Could not parse __all__ from {path}.")
    if not tags:
        detail = "No release tags to compare the public surface against."
        return Gate("public_api_stable", UNKNOWN, detail, {"current": sorted(current)})

    removed: dict[str, list[str]] = {}
    compared = []
    for _, tag in tags:
        source = _run_git(["show", f"{tag}:{path}"], cwd)
        if source is None:
            continue
        names = exported_names(source)
        if names is None:
            continue
        compared.append(tag)
        gone = sorted(names - current)
        if gone:
            removed[tag] = gone

    evidence = {"current": sorted(current), "compared_tags": compared, "removed": removed}
    if not compared:
        detail = "No tagged revision exposed a readable __all__."
        return Gate("public_api_stable", UNKNOWN, detail, evidence)
    if removed:
        summary = "; ".join(f"{tag}: {', '.join(names)}" for tag, names in removed.items())
        return Gate("public_api_stable", UNMET, f"Public names removed since {summary}.", evidence)
    detail = f"No public name removed across {len(compared)} tagged release(s)."
    return Gate("public_api_stable", MET, detail, evidence)


def gate_ledger_schema(tags: Sequence[tuple[Any, str]], cwd: Path = ROOT) -> Gate:
    path = "lotteries_core/realized_roi.py"
    evidence: dict[str, Any] = {"current": ROI_SCHEMA_VERSION, "history": {}}
    if not tags:
        detail = "No release tags to compare the ROI ledger schema against."
        return Gate("ledger_schema_stable", UNKNOWN, detail, evidence)

    changed = []
    for _, tag in tags:
        source = _run_git(["show", f"{tag}:{path}"], cwd)
        if source is None:
            continue
        match = SCHEMA_VERSION_LINE.search(source)
        if not match:
            continue
        version = int(match.group(1))
        evidence["history"][tag] = version
        if version != ROI_SCHEMA_VERSION:
            changed.append(f"{tag}={version}")

    if not evidence["history"]:
        detail = "No tagged revision declared ROI_SCHEMA_VERSION."
        return Gate("ledger_schema_stable", UNKNOWN, detail, evidence)
    if changed:
        detail = f"ROI schema is v{ROI_SCHEMA_VERSION} but was {', '.join(changed)}."
        return Gate("ledger_schema_stable", UNMET, detail, evidence)
    detail = (
        f"ROI schema held at v{ROI_SCHEMA_VERSION} across "
        f"{len(evidence['history'])} tagged release(s)."
    )
    return Gate("ledger_schema_stable", MET, detail, evidence)


def _synthetic_history(path: Path, rows: int = 120) -> None:
    """A fixed-seed euromillions-shaped history, so the gate runs without the untracked CSV."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(2026)
    records = []
    for index in range(rows):
        main = sorted(int(value) for value in rng.choice(np.arange(1, 51), 5, replace=False))
        stars = sorted(int(value) for value in rng.choice(np.arange(1, 13), 2, replace=False))
        records.append(
            {
                "draw_date": (
                    pd.Timestamp("2026-01-01") + pd.Timedelta(days=index * 4)
                ).date().isoformat(),
                **{f"ball_{position + 1}": value for position, value in enumerate(main)},
                **{f"star_{position + 1}": value for position, value in enumerate(stars)},
            }
        )
    pd.DataFrame(records).to_csv(path, index=False)


def gate_benchmark_reproducible(
    history: Path | None, game: str, budget: int, holdout: int, rows: int = 120
) -> Gate:
    """Run the shipped benchmark CLI twice and require byte-identical output.

    Deliberately a small replay: the most recent ``rows`` draws over a short holdout. Determinism
    is a property of the pipeline rather than of history length, and the full canonical history
    costs over ten minutes per run once the ``ml`` and ``transformer`` extras are installed -- long
    enough that the check would be skipped in practice, which is worse than a smaller one that
    actually runs. ``--all-providers`` is kept, because the optional heavy providers are exactly
    the ones carrying enough internal randomness to be worth pinning down.
    """
    import pandas as pd

    with tempfile.TemporaryDirectory(prefix="lottobench-repro-") as directory:
        root = Path(directory)
        source = root / "history.csv"
        if history is not None and history.exists():
            frame = pd.read_csv(history)
            frame.tail(rows).to_csv(source, index=False)
            provenance = f"{history} (most recent {min(rows, len(frame))} of {len(frame)} draws)"
        else:
            _synthetic_history(source, rows)
            provenance = "deterministic synthetic fixture (canonical CSV not present)"

        digests = []
        for run in (1, 2):
            out = root / f"summary-{run}.json"
            done = subprocess.run(
                [
                    sys.executable, "-m", "lotteries_core.benchmark",
                    "--history", str(source),
                    "--game", game,
                    "--budget", str(budget),
                    "--holdout", str(holdout),
                    "--all-providers",
                    "--out", str(out),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            if done.returncode != 0 or not out.exists():
                tail = (done.stderr or "").strip().splitlines()[-1:] or ["no stderr"]
                detail = f"Benchmark run {run} did not complete: {tail[0]}"
                return Gate("benchmark_reproducible", UNKNOWN, detail, {"history": provenance})
            digests.append(hashlib.sha256(out.read_bytes()).hexdigest())

    evidence = {"history": provenance, "digests": digests}
    if digests[0] != digests[1]:
        detail = "Two identical benchmark invocations produced different output."
        return Gate("benchmark_reproducible", UNMET, detail, evidence)
    detail = f"Repeated benchmark output is byte-identical (sha256 {digests[0][:12]})."
    return Gate("benchmark_reproducible", MET, detail, evidence)


def gate_settled_observations(
    pairs: dict[ProviderKey, list[DrawPair]], thresholds: Thresholds
) -> Gate:
    total = len({pair.draw_key for draws in pairs.values() for pair in draws})
    evidence = {"settled_matched_draws": total, "required": thresholds.min_settled_draws}
    if total < thresholds.min_settled_draws:
        detail = (
            f"{total} settled control-matched draw(s), {thresholds.min_settled_draws} required."
        )
        return Gate("settled_roi_observations", UNMET, detail, evidence)
    detail = f"{total} settled control-matched draws recorded."
    return Gate("settled_roi_observations", MET, detail, evidence)


def gate_no_critical_issues(open_count: int | None, ledger_problems: Sequence[str]) -> Gate:
    evidence = {"open_critical_issues": open_count, "ledger_problems": list(ledger_problems)}
    if ledger_problems:
        detail = f"{len(ledger_problems)} ROI ledger record(s) fail integrity validation."
        return Gate("no_critical_issues", UNMET, detail, evidence)
    if open_count is None:
        detail = "Open critical-issue count was not supplied."
        return Gate("no_critical_issues", UNKNOWN, detail, evidence)
    if open_count > 0:
        return Gate("no_critical_issues", UNMET, f"{open_count} critical issue(s) open.", evidence)
    detail = "No critical issue open and every ROI ledger record validates."
    return Gate("no_critical_issues", MET, detail, evidence)


def notification_key(text: str) -> str:
    return SAFE_KEY.sub("-", text).strip("-")


def _gate_table(gates: Sequence[Gate]) -> str:
    marks = {MET: "pass", UNMET: "fail", UNKNOWN: "unknown"}
    lines = ["| Gate | Status | Detail |", "| --- | --- | --- |"]
    for gate in gates:
        lines.append(f"| `{gate.name}` | {marks[gate.status]} | {gate.detail} |")
    return "\n".join(lines)


def _evidence_table(entry: dict[str, Any]) -> str:
    roi_lift = entry["roi_lift"]
    roi_cell = "n/a" if roi_lift is None else f"{roi_lift:+.4f}"
    return "\n".join(
        [
            "| Measure | Value |",
            "| --- | --- |",
            f"| Settled control-matched draws | {entry['settled_draws']} |",
            f"| Decisive draws (not a tie) | {entry['decisive_draws']} |",
            f"| Draws won against the control | {entry['draws_won']} |",
            f"| Realized ROI lift | {roi_cell} |",
            f"| Net lift excluding its best draw | {entry['net_lift_excluding_best_draw']:+.2f} |",
            f"| Exact one-sided sign-test p | {entry['sign_test_p_value']:.4g} |",
        ]
    )


def build_notifications(
    report: dict[str, Any], gates: Sequence[Gate], thresholds: Thresholds, version: str
) -> list[dict[str, str]]:
    """One notification per milestone actually reached. Nothing speculative, nothing repeated."""
    notifications = []
    quoted = f"Thresholds applied: `{json.dumps(thresholds.as_dict(), sort_keys=True)}`"

    if report["graduation"]["ready"]:
        key = notification_key(f"graduation-{version}")
        notifications.append(
            {
                "key": key,
                "title": f"Stable-release gates met at {version}",
                "body": "\n\n".join(
                    [
                        f"Every stable-release gate is satisfied at `{version}`.",
                        _gate_table(gates),
                        "This is a prompt to decide, not a decision. Before releasing, confirm "
                        "that the `Development Status` classifier, the version string, and the "
                        "README status block still describe reality.",
                        quoted,
                        f"<!-- lottobench-watch: {key} -->",
                    ]
                ),
            }
        )

    for entry in report["provider_evidence"]["qualifying"]:
        digest = (entry["provider_config_sha256"] or "noconfig")[:12]
        label = f"{entry['provider_name']}@{entry['provider_version']}"
        key = notification_key(f"evidence-{entry['provider_name']}-{entry['provider_version']}-{digest}")
        notifications.append(
            {
                "key": key,
                "title": (
                    f"{label} beat {CONTROL_PROVIDER} on settled {entry['game']} draws"
                ),
                "body": "\n\n".join(
                    [
                        f"`{label}` cleared every evidence threshold against `{CONTROL_PROVIDER}` "
                        f"on settled {entry['game']} draws.",
                        _evidence_table(entry),
                        "The sign test and the leave-the-best-draw-out check are both required "
                        "precisely so that one large payout cannot open this issue on its own. "
                        "This is still prospective evidence from a single ledger, not a claim of "
                        "an exploitable edge, and it is not betting advice. Replicate it on an "
                        "independent ledger before writing it anywhere a reader could mistake "
                        "for a recommendation.",
                        quoted,
                        f"<!-- lottobench-watch: {key} -->",
                    ]
                ),
            }
        )
    return notifications


def build_report(
    *,
    ledger: Path,
    pypi_report: dict[str, Any] | None,
    open_critical_issues: int | None,
    history: Path | None,
    game: str,
    budget: int,
    repro_rows: int,
    repro_holdout: int,
    thresholds: Thresholds,
    pythons: Sequence[str],
    skip_benchmark: bool,
    version: str,
) -> dict[str, Any]:
    pairs, ledger_problems = paired_draws(read_ledger(ledger))
    scored = [evaluate_provider(key, draws, thresholds) for key, draws in sorted(pairs.items())]
    scored.sort(key=lambda row: (not row["qualifies"], -row["net_lift"], row["provider_name"]))

    tags = release_tags()
    gates = [
        gate_pypi_install(pypi_report, pythons),
        gate_release_history(tags, thresholds),
        gate_public_api(tags),
        gate_ledger_schema(tags),
        gate_settled_observations(pairs, thresholds),
        gate_no_critical_issues(open_critical_issues, ledger_problems),
    ]
    if skip_benchmark:
        detail = "Reproducibility check was skipped by request."
        gates.append(Gate("benchmark_reproducible", UNKNOWN, detail))
    else:
        gates.append(gate_benchmark_reproducible(history, game, budget, repro_holdout, repro_rows))
    gates.sort(key=lambda gate: gate.name)

    report: dict[str, Any] = {
        "lottobench_version": version,
        "roi_schema_version": ROI_SCHEMA_VERSION,
        "thresholds": thresholds.as_dict(),
        "ledger": str(ledger),
        "ledger_problems": ledger_problems,
        "graduation": {
            "ready": all(gate.met for gate in gates),
            "gates": [gate.as_dict() for gate in gates],
        },
        "provider_evidence": {
            "ready": any(row["qualifies"] for row in scored),
            "qualifying": [row for row in scored if row["qualifies"]],
            "candidates": scored,
        },
    }
    report["notifications"] = build_notifications(report, gates, thresholds, version)
    return report


def _print_summary(report: dict[str, Any]) -> None:
    print(f"LottoBench {report['lottobench_version']} -- graduation watch")
    print()
    for gate in report["graduation"]["gates"]:
        print(f"  {gate['status']:<8} {gate['name']:<26} {gate['detail']}")
    print()
    ready = report["graduation"]["ready"]
    print(f"  stable-release gates: {'ALL MET' if ready else 'not yet met'}")

    candidates = report["provider_evidence"]["candidates"]
    if not candidates:
        print("  provider evidence:    no settled control-matched draws in the ledger")
    for row in candidates:
        label = f"{row['provider_name']}@{row['provider_version']}"
        verdict = "QUALIFIES" if row["qualifies"] else "; ".join(row["blocking_reasons"])
        print(f"  provider evidence:    {label:<32} {verdict}")
    if report["notifications"]:
        keys = ", ".join(entry["key"] for entry in report["notifications"])
        print(f"\n  notifications to open: {keys}")


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ledger", type=Path, default=Path("ledger/euromillions"))
    parser.add_argument("--history", type=Path, default=Path("data/euromillions.csv"))
    parser.add_argument("--game", default="euromillions")
    parser.add_argument("--budget", type=int, default=25)
    parser.add_argument(
        "--pypi-report",
        type=Path,
        default=None,
        help="JSON mapping supported Python versions to install success",
    )
    parser.add_argument(
        "--open-critical-issues",
        type=int,
        default=None,
        help="Count of open critical packaging/data-integrity issues; omit to leave it unknown",
    )
    parser.add_argument("--min-draws", type=int, default=Thresholds.min_draws)
    parser.add_argument("--min-decisive-draws", type=int, default=Thresholds.min_decisive_draws)
    parser.add_argument("--alpha", type=float, default=Thresholds.alpha)
    parser.add_argument("--min-settled-draws", type=int, default=Thresholds.min_settled_draws)
    parser.add_argument("--min-prereleases", type=int, default=Thresholds.min_prereleases)
    parser.add_argument(
        "--repro-rows",
        type=int,
        default=120,
        help="Most recent draws replayed by the reproducibility gate",
    )
    parser.add_argument(
        "--repro-holdout",
        type=int,
        default=3,
        help="Holdout draws used by the reproducibility gate; small on purpose, see its docstring",
    )
    parser.add_argument(
        "--skip-benchmark", action="store_true", help="Skip the repeated benchmark run"
    )
    parser.add_argument("--out", type=Path, default=None, help="Write the full JSON report here")
    parser.add_argument(
        "--notify-dir",
        type=Path,
        default=None,
        help="Write one <key>.json/<key>.md pair per milestone reached",
    )
    args = parser.parse_args(argv)

    thresholds = Thresholds(
        min_draws=args.min_draws,
        min_decisive_draws=args.min_decisive_draws,
        alpha=args.alpha,
        min_settled_draws=args.min_settled_draws,
        min_prereleases=args.min_prereleases,
    )
    report = build_report(
        ledger=args.ledger,
        pypi_report=_load_json(args.pypi_report),
        open_critical_issues=args.open_critical_issues,
        history=args.history,
        game=args.game,
        budget=args.budget,
        repro_rows=args.repro_rows,
        repro_holdout=args.repro_holdout,
        thresholds=thresholds,
        pythons=SUPPORTED_PYTHONS,
        skip_benchmark=args.skip_benchmark,
        version=__version__,
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.notify_dir:
        args.notify_dir.mkdir(parents=True, exist_ok=True)
        for entry in report["notifications"]:
            (args.notify_dir / f"{entry['key']}.md").write_text(entry["body"], encoding="utf-8")
            (args.notify_dir / f"{entry['key']}.json").write_text(
                json.dumps(entry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
