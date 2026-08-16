"""Refresh a draw history and rewrite its canonical metadata sidecar.

This is the single supported way to update the datasets the documented results and the prospective
ledger are computed on. Run it manually, or let `.github/workflows/refresh-history.yml` run it on a
schedule.

    python scripts/refresh_history.py                      # refresh EuroMillions into data/
    python scripts/refresh_history.py --check              # report staleness, change nothing
    python scripts/refresh_history.py --out data/em.csv

Exit codes: ``0`` success (``--check``: fresh), ``1`` failure, ``2`` (``--check`` only) stale.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lotteries_core import dataset


def _check(out: Path, max_age_days: int) -> int:
    if not out.exists():
        print(f"[check] {out} does not exist -- run without --check to fetch it")
        return 2
    ok, reason = dataset.verify(out)
    print(f"[check] {'ok' if ok else 'DRIFT'}: {reason}")
    days = dataset.staleness_days(out)
    stale = days > max_age_days
    print(f"[check] newest draw is {days} day(s) old (threshold {max_age_days})")
    if stale:
        print("[check] STALE -- refresh before recording predictions or publishing results")
    return 2 if (stale or not ok) else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/euromillions.csv", help="history CSV to write")
    ap.add_argument("--game", default="euromillions", choices=["euromillions"])
    ap.add_argument(
        "--max-age-days",
        type=int,
        default=dataset.DEFAULT_STALE_AFTER_DAYS,
        help="how old the newest draw may be before the history counts as stale",
    )
    ap.add_argument("--check", action="store_true", help="report status only; fetch nothing")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if args.check:
        return _check(out, args.max_age_days)

    before = dataset.read(out)
    from euromillions.get_draws import main as fetch_main

    try:
        fetch_main(["--out", str(out)])
    except SystemExit as exc:  # argparse/`FetchError` surfaced by the fetcher CLI
        if exc.code:
            print(f"[refresh] fetch failed with exit code {exc.code}", file=sys.stderr)
            return 1
    except Exception as exc:
        print(f"[refresh] fetch failed: {exc}", file=sys.stderr)
        return 1

    meta = dataset.write(out, game=args.game)
    changed = before is None or before.content_sha256 != meta.content_sha256
    print(f"[refresh] {meta.rows} draws, {meta.first_draw} .. {meta.last_draw}")
    print(f"[refresh] sha256 {meta.content_sha256[:16]} -> {dataset.meta_path(out)}")
    if before is not None:
        delta = meta.rows - before.rows
        print(f"[refresh] {'changed' if changed else 'unchanged'} ({delta:+d} rows since last refresh)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
