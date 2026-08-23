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
import tempfile
from pathlib import Path

from lotteries_core import dataset, storage


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
    ap.add_argument("--out", default="data/lotteries.db", help="SQLite database (or legacy CSV)")
    ap.add_argument("--game", default="euromillions", choices=["euromillions"])
    ap.add_argument(
        "--max-age-days",
        type=int,
        default=dataset.DEFAULT_STALE_AFTER_DAYS,
        help="how old the newest draw may be before the history counts as stale",
    )
    ap.add_argument("--check", action="store_true", help="report status only; fetch nothing")
    ap.add_argument(
        "--allow-stale",
        action="store_true",
        help="fall back to local/bundled data when every upstream source fails",
    )
    args = ap.parse_args(argv)

    out = Path(args.out)
    if args.check:
        return _check(out, args.max_age_days)

    before = dataset.read(out, game=args.game)
    # Call the library entry point rather than the CLI `main()`, which parses sys.argv directly.
    from euromillions.get_draws import fetch_and_normalize

    out.parent.mkdir(parents=True, exist_ok=True)
    fetch_path = out
    temporary = None
    if storage.is_database(out):
        temporary = tempfile.TemporaryDirectory(prefix="mslt-refresh-")
        fetch_path = Path(temporary.name) / f"{args.game}.csv"
    try:
        result = fetch_and_normalize(out_path=fetch_path, allow_stale=args.allow_stale)
    except Exception as exc:
        print(f"[refresh] fetch failed: {exc}", file=sys.stderr)
        if temporary is not None:
            temporary.cleanup()
        return 1
    print(f"[refresh] fetched {len(result.dataframe)} rows (cache: {result.cache_path})")

    if storage.is_database(out):
        storage.write_history(out, result.dataframe, game=args.game)
        temporary.cleanup()

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
