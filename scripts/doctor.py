"""Fail early with an actionable setup message for core development commands."""

from __future__ import annotations

import importlib.util
import sys

REQUIRED = ("numpy", "pandas", "pytest")


def main() -> int:
    missing = [name for name in REQUIRED if importlib.util.find_spec(name) is None]
    if missing:
        print(f"LottoBench setup is incomplete for {sys.executable}.", file=sys.stderr)
        print(f"Missing: {', '.join(missing)}", file=sys.stderr)
        print("Run: make setup PYTHON=python", file=sys.stderr)
        return 2
    print(f"LottoBench core environment ready: Python {sys.version.split()[0]} ({sys.executable})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
