"""Reject distribution archives that leak repository data or omit public package files."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

FORBIDDEN_ROOTS = {"data", "ledger", "outputs", "runs", "tests"}
FORBIDDEN_SUFFIXES = {".csv", ".db", ".jsonl", ".parquet", ".pt", ".xlsx"}
REQUIRED = {
    "lottobench/__init__.py",
    "lottobench/cli.py",
    "lottobench/games.py",
    "lotteries_core/registry.py",
    "lotteries_core/storage.py",
}


def _members(path: Path) -> list[str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return archive.namelist()
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            return [name.split("/", 1)[1] for name in archive.getnames() if "/" in name]
    raise ValueError(f"unsupported distribution archive: {path}")


def check(path: Path) -> None:
    normalized = {str(PurePosixPath(name)) for name in _members(path)}
    missing = REQUIRED - normalized
    if missing:
        raise SystemExit(f"{path}: required package files missing: {sorted(missing)}")
    leaked = []
    for name in normalized:
        parts = PurePosixPath(name).parts
        if not parts:
            continue
        if parts[0] in FORBIDDEN_ROOTS or PurePosixPath(name).suffix.lower() in FORBIDDEN_SUFFIXES:
            leaked.append(name)
    if leaked:
        raise SystemExit(f"{path}: repository artifacts leaked: {sorted(leaked)}")
    if any(name.startswith("mslt/") for name in normalized):
        raise SystemExit(f"{path}: obsolete mslt namespace leaked into distribution")
    print(f"{path}: {len(normalized)} files checked; contents verified")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archives", nargs="+", type=Path)
    args = parser.parse_args()
    for archive in args.archives:
        check(archive)


if __name__ == "__main__":
    main()
