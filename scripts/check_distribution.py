"""Fail when built distributions contain repository data, outputs, or non-core packages."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

FORBIDDEN_ROOTS = {
    "data",
    "eurodreams",
    "euromillions",
    "euromillions_agent",
    "ledger",
    "lotto_lab_out",
    "outputs",
    "runs",
    "tests",
    "totoloto",
}
FORBIDDEN_SUFFIXES = {".csv", ".jsonl", ".parquet", ".png", ".xlsx"}
REQUIRED = {"lotteries_core/registry.py", "lotteries_core/roi.py"}


def _members(path: Path) -> list[str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return archive.namelist()
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            names = archive.getnames()
        return [name.split("/", 1)[1] for name in names if "/" in name]
    raise ValueError(f"unsupported distribution archive: {path}")


def check(path: Path) -> None:
    names = _members(path)
    normalized = {str(PurePosixPath(name)) for name in names}
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
        raise SystemExit(f"{path}: repository artifacts leaked into distribution: {sorted(leaked)}")
    print(f"{path}: {len(normalized)} files checked; core-only contents verified")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archives", nargs="+", type=Path)
    args = parser.parse_args()
    for archive in args.archives:
        check(archive)


if __name__ == "__main__":
    main()
