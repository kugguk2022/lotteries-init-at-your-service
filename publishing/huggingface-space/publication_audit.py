"""Verify a displayed slate against a dated, revision-pinned public Space record.

Publication dates come from Hugging Face's file history, never from the history
cutoff or a locally generated timestamp. This module uses only the standard library.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.request import urlopen

SPACE_URL = "https://huggingface.co/spaces/kugguk/lottobench-community-leaderboard"
API_URL = "https://huggingface.co/api/spaces/kugguk/lottobench-community-leaderboard"
SLATE_FILE = "crowd_escape_forecasted_draws.csv"
SUMMARY_FILE = "crowd_escape_summary.json"
PROFILE_KEYS = ("euromillions", "nl-lotto")


def public_links(key: str, revision: str = "main") -> dict[str, str]:
    if key not in PROFILE_KEYS:
        raise ValueError("Publication records apply to observed lottery profiles only")
    if revision != "main" and not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("A full Space commit SHA is required")
    directory = f"data/profiles/{key}"
    return {
        "slate": f"{SPACE_URL}/resolve/{revision}/{directory}/{SLATE_FILE}",
        "manifest": f"{SPACE_URL}/resolve/{revision}/{directory}/{SUMMARY_FILE}",
        "million_set": (
            f"{SPACE_URL}/resolve/{revision}/{directory}/temporal_hybrid_candidates_1m.csv.gz"
        ),
        "million_manifest": (
            f"{SPACE_URL}/resolve/{revision}/{directory}/temporal_hybrid_summary.json"
        ),
        "files": f"{SPACE_URL}/tree/{revision}/{directory}",
        "history": f"{SPACE_URL}/commits/main/{directory}",
        "commit": f"{SPACE_URL}/commit/{revision}",
    }


def verify_slate(slate: bytes, summary: dict) -> None:
    """Check the full file and reconstruct every ticket's original commitment."""
    digest = hashlib.sha256(slate).hexdigest()
    if digest != summary["artifacts"][SLATE_FILE]:
        raise ValueError("Crowd Escape CSV does not match its published SHA-256")
    if summary["provider"] != "unpopularity" or summary["status"] != "PENDING":
        raise ValueError("Expected an unsettled Crowd Escape publication")
    cutoff = date.fromisoformat(summary["history_cutoff"])
    target = date.fromisoformat(summary["target_draw_date"])
    if cutoff >= target:
        raise ValueError("History cutoff must precede the target draw")
    rows = list(csv.DictReader(io.StringIO(slate.decode("utf-8"))))
    if len(rows) != summary["ticket_count"] or not rows:
        raise ValueError("Published ticket count does not match the CSV")
    ranks = [int(row["rank"]) for row in rows]
    if sorted(ranks) != list(range(1, len(rows) + 1)):
        raise ValueError("Published ranks must be consecutive and unique")
    for row in rows:
        if (row["history_cutoff"] != str(cutoff)
                or row["target_draw_date"] != str(target)
                or row["score_status"] != "PENDING"):
            raise ValueError("Ticket dates or status disagree with the publication manifest")
        main = tuple(int(value) for value in row["main_draw"].split())
        auxiliary = tuple(int(value) for value in row["auxiliary_draw"].split())
        source = (
            f"{summary['snapshot_sha256']}|after:{cutoff}|unpopularity|"
            f"{int(row['rank'])}|{main}|{auxiliary}"
        )
        if hashlib.sha256(source.encode("utf-8")).hexdigest() != row["commitment_sha256"]:
            raise ValueError(f"Ticket {row['rank']} does not match its original commitment")


def publication_timing(published_utc: str, target_draw_date: str) -> str:
    timestamp = datetime.fromisoformat(published_utc.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("Publication timestamp must include a timezone")
    published_day = timestamp.astimezone(timezone.utc).date()
    target = date.fromisoformat(target_draw_date)
    if published_day < target:
        return "Published before the target draw date"
    if published_day == target:
        return "Published on draw day; compare the timestamp with the official draw time"
    return "Published after the target draw date; this revision is not pre-draw evidence"


def _read_url(url: str) -> bytes:
    with urlopen(url, timeout=10) as response:
        return response.read()


def audit_publication(directory: Path, key: str, *, read_url=None) -> dict:
    """Return evidence only when the displayed bytes match a public commit.

    On failure, callers must show verification as unavailable, while keeping the
    existing downloads and public history reachable. No source refresh is performed.
    """
    links = public_links(key)
    read_url = read_url or _read_url
    summary_bytes = (directory / SUMMARY_FILE).read_bytes()
    summary = json.loads(summary_bytes)
    slate = (directory / SLATE_FILE).read_bytes()
    verify_slate(slate, summary)
    tree = json.loads(read_url(f"{API_URL}/tree/main/data/profiles/{key}?expand=true"))
    entry = next(
        (item for item in tree if item.get("path") == f"data/profiles/{key}/{SUMMARY_FILE}"),
        None,
    )
    if entry is None:
        raise ValueError("No public publication manifest was found")
    commit = entry["lastCommit"]
    pinned = public_links(key, commit["id"])
    if read_url(pinned["manifest"]) != summary_bytes or read_url(pinned["slate"]) != slate:
        raise ValueError("Displayed slate differs from the latest public record; reload the Space")
    return {
        "revision": commit["id"],
        "published_utc": commit["date"],
        "target_draw_date": summary["target_draw_date"],
        "history_cutoff": summary["history_cutoff"],
        "ticket_count": summary["ticket_count"],
        "slate_sha256": summary["artifacts"][SLATE_FILE],
        "timing": publication_timing(commit["date"], summary["target_draw_date"]),
        "links": pinned,
        "history_url": links["history"],
    }
