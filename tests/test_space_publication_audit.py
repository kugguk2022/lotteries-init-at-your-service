from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "publishing" / "huggingface-space" / "publication_audit.py"
)
SPEC = importlib.util.spec_from_file_location("publication_audit", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@pytest.fixture
def publication(tmp_path):
    cutoff, target, snapshot = "2026-09-11", "2026-09-15", "a" * 64
    rows = []
    for rank, main in enumerate(((1, 2, 3, 4, 5), (6, 7, 8, 9, 10)), start=1):
        auxiliary = (1, 2)
        source = f"{snapshot}|after:{cutoff}|unpopularity|{rank}|{main}|{auxiliary}"
        rows.append({
            "rank": rank,
            "main_draw": " ".join(map(str, main)),
            "auxiliary_draw": "1 2",
            "history_cutoff": cutoff,
            "target_draw_date": target,
            "score_status": "PENDING",
            "commitment_sha256": hashlib.sha256(source.encode()).hexdigest(),
        })
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    slate = buffer.getvalue().encode()
    summary = {
        "provider": "unpopularity", "status": "PENDING", "ticket_count": 2,
        "history_cutoff": cutoff, "target_draw_date": target, "snapshot_sha256": snapshot,
        "artifacts": {audit.SLATE_FILE: hashlib.sha256(slate).hexdigest()},
    }
    summary_bytes = json.dumps(summary).encode()
    (tmp_path / audit.SLATE_FILE).write_bytes(slate)
    (tmp_path / audit.SUMMARY_FILE).write_bytes(summary_bytes)
    revision = "b" * 40
    links = audit.public_links("euromillions", revision)
    tree = [{
        "path": "data/profiles/euromillions/crowd_escape_summary.json",
        "lastCommit": {"id": revision, "date": "2026-09-12T12:43:02Z"},
    }]
    responses = {
        f"{audit.API_URL}/tree/main/data/profiles/euromillions?expand=true": json.dumps(tree).encode(),
        links["manifest"]: summary_bytes,
        links["slate"]: slate,
    }
    return tmp_path, slate, summary, responses, links


def test_matches_dated_revision_and_reconstructs_commitments(publication):
    directory, _, summary, responses, links = publication
    receipt = audit.audit_publication(directory, "euromillions", read_url=responses.__getitem__)
    assert receipt["revision"] == "b" * 40
    assert receipt["published_utc"] == "2026-09-12T12:43:02Z"
    assert receipt["slate_sha256"] == summary["artifacts"][audit.SLATE_FILE]
    assert receipt["links"] == links
    assert receipt["timing"] == "Published before the target draw date"


def test_rejects_modified_csv(publication):
    _, slate, summary, _, _ = publication
    with pytest.raises(ValueError, match="published SHA-256"):
        audit.verify_slate(slate.replace(b"1 2 3 4 5", b"1 2 3 4 6"), summary)


def test_rejects_rehashed_csv_with_broken_ticket_commitment(publication):
    _, slate, summary, _, _ = publication
    changed = slate.replace(b"1 2 3 4 5", b"1 2 3 4 6")
    summary["artifacts"][audit.SLATE_FILE] = hashlib.sha256(changed).hexdigest()
    with pytest.raises(ValueError, match="original commitment"):
        audit.verify_slate(changed, summary)


def test_rejects_local_remote_version_mismatch(publication):
    directory, _, _, responses, links = publication
    responses[links["slate"]] += b"\n"
    with pytest.raises(ValueError, match="differs from the latest public record"):
        audit.audit_publication(directory, "euromillions", read_url=responses.__getitem__)


def test_network_failure_is_not_a_verified_publication(publication):
    directory = publication[0]

    def unavailable(url):
        raise OSError("publication service unavailable")

    with pytest.raises(OSError, match="publication service unavailable"):
        audit.audit_publication(directory, "euromillions", read_url=unavailable)
    assert (directory / audit.SLATE_FILE).is_file()
    assert audit.public_links("euromillions")["history"].endswith("/data/profiles/euromillions")


@pytest.mark.parametrize("timestamp, expected", [
    ("2026-09-14T23:59:59Z", "Published before the target draw date"),
    ("2026-09-15T00:00:00Z", "Published on draw day; compare the timestamp with the official draw time"),
    ("2026-09-16T00:00:00Z", "Published after the target draw date; this revision is not pre-draw evidence"),
    ("2026-09-15T00:30:00+01:00", "Published before the target draw date"),
])
def test_never_labels_draw_day_or_late_records_as_verified_pre_draw(timestamp, expected):
    assert audit.publication_timing(timestamp, "2026-09-15") == expected


def test_rejects_undated_or_synthetic_publication_records():
    with pytest.raises(ValueError, match="timezone"):
        audit.publication_timing("2026-09-14T12:00:00", "2026-09-15")
    with pytest.raises(ValueError, match="observed lottery"):
        audit.public_links("synthetic")
    with pytest.raises(ValueError, match="full Space commit SHA"):
        audit.public_links("euromillions", "moving-branch")
