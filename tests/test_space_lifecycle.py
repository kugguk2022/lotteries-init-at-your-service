from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "publishing"
    / "huggingface-space"
    / "lifecycle.py"
)
SPEC = importlib.util.spec_from_file_location("space_lifecycle", MODULE_PATH)
lifecycle = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = lifecycle
SPEC.loader.exec_module(lifecycle)


def _state(target: str | None, *, available: bool = True):
    return lifecycle.classify_profile_state(
        data_kind="observed_public_history",
        target_draw_date=target,
        publication_available=available,
        today=date(2026, 9, 16),
    )


def test_synthetic_control_is_never_labelled_pending():
    state = lifecycle.classify_profile_state(
        data_kind="deterministic_synthetic",
        target_draw_date=None,
        publication_available=False,
        today=date(2026, 9, 16),
    )
    assert state.code == "DEMONSTRATION_ONLY"
    assert state.label == "DEMO ONLY — NOT PENDING"
    assert "does not transition" in state.exit_condition


def test_observed_profile_has_explicit_date_driven_lifecycle():
    assert _state("2026-09-17").code == "PRE_DRAW_PUBLISHED"
    assert _state("2026-09-16").code == "DRAW_DAY_PENDING"
    assert _state("2026-09-15").code == "AWAITING_VERIFIED_RESULT"


def test_missing_or_invalid_publication_cannot_claim_a_pending_draw():
    assert _state(None).code == "PUBLICATION_DATA_MISSING"
    assert _state("2026-09-17", available=False).code == "PUBLICATION_DATA_MISSING"
    assert _state("17-09-2026").code == "INVALID_TARGET_DATE"


def test_guide_defines_state_exit_conditions_and_space_settlement_boundary():
    guide = MODULE_PATH.with_name("SPACE_GUIDE.md").read_text(encoding="utf-8")
    for term in (
        "DEMONSTRATION_ONLY",
        "PRE_DRAW_PUBLISHED",
        "DRAW_DAY_PENDING",
        "AWAITING_VERIFIED_RESULT",
        "first-class `SETTLED` rows",
        "Machine-readable reading order",
    ):
        assert term in guide


def test_app_exposes_guide_and_uses_predraw_not_pending_as_navigation_label():
    app = MODULE_PATH.with_name("app.py").read_text(encoding="utf-8")
    assert "START HERE / COMPLETE SPACE WIKI" in app
    assert "SCREEN B / PRE-DRAW SET LAB" in app
    assert "SCREEN B / PENDING SET LAB" not in app


def test_space_hydration_retries_api_then_fails_closed_after_git_fallback():
    workflow = (
        MODULE_PATH.parents[2] / ".github" / "workflows" / "publish-space.yml"
    ).read_text(encoding="utf-8")
    assert workflow.count("for attempt in 1 2 3; do") == 2
    assert workflow.count("git clone --depth 1") == 2
    assert workflow.count("refusing to replace current profile data") == 2
    assert "using repository fallback profiles" not in workflow


def test_space_publication_retries_api_then_verifies_git_fallback():
    root = MODULE_PATH.parents[2]
    workflow = (root / ".github" / "workflows" / "publish-space.yml").read_text(
        encoding="utf-8"
    )
    helper = (root / "scripts" / "publish_space_bundle.sh").read_text(encoding="utf-8")
    assert workflow.count("scripts/publish_space_bundle.sh") == 3
    assert "for attempt in 1 2 3; do" in helper
    assert "Space upload API remained unavailable; publishing through Git" in helper
    assert "http.extraheader" in helper
    assert "test \"$local_sha\" = \"$remote_sha\"" in helper
