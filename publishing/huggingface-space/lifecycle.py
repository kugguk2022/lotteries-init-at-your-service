"""Small, dependency-free lifecycle contract for the public LottoBench Space."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone


@dataclass(frozen=True)
class LifecycleState:
    """Human- and machine-readable interpretation of a profile's current slate."""

    code: str
    label: str
    explanation: str
    exit_condition: str


def classify_profile_state(
    *,
    data_kind: str,
    target_draw_date: str | None,
    publication_available: bool,
    today: date | None = None,
) -> LifecycleState:
    """Classify a profile without implying a result has been settled.

    Space candidate files are current publications, not the canonical prospective
    settlement ledger. A successful post-draw refresh advances the live profile and
    leaves the old publication in Hugging Face revision history.
    """

    if data_kind == "deterministic_synthetic":
        return LifecycleState(
            code="DEMONSTRATION_ONLY",
            label="DEMO ONLY — NOT PENDING",
            explanation=(
                "This fixed synthetic control tests the interface and scoring machinery. "
                "It is not mapped to an operator draw or an automatic refresh."
            ),
            exit_condition=(
                "It does not transition to a real result. Select EuroMillions or NL Lotto "
                "for dated pre-draw publications."
            ),
        )

    if not publication_available or not target_draw_date:
        return LifecycleState(
            code="PUBLICATION_DATA_MISSING",
            label="PUBLICATION DATA UNAVAILABLE",
            explanation=(
                "The observed profile is loaded, but the files needed to identify a current "
                "target draw are missing from this build."
            ),
            exit_condition=(
                "A validated profile build must generate the candidate files and publish them "
                "to the Space."
            ),
        )

    try:
        target = date.fromisoformat(target_draw_date)
    except ValueError:
        return LifecycleState(
            code="INVALID_TARGET_DATE",
            label="INVALID TARGET DATE",
            explanation="The publication does not carry an ISO-formatted target draw date.",
            exit_condition="Rebuild the profile with a valid YYYY-MM-DD target draw date.",
        )

    current = today or datetime.now(timezone.utc).date()
    if target > current:
        return LifecycleState(
            code="PRE_DRAW_PUBLISHED",
            label="PENDING — PUBLISHED BEFORE TARGET DRAW",
            explanation=(
                "The candidates are frozen against the displayed history cutoff for a future "
                "target draw. Pending does not mean waiting for approval or model confidence."
            ),
            exit_condition=(
                "The target draw must occur, its official result must be fetched and validated, "
                "and the next scheduled Space publication must complete."
            ),
        )
    if target == current:
        return LifecycleState(
            code="DRAW_DAY_PENDING",
            label="PENDING — TARGET DRAW DAY",
            explanation=(
                "The candidates still target today's draw. Date alone cannot prove whether a "
                "same-day publication preceded the official draw time."
            ),
            exit_condition=(
                "After the draw, the official result must be fetched and validated and the next "
                "scheduled Space publication must complete."
            ),
        )
    return LifecycleState(
        code="AWAITING_VERIFIED_RESULT",
        label="PENDING — AWAITING VERIFIED RESULT REFRESH",
        explanation=(
            "The target date has passed, but the live profile has not yet advanced. The same "
            "dated files remain visible rather than being silently relabeled or regenerated."
        ),
        exit_condition=(
            "The official source must return the target result, every profile and artifact "
            "validation must pass, and Hugging Face publication plus the public health check "
            "must succeed. A failed step preserves this revision."
        ),
    )
