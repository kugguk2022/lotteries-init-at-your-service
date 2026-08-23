"""LottoBench — auditable benchmarking for lottery strategies.

The public package is intentionally small. Existing ``lotteries_core`` imports remain supported
for compatibility with the research code.
"""

from lotteries_core import GameSpec, InferenceProvider, Ticket
from lotteries_core.realized_roi import comparison as compare_realized_roi
from lotteries_core.realized_roi import export_bundle as export_roi_bundle
from lotteries_core.registry import available, create, names

from .games import GAMES, GameDefinition, game

__all__ = [
    "GAMES",
    "GameDefinition",
    "GameSpec",
    "InferenceProvider",
    "Ticket",
    "available",
    "compare_realized_roi",
    "create",
    "game",
    "export_roi_bundle",
    "names",
]
