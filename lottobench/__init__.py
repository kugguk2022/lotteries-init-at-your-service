"""LottoBench — auditable benchmarking for lottery strategies.

The public package is intentionally small. Existing ``lotteries_core`` imports remain supported
for compatibility with the research code.
"""

from lotteries_core import GameSpec, InferenceProvider, Ticket
from lotteries_core.registry import available, create, names

from .games import GAMES, GameDefinition, game

__all__ = [
    "GAMES",
    "GameDefinition",
    "GameSpec",
    "InferenceProvider",
    "Ticket",
    "available",
    "create",
    "game",
    "names",
]
