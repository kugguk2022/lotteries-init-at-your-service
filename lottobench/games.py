"""National game catalogue.

``GAMES`` lists only what LottoBench supports **end to end**: retrieval, normalization, storage,
benchmarking, and realized-ROI settlement all working against the same game.

Games whose combinatorial shape is known but whose data path is not implemented live in
``BACKLOG_GAMES``. They are deliberately not in ``GAMES``, because listing a game the user cannot
actually fetch or benchmark is the misleading part of a package journey -- the catalogue would be
advertising six games and delivering one. Each moves across only when it passes the same contract:
see ``docs/wiki/Backlog.md``.

Only numbers selected by the player belong in ``GameSpec``. Drawn bonus balls and Germany's
pre-printed Superzahl remain source/result metadata rather than selectable ticket fields.
"""

from __future__ import annotations

from dataclasses import dataclass

from lotteries_core.protocol import GameSpec


@dataclass(frozen=True)
class GameDefinition:
    key: str
    country_code: str
    display_name: str
    spec: GameSpec
    source_url: str
    notes: str = ""


@dataclass(frozen=True)
class BacklogGame:
    """A game whose shape is known but which has no supported end-to-end path yet."""

    key: str
    country_code: str
    display_name: str
    source_url: str
    #: What is missing before this can move into ``GAMES``.
    blocked_on: str


GAMES: dict[str, GameDefinition] = {
    "euromillions": GameDefinition(
        "euromillions", "EU", "EuroMillions", GameSpec.euromillions(),
        "https://www.euro-millions.com/",
        "Supported end to end: fetch, store, benchmark, and settle.",
    ),
    "nl-lotto": GameDefinition(
        "nl-lotto", "NL", "Nederlandse Lotto", GameSpec("nl-lotto", 45, 6),
        "https://lotto.nederlandseloterij.nl/trekkingsuitslag",
        "Primary Lotto series: six selected numbers; reserve number and jackpot colour are metadata.",
    ),
}


BACKLOG_GAMES: dict[str, BacklogGame] = {
    "dk-lotto": BacklogGame(
        "dk-lotto", "DK", "Danske Lotto",
        "https://danskespil.dk/regler--a--vilkaar/regler/spilleregler_dlo",
        "No retrieval adapter; separately drawn bonus number is not a selectable field.",
    ),
    "de-lotto-6aus49": BacklogGame(
        "de-lotto-6aus49", "DE", "LOTTO 6aus49",
        "https://www.lotto.de/lotto-6aus49/spielregeln",
        "No retrieval adapter; Superzahl is 0-9 and the core pool model is 1-based.",
    ),
    "uk-lotto": BacklogGame(
        "uk-lotto", "GB", "UK Lotto",
        "https://www.national-lottery.co.uk/games/lotto",
        "No retrieval adapter; bonus ball is drawn from the remaining main pool.",
    ),
    "se-lotto": BacklogGame(
        "se-lotto", "SE", "Svenska Spel Lotto",
        "https://www.svenskaspel.se/lotto/spelguide",
        "No retrieval adapter.",
    ),
}


def game(key: str) -> GameDefinition:
    """Return a supported game definition.

    A key that is merely on the backlog gets its own message: "not supported yet, and here is
    why" is actionable, whereas "unknown game" would suggest a typo.
    """
    try:
        return GAMES[key]
    except KeyError as exc:
        if key in BACKLOG_GAMES:
            entry = BACKLOG_GAMES[key]
            raise KeyError(
                f"{key!r} ({entry.display_name}) is defined but not supported end to end: "
                f"{entry.blocked_on} See docs/wiki/Backlog.md. Supported: {sorted(GAMES)}"
            ) from exc
        raise KeyError(
            f"unknown game {key!r}; supported: {sorted(GAMES)}; "
            f"on the backlog: {sorted(BACKLOG_GAMES)}"
        ) from exc
