"""National game catalogue used by the library and local API.

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


GAMES: dict[str, GameDefinition] = {
    "dk-lotto": GameDefinition(
        "dk-lotto",
        "DK",
        "Danske Lotto",
        GameSpec("dk-lotto", 36, 7),
        "https://danskespil.dk/regler--a--vilkaar/regler/spilleregler_dlo",
        "The separately drawn bonus number is not selected on a standard line.",
    ),
    "de-lotto-6aus49": GameDefinition(
        "de-lotto-6aus49",
        "DE",
        "LOTTO 6aus49",
        GameSpec("de-lotto-6aus49", 49, 6),
        "https://www.lotto.de/lotto-6aus49/spielregeln",
        "Superzahl is derived from the ticket number, not selected as one of the six numbers.",
    ),
    "uk-lotto": GameDefinition(
        "uk-lotto",
        "GB",
        "UK Lotto",
        GameSpec("uk-lotto", 59, 6),
        "https://www.national-lottery.co.uk/games/lotto",
        "The bonus ball is drawn from the remaining pool and is not selected separately.",
    ),
    "nl-lotto": GameDefinition(
        "nl-lotto",
        "NL",
        "Nederlandse Lotto",
        GameSpec("nl-lotto", 45, 6),
        "https://www.lotto.nl/lotto/hoe-werkt-het",
        "Base six-number line; add-on game attributes are kept outside the core ticket shape.",
    ),
    "se-lotto": GameDefinition(
        "se-lotto",
        "SE",
        "Svenska Spel Lotto",
        GameSpec("se-lotto", 35, 7),
        "https://www.svenskaspel.se/lotto/spelguide",
    ),
    "euromillions": GameDefinition(
        "euromillions",
        "EU",
        "EuroMillions",
        GameSpec.euromillions(),
        "https://www.euro-millions.com/",
    ),
}


def game(key: str) -> GameDefinition:
    """Return a game definition or raise a useful key error."""
    try:
        return GAMES[key]
    except KeyError as exc:
        raise KeyError(f"unknown game {key!r}; available: {sorted(GAMES)}") from exc
