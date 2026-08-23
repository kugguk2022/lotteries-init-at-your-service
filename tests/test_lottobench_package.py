from __future__ import annotations

import pytest

import lottobench


@pytest.mark.parametrize(
    ("key", "main_n", "main_k"), [("euromillions", 50, 5), ("nl-lotto", 45, 6)]
)
def test_supported_game_catalogue(key, main_n, main_k):
    """``GAMES`` lists only games supported end to end."""
    definition = lottobench.game(key)
    assert definition.spec.main_n == main_n
    assert definition.spec.main_k == main_k
    assert definition.source_url.startswith("https://")


@pytest.mark.parametrize(
    "key", ["dk-lotto", "de-lotto-6aus49", "uk-lotto", "se-lotto"]
)
def test_backlog_games_are_not_advertised_as_supported(key):
    """Regression: the catalogue used to return these, implying a journey that does not exist.

    They have no retrieval adapter, so ``lottobench fetch`` cannot serve them. Listing them as
    supported is what made the published package journey misleading.
    """
    from lottobench.games import BACKLOG_GAMES

    assert key not in lottobench.GAMES
    assert key in BACKLOG_GAMES
    assert BACKLOG_GAMES[key].blocked_on
    with pytest.raises(KeyError, match="not supported end to end"):
        lottobench.game(key)


def test_public_strategy_api_is_functional():
    assert "frequency" in lottobench.names()
    assert lottobench.create("frequency").name == "frequency"


def test_cli_lists_versioned_providers(capsys):
    from lottobench.cli import main

    assert main(["providers"]) == 0
    output = capsys.readouterr().out
    assert "frequency" in output
    assert "1.0.0" in output
    assert "12 selectable entrants backed by 9 implementation families" in output
    assert "spectral_contrarian" in output
    assert "garch_markov_branch" in output
    assert "sequence_transformer" in output
    assert "uniform_random" in output
    assert "gingerm" in output
    assert lottobench.names().count("sequence_transformer") == 1


def test_optional_providers_name_their_install_extras():
    from lotteries_core import registry

    assert registry.PROVIDERS["sequence_transformer"].install_extra == "transformer"
    assert registry.PROVIDERS["ml_ensemble"].install_extra == "ml"
