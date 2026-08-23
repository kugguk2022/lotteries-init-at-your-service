from __future__ import annotations

import pytest

import lottobench


@pytest.mark.parametrize(
    ("key", "main_n", "main_k"),
    [
        ("dk-lotto", 36, 7),
        ("de-lotto-6aus49", 49, 6),
        ("uk-lotto", 59, 6),
        ("nl-lotto", 45, 6),
        ("se-lotto", 35, 7),
    ],
)
def test_country_game_catalogue(key, main_n, main_k):
    definition = lottobench.game(key)
    assert definition.spec.main_n == main_n
    assert definition.spec.main_k == main_k
    assert definition.source_url.startswith("https://")


def test_public_strategy_api_is_functional():
    assert "frequency" in lottobench.names()
    assert lottobench.create("frequency").name == "frequency"


def test_cli_lists_versioned_providers(capsys):
    from lottobench.cli import main

    assert main(["providers"]) == 0
    output = capsys.readouterr().out
    assert "frequency" in output
    assert "1.0.0" in output
