"""Smoke test executed from outside the repository against an installed wheel."""

from __future__ import annotations

import subprocess
import sys

import lotteries_core
import lottobench

assert lotteries_core.__version__ == "0.1.0a1"
assert lottobench.game("uk-lotto").spec.main_n == 59
assert "frequency" in lottobench.names()
assert lottobench.create("frequency").name == "frequency"
assert callable(lottobench.compare_realized_roi)

result = subprocess.run(
    [sys.executable, "-m", "lottobench.cli", "games"],
    check=True,
    capture_output=True,
    text=True,
)
assert "uk-lotto" in result.stdout
assert "se-lotto" in result.stdout
providers = subprocess.run(
    [sys.executable, "-m", "lottobench.cli", "providers"],
    check=True,
    capture_output=True,
    text=True,
)
assert "frequency" in providers.stdout
assert "uniform_random" in providers.stdout
assert "garch_markov_branch" in providers.stdout
assert "sequence_transformer" in providers.stdout
assert "12 selectable entrants backed by 9 implementation families" in providers.stdout
assert "install [transformer] extra" in providers.stdout
assert "1.0.0" in providers.stdout
subprocess.run(
    [sys.executable, "-m", "lotteries_core.realized_roi", "--help"],
    check=True,
    capture_output=True,
    text=True,
)
print("installed LottoBench wheel smoke test passed")
