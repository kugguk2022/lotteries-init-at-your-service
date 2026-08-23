"""EuroMillions lab public surface, resolved lazily to keep module CLIs safe."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_EXPORTS = {
    "EuroMillionsGuess": ".guess",
    "evaluate_guess": ".guess",
    "normalize": ".get_draws",
    "generate_candidates": ".infer",
    "probability_tables": ".infer",
    "random_candidates": ".infer",
    "validate_df": ".schema",
}

__all__ = sorted(_EXPORTS)

if TYPE_CHECKING:
    from .get_draws import normalize as normalize
    from .guess import EuroMillionsGuess as EuroMillionsGuess
    from .guess import evaluate_guess as evaluate_guess
    from .infer import generate_candidates as generate_candidates
    from .infer import probability_tables as probability_tables
    from .infer import random_candidates as random_candidates
    from .schema import validate_df as validate_df


def __getattr__(name: str):
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_EXPORTS])
