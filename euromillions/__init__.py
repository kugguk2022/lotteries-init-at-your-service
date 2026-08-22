"""EuroMillions lab: draw fetching, schema validation, scoring, and baseline generation.

This module defines the package's public surface. Everything listed in ``__all__`` is covered by
tests and safe to import directly::

    from euromillions import EuroMillionsGuess, evaluate_guess, normalize

Anything else in the package is lab code with an unstable interface -- import it by module path
(``from euromillions.arithmetic_branch import ...``) so the dependency is explicit.

The stable framework surface for building new strategies is :mod:`lotteries_core`, not this package.

Exports resolve lazily (PEP 562). Importing the submodules eagerly here would mean that
``python -m euromillions.get_draws`` imports ``get_draws`` once as a package attribute and again as
``__main__``, which emits a RuntimeWarning and can execute module-level code twice.
"""

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

if TYPE_CHECKING:  # import-time types for editors and mypy only
    from .get_draws import normalize
    from .guess import EuroMillionsGuess, evaluate_guess
    from .infer import generate_candidates, probability_tables, random_candidates
    from .schema import validate_df


def __getattr__(name: str):
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path, __name__), name)
    globals()[name] = value  # cache so later lookups skip this hook
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_EXPORTS])
