"""Retrieval adapters that turn a published draw archive into canonical history rows.

This package is *shipped*. Retrieval used to live in ``experiments/``, which meant the documented
user journey -- install, fetch, benchmark -- could not actually be completed from the published
wheel. A benchmark whose data step is unavailable to its users is not a benchmark.

Dependency contract: the CSV sources here run on the base install (numpy + pandas only). The two
HTML archive fallbacks need ``beautifulsoup4`` and ``requests``, so they live in
:mod:`lotteries_core.sources.html_archive` and are imported lazily. A user who only ever fetches
CSV never pays for a scraper.

Only EuroMillions is wired end to end. See ``docs/wiki/Backlog.md`` for the games that are defined
but not yet retrievable.
"""

from __future__ import annotations

from .errors import ContentTypeError, FetchError, NormalizationError
from .euromillions import (
    SOURCE_CHOICES,
    canonicalize_columns,
    fetch_euromillions,
    fetch_raw_csv,
    normalize,
)
from .netherlands import fetch_netherlands
from .schema import CANONICAL_COLUMNS, validate_frame

__all__ = [
    "CANONICAL_COLUMNS",
    "ContentTypeError",
    "FetchError",
    "NormalizationError",
    "SOURCE_CHOICES",
    "canonicalize_columns",
    "fetch_euromillions",
    "fetch_netherlands",
    "fetch_raw_csv",
    "normalize",
    "validate_frame",
]
