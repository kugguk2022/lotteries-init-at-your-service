"""Concrete inference providers.

Every provider implements :class:`lotteries_core.protocol.InferenceProvider`. Import them lazily
where possible so optional heavy dependencies (xgboost, torch) never break a base install.
"""

from __future__ import annotations

from ..likely_set_generator import CooccurrenceLevelSetProvider
from .frequency import FrequencyProvider
from .spectral import PerronFrobeniusProvider, null_tv_band, stationary_distribution
from .unpopularity import UnpopularityProvider

__all__ = [
    "CooccurrenceLevelSetProvider",
    "FrequencyProvider",
    "PerronFrobeniusProvider",
    "UnpopularityProvider",
    "load_ml_ensemble",
    "null_tv_band",
    "stationary_distribution",
]


def load_ml_ensemble():
    """Return the ``MLEnsembleProvider`` class, importing it lazily (soft optional deps)."""
    from .ml_ensemble import MLEnsembleProvider

    return MLEnsembleProvider
