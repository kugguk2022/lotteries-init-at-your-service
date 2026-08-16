"""Concrete inference providers.

Every provider implements :class:`lotteries_core.protocol.InferenceProvider`. Import them lazily
where possible so optional heavy dependencies (xgboost, torch) never break a base install.
"""

from __future__ import annotations

from .frequency import FrequencyProvider
from .unpopularity import UnpopularityProvider

__all__ = ["FrequencyProvider", "UnpopularityProvider", "load_ml_ensemble"]


def load_ml_ensemble():
    """Return the ``MLEnsembleProvider`` class, importing it lazily (soft optional deps)."""
    from .ml_ensemble import MLEnsembleProvider

    return MLEnsembleProvider
