"""Concrete inference providers.

Every provider implements :class:`lotteries_core.protocol.InferenceProvider`. Import them lazily
where possible so optional heavy dependencies (xgboost, torch) never break a base install.
"""

from __future__ import annotations

from ..likely_set_generator import CooccurrenceLevelSetProvider
from .frequency import FrequencyProvider
from .parallax import ParallaxGuardProvider, replicated_evidence
from .spectral import PerronFrobeniusProvider, null_tv_band, stationary_distribution
from .uniform import UniformRandomProvider
from .unpopularity import UnpopularityProvider

__all__ = [
    "CooccurrenceLevelSetProvider",
    "FrequencyProvider",
    "ParallaxGuardProvider",
    "PerronFrobeniusProvider",
    "GarchMarkovBranchProvider",
    "SequenceTransformerProvider",
    "UnpopularityProvider",
    "UniformRandomProvider",
    "load_ml_ensemble",
    "null_tv_band",
    "replicated_evidence",
    "stationary_distribution",
]


def load_ml_ensemble():
    """Return the ``MLEnsembleProvider`` class, importing it lazily (soft optional deps)."""
    from .ml_ensemble import MLEnsembleProvider

    return MLEnsembleProvider


def load_temporal_providers():
    """Return temporal provider classes while keeping PyTorch an optional dependency."""
    from .temporal import GarchMarkovBranchProvider, SequenceTransformerProvider

    return GarchMarkovBranchProvider, SequenceTransformerProvider
