"""The single registry of selectable providers.

Provider names were previously spelled out in three places -- the benchmark CLI's ``--with-*`` flags,
the outcome tracker's ``_make_provider``, and (now) the HTTP API. Three lists drift. This is the one
list; everything else asks it.

Adding a strategy means one entry here plus the provider module itself. See
``docs/wiki/Contributing-a-Provider.md``.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Callable

from .protocol import InferenceProvider


@dataclass(frozen=True)
class ProviderSpec:
    """A selectable strategy: how to build it, and what it claims."""

    name: str
    summary: str
    #: Providers that pair with an ablation naming it here makes the control discoverable, which is
    #: the whole point of shipping one (see the contributing guide).
    ablation_of: str | None = None
    #: Optional third-party dependencies. Absent ones make the provider unavailable, never fatal.
    optional: bool = False

    def build(self) -> InferenceProvider:
        return _FACTORIES[self.name]()


def _frequency() -> InferenceProvider:
    from .providers import FrequencyProvider

    return FrequencyProvider()


def _unpopularity() -> InferenceProvider:
    from .providers import UnpopularityProvider

    return UnpopularityProvider()


def _cooccurrence() -> InferenceProvider:
    from .providers import CooccurrenceLevelSetProvider

    return CooccurrenceLevelSetProvider()


def _named(provider: InferenceProvider, name: str) -> InferenceProvider:
    """Give a public entrant name to an existing implementation."""
    provider.name = name
    return provider


def _gingerm() -> InferenceProvider:
    return _named(_cooccurrence(), "gingerm")


def _claude_inference() -> InferenceProvider:
    return _named(_perron("contrarian")(), "claude_inference")


def _public_parallax() -> InferenceProvider:
    return _named(_parallax("guarded")(), "parallax")


def _perron(orientation: str) -> Callable[[], InferenceProvider]:
    def make() -> InferenceProvider:
        from .providers import PerronFrobeniusProvider

        return PerronFrobeniusProvider(orientation=orientation)

    return make


def _parallax(mode: str) -> Callable[[], InferenceProvider]:
    def make() -> InferenceProvider:
        from .providers import ParallaxGuardProvider

        return ParallaxGuardProvider(mode=mode)

    return make


def _ml_ensemble() -> InferenceProvider:
    if importlib.util.find_spec("sklearn") is None:
        raise ImportError("ml_ensemble requires the 'ml' extra: pip install 'lotteries-core[ml]'")
    from .providers import load_ml_ensemble

    return load_ml_ensemble()()


_FACTORIES: dict[str, Callable[[], InferenceProvider]] = {
    "gingerm": _gingerm,
    "claude_inference": _claude_inference,
    "parallax": _public_parallax,
    "frequency": _frequency,
    "unpopularity": _unpopularity,
    "cooccurrence_level_set": _cooccurrence,
    "perron_frobenius_affinity": _perron("affinity"),
    "perron_frobenius_contrarian": _perron("contrarian"),
    "perron_frobenius_uniform": _perron("uniform"),
    "parallax_guard": _parallax("guarded"),
    "parallax_guard_ablation": _parallax("ablation"),
    "ml_ensemble": _ml_ensemble,
}

PROVIDERS: dict[str, ProviderSpec] = {
    "gingerm": ProviderSpec(
        "gingerm",
        "GINGERM: the owner's forward-only pair-co-occurrence level-set strategy.",
    ),
    "claude_inference": ProviderSpec(
        "claude_inference",
        "Claude inference: contrarian Perron-Frobenius ranking of the co-occurrence graph.",
    ),
    "parallax": ProviderSpec(
        "parallax",
        "Parallax: replication-guarded residual inference with coverage-first portfolio selection.",
    ),
    "frequency": ProviderSpec(
        "frequency",
        "Smoothed historical-frequency weighted sampling. No predictive edge on a fair draw; the "
        "reference baseline every other strategy must beat.",
    ),
    "unpopularity": ProviderSpec(
        "unpopularity",
        "Prefers combinations the crowd avoids. Does not change the odds of winning; improves the "
        "expected payout conditional on winning, because fewer people share the jackpot.",
    ),
    "cooccurrence_level_set": ProviderSpec(
        "cooccurrence_level_set",
        "Forward-only pair-co-occurrence level-set generator. Slow: enumerates every main "
        "combination against every star combination.",
    ),
    "perron_frobenius_affinity": ProviderSpec(
        "perron_frobenius_affinity",
        "PageRank stationary ranking of the co-occurrence graph, preferring high-rank numbers.",
    ),
    "perron_frobenius_contrarian": ProviderSpec(
        "perron_frobenius_contrarian",
        "The same stationary ranking, preferring low-rank numbers.",
    ),
    "perron_frobenius_uniform": ProviderSpec(
        "perron_frobenius_uniform",
        "Ablation control: the identical sampler with the stationary vector discarded.",
        ablation_of="perron_frobenius_affinity",
    ),
    "parallax_guard": ProviderSpec(
        "parallax_guard",
        "Residuals admitted only when they replicate across two disjoint history views and clear a "
        "family-wise threshold, plus a coverage-first portfolio optimiser.",
    ),
    "parallax_guard_ablation": ProviderSpec(
        "parallax_guard_ablation",
        "Ablation control: identical candidate pool and portfolio objective with the residual "
        "forced to zero.",
        ablation_of="parallax_guard",
    ),
    "ml_ensemble": ProviderSpec(
        "ml_ensemble",
        "GLM + gradient boosting (+ optional MLP) aimed at crowd popularity, never at the draw. "
        "Requires scikit-learn; xgboost and torch are used when present.",
        optional=True,
    ),
}


def names() -> list[str]:
    """Every registered provider name, in a stable order."""
    return list(PROVIDERS)


def available() -> list[str]:
    """Names that can actually be instantiated here, skipping any with missing optional deps."""
    out = []
    for name, spec in PROVIDERS.items():
        if not spec.optional:
            out.append(name)
            continue
        try:
            spec.build()
        except Exception:  # optional dependency missing or unusable -- not an error
            continue
        out.append(name)
    return out


def create(name: str) -> InferenceProvider:
    """Instantiate a registered provider by name."""
    if name not in PROVIDERS:
        raise KeyError(f"unknown provider {name!r}; available: {names()}")
    return PROVIDERS[name].build()
