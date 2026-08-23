"""The single registry of selectable providers.

Provider names were previously spelled out in three places -- the benchmark CLI's ``--with-*`` flags,
the outcome tracker's ``_make_provider``, and (now) the HTTP API. Three lists drift. This is the one
list; everything else asks it.

Adding a strategy means one entry here plus the provider module itself. See
``docs/wiki/Contributing-a-Provider.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .protocol import InferenceProvider


@dataclass(frozen=True)
class ProviderSpec:
    """A selectable strategy: how to build it, and what it claims."""

    name: str
    summary: str
    #: Algorithm family backing this selectable identity. Several public names and ablation modes
    #: intentionally share an implementation; exposing that relationship avoids presenting them as
    #: independent models.
    implementation: str
    #: Providers that pair with an ablation naming it here makes the control discoverable, which is
    #: the whole point of shipping one (see the contributing guide).
    ablation_of: str | None = None
    #: Optional third-party dependencies. Absent ones make the provider unavailable, never fatal.
    optional: bool = False
    #: Bump whenever an algorithm or default affecting generated portfolios changes.
    version: str = "1.0.0"

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


def _spectral_contrarian() -> InferenceProvider:
    return _named(_perron("contrarian")(), "spectral_contrarian")


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
    from .providers import load_ml_ensemble

    return load_ml_ensemble()()


def _garch_markov_branch() -> InferenceProvider:
    from .providers import load_temporal_providers

    provider, _ = load_temporal_providers()
    return provider()


def _sequence_transformer() -> InferenceProvider:
    from .providers import load_temporal_providers

    _, provider = load_temporal_providers()
    return provider()


_FACTORIES: dict[str, Callable[[], InferenceProvider]] = {
    "gingerm": _gingerm,
    "spectral_contrarian": _spectral_contrarian,
    "parallax": _public_parallax,
    "frequency": _frequency,
    "unpopularity": _unpopularity,
    "perron_frobenius_affinity": _perron("affinity"),
    "perron_frobenius_uniform": _perron("uniform"),
    "parallax_guard_ablation": _parallax("ablation"),
    "ml_ensemble": _ml_ensemble,
    "garch_markov_branch": _garch_markov_branch,
    "sequence_transformer": _sequence_transformer,
}

PROVIDERS: dict[str, ProviderSpec] = {
    "gingerm": ProviderSpec(
        "gingerm",
        "GINGERM: the owner's forward-only pair-co-occurrence level-set strategy.",
        "cooccurrence_level_set",
    ),
    "spectral_contrarian": ProviderSpec(
        "spectral_contrarian",
        "Vendor-neutral contrarian Perron-Frobenius ranking of the co-occurrence graph.",
        "perron_frobenius",
    ),
    "parallax": ProviderSpec(
        "parallax",
        "Parallax: replication-guarded residual inference with coverage-first portfolio selection.",
        "parallax_guard",
    ),
    "frequency": ProviderSpec(
        "frequency",
        "Smoothed historical-frequency weighted sampling. No predictive edge on a fair draw; the "
        "reference baseline every other strategy must beat.",
        "frequency",
    ),
    "unpopularity": ProviderSpec(
        "unpopularity",
        "Prefers combinations the crowd avoids. Does not change the odds of winning; improves the "
        "expected payout conditional on winning, because fewer people share the jackpot.",
        "unpopularity",
    ),
    "perron_frobenius_affinity": ProviderSpec(
        "perron_frobenius_affinity",
        "PageRank stationary ranking of the co-occurrence graph, preferring high-rank numbers.",
        "perron_frobenius",
    ),
    "perron_frobenius_uniform": ProviderSpec(
        "perron_frobenius_uniform",
        "Ablation control: the identical sampler with the stationary vector discarded.",
        "perron_frobenius",
        ablation_of="perron_frobenius_affinity",
    ),
    "parallax_guard_ablation": ProviderSpec(
        "parallax_guard_ablation",
        "Ablation control: identical candidate pool and portfolio objective with the residual "
        "forced to zero.",
        "parallax_guard",
        ablation_of="parallax",
    ),
    "ml_ensemble": ProviderSpec(
        "ml_ensemble",
        "GLM + gradient boosting (+ optional MLP) aimed at crowd popularity, never at the draw. "
        "Requires scikit-learn; xgboost and torch are used when present.",
        "ml_ensemble",
        optional=True,
    ),
    "garch_markov_branch": ProviderSpec(
        "garch_markov_branch",
        "GARCH(1,1) co-occurrence-score variance with a two-state empirical Markov branch model.",
        "garch_markov_branch",
    ),
    "sequence_transformer": ProviderSpec(
        "sequence_transformer",
        "Causal Transformer forecast of the co-occurrence-score sequence; requires PyTorch.",
        "sequence_transformer",
        optional=True,
    ),
}


def names() -> list[str]:
    """Every registered provider name, in a stable order."""
    return list(PROVIDERS)


def version(name: str) -> str:
    """Stable algorithm version used to separate realized-ROI benchmark cohorts."""
    if name not in PROVIDERS:
        raise KeyError(f"unknown provider {name!r}; available: {names()}")
    return PROVIDERS[name].version


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
