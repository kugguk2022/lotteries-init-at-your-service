"""The common inference protocol shared by every strategy ("provider").

A *provider* is any strategy that, given a lottery's combinatorial shape (:class:`GameSpec`)
and a fixed **ticket budget**, proposes that many candidate tickets. Providers are the units
that a distributed run coordinates: each node runs one or more providers, writes its proposals
to a reproducible :class:`~lotteries_core.envelope.InferenceEnvelope`, and a coordinator
aggregates the envelopes under a shared budget (see :mod:`lotteries_core.aggregation`).

Nothing in this protocol assumes a provider can predict the draw. A provider only expresses a
*preference ordering over tickets*; the framework then evaluates those preferences forward-only
and on coverage / expected-return-per-ticket, never on an assumption of predictive power.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from math import comb
from typing import Sequence

import numpy as np

# A ticket is (main numbers, bonus/star numbers). Both are sorted tuples of 1-based ints.
Ticket = tuple[tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True)
class GameSpec:
    """The combinatorial shape of a lottery game.

    Attributes
    ----------
    name:
        Human-readable identifier, e.g. ``"euromillions"``.
    main_n, main_k:
        Draw ``main_k`` distinct numbers from ``1..main_n`` (EuroMillions: 5 from 50).
    star_n, star_k:
        Draw ``star_k`` distinct bonus numbers from ``1..star_n`` (EuroMillions: 2 from 12).
        Set ``star_k = 0`` for games without a bonus pool.
    """

    name: str
    main_n: int
    main_k: int
    star_n: int = 0
    star_k: int = 0

    # Common presets are provided as classmethods for convenience/reproducibility.
    @classmethod
    def euromillions(cls) -> "GameSpec":
        return cls("euromillions", main_n=50, main_k=5, star_n=12, star_k=2)

    @classmethod
    def totoloto(cls) -> "GameSpec":
        # Portuguese Totoloto: 5 from 49 + 1 lucky number from 13.
        return cls("totoloto", main_n=49, main_k=5, star_n=13, star_k=1)

    @classmethod
    def eurodreams(cls) -> "GameSpec":
        # EuroDreams: 6 from 40 + 1 "dream" number from 5.
        return cls("eurodreams", main_n=40, main_k=6, star_n=5, star_k=1)

    def n_main_combinations(self) -> int:
        return comb(self.main_n, self.main_k)

    def n_star_combinations(self) -> int:
        return comb(self.star_n, self.star_k) if self.star_k > 0 else 1

    def n_tickets(self) -> int:
        """Total number of distinct tickets in the game (the combinatorial universe size)."""
        return self.n_main_combinations() * self.n_star_combinations()

    def validate_ticket(self, ticket: Ticket) -> None:
        """Raise ``ValueError`` if ``ticket`` is not a legal ticket for this game."""
        main, star = ticket
        if len(main) != self.main_k or len(set(main)) != self.main_k:
            raise ValueError(f"main part must be {self.main_k} distinct numbers, got {main!r}")
        if any(not (1 <= v <= self.main_n) for v in main):
            raise ValueError(f"main numbers must be in 1..{self.main_n}, got {main!r}")
        if tuple(sorted(main)) != tuple(main):
            raise ValueError(f"main numbers must be sorted ascending, got {main!r}")
        if self.star_k > 0:
            if len(star) != self.star_k or len(set(star)) != self.star_k:
                raise ValueError(f"star part must be {self.star_k} distinct numbers, got {star!r}")
            if any(not (1 <= v <= self.star_n) for v in star):
                raise ValueError(f"star numbers must be in 1..{self.star_n}, got {star!r}")
            if tuple(sorted(star)) != tuple(star):
                raise ValueError(f"star numbers must be sorted ascending, got {star!r}")
        elif star:
            raise ValueError(f"game has no star pool but ticket has stars: {star!r}")


@dataclass
class ProviderResult:
    """What a provider returns: an ordered list of tickets plus an optional per-ticket score.

    ``tickets[i]`` has preference weight ``scores[i]`` (higher = more preferred by the provider).
    Scores are provider-internal and are *not* comparable across providers until normalized by
    the aggregator. ``diagnostics`` is a free-form JSON-serialisable dict for provenance.
    """

    tickets: list[Ticket]
    scores: np.ndarray
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.scores = np.asarray(self.scores, dtype=float)
        if self.scores.shape[0] != len(self.tickets):
            raise ValueError("scores length must match number of tickets")


class InferenceProvider(abc.ABC):
    """Interface every strategy implements.

    Lifecycle: ``fit(history)`` once, then ``propose(spec, budget, rng)`` any number of times.
    Implementations MUST be deterministic given the same ``history`` and the same seeded ``rng``;
    this is what makes envelopes reproducible.
    """

    #: Stable, unique, human-readable provider name (used in envelope provenance).
    name: str = "base"

    #: Short description of the mechanism and its operating scope.
    description: str = ""

    def fit(self, history) -> "InferenceProvider":  # noqa: ANN001 - history is a DataFrame-like
        """Fit any internal state from the training history. Default: no-op. Returns self."""
        return self

    @abc.abstractmethod
    def propose(
        self,
        spec: GameSpec,
        budget: int,
        rng: np.random.Generator,
    ) -> ProviderResult:
        """Propose ``budget`` distinct legal tickets for ``spec`` using the seeded ``rng``."""
        raise NotImplementedError

    # ---- helpers available to all providers -------------------------------------------------
    @staticmethod
    def _sample_distinct_tickets(
        spec: GameSpec,
        budget: int,
        rng: np.random.Generator,
        main_probs: np.ndarray | None = None,
        star_probs: np.ndarray | None = None,
        max_tries_factor: int = 50,
    ) -> list[Ticket]:
        """Sample ``budget`` distinct legal tickets, optionally weighting numbers by probability.

        Draws numbers *without replacement within a ticket* and rejects duplicate tickets across
        the set. Falls back gracefully if the requested budget exceeds the universe size.
        """
        universe = spec.n_tickets()
        budget = min(budget, universe)
        seen: set[Ticket] = set()
        out: list[Ticket] = []
        main_pop = np.arange(1, spec.main_n + 1)
        star_pop = np.arange(1, spec.star_n + 1) if spec.star_k > 0 else np.array([], dtype=int)
        mp = _normalize(main_probs, spec.main_n)
        sp = _normalize(star_probs, spec.star_n) if spec.star_k > 0 else None
        tries = 0
        max_tries = max(budget * max_tries_factor, budget + 100)
        while len(out) < budget and tries < max_tries:
            tries += 1
            main = tuple(sorted(int(x) for x in rng.choice(main_pop, spec.main_k, replace=False, p=mp)))
            if spec.star_k > 0:
                star = tuple(sorted(int(x) for x in rng.choice(star_pop, spec.star_k, replace=False, p=sp)))
            else:
                star = ()
            ticket: Ticket = (main, star)
            if ticket in seen:
                continue
            seen.add(ticket)
            out.append(ticket)
        return out


def _normalize(probs: np.ndarray | None, size: int) -> np.ndarray | None:
    if probs is None:
        return None
    probs = np.asarray(probs, dtype=float)
    if probs.shape[0] != size:
        raise ValueError(f"probability length {probs.shape[0]} != population size {size}")
    if np.any(probs < 0):
        raise ValueError("probabilities must be non-negative")
    total = probs.sum()
    if total <= 0:
        return None
    return probs / total


def tickets_equal(a: Sequence[Ticket], b: Sequence[Ticket]) -> bool:
    """Order-insensitive equality of two ticket collections (used in tests/fixtures)."""
    return sorted(a) == sorted(b)
