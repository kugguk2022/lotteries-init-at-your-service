"""A model of how *other players* choose numbers -- the basis of the only valid draw-game lever.

Why this exists
---------------
For a fair draw, every legal ticket has identical probability of winning. You cannot change that.
But most jackpots are **pari-mutuel**: the prize is *split* among all winners. So while you can't
raise your probability of winning, you *can* raise your **expected payout conditional on winning**
by picking combinations that few other people pick. The lever is entirely on the *human* side of
the game, not the random side -- which is exactly why it is legitimate and non-magical.

To exploit it you need an estimate of the crowd's ticket-selection distribution. Nobody publishes
that distribution, but decades of published research on lottery number choice give robust, stable
biases we can encode as a prior. This module builds a *relative popularity weight* for any ticket
from those biases; :mod:`lotteries_core.roi` turns weights into expected co-winners and ROI.

Documented human biases encoded here (all raise a combination's popularity):

* **Calendar bias** -- heavy over-selection of numbers 1..31 (birthdays) and especially 1..12.
* **Low-number bias** -- small numbers are picked more than large ones.
* **Lucky numbers** -- 7 (and multiples), and culturally lucky digits, are over-picked.
* **Round/patterned tickets** -- arithmetic sequences, straight lines on the play-slip,
  evenly spaced "spread" patterns, and all-same-decade tickets.
* **Recency** -- recently drawn combinations get copied.

None of these say anything about which numbers will be *drawn*. They only describe what humans
*pick*. Feeding real data (e.g. aggregated, anonymised winner-location / sales data -- see
``docs/GEOGRAPHY.md``) can *calibrate* these weights, but the data must be normalised for sales
volume and population first, or it just re-measures where tickets are sold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .protocol import GameSpec, Ticket


@dataclass(frozen=True)
class PopularityModel:
    """Relative popularity weights for individual numbers and simple combination patterns.

    All weights are *multiplicative* factors on a uniform baseline of 1.0. A ticket's popularity
    is the product of its number weights times pattern multipliers. Only *ratios* matter, so the
    absolute scale is irrelevant; :func:`ticket_popularity` returns an unnormalised score and the
    ROI layer normalises across the ticket set it is comparing.
    """

    calendar_weight: float = 2.6   # numbers 1..31 over-picked (birthdays)
    month_weight: float = 1.4      # extra bump for 1..12 (months)
    lucky7_weight: float = 1.5     # 7 and its multiples
    low_number_decay: float = 0.985  # each number n gets low_number_decay**(n-1) taper
    consecutive_pattern: float = 1.6  # runs of consecutive numbers (players love sequences)
    arithmetic_pattern: float = 1.8   # equal-spacing arithmetic tickets (1,2,3,4,5 etc.)
    same_decade_pattern: float = 1.3  # all numbers in one 1..10 / 11..20 band

    def number_weights(self, spec: GameSpec) -> np.ndarray:
        """Per-number relative popularity for the main pool, length ``spec.main_n``."""
        n = spec.main_n
        w = np.ones(n, dtype=float)
        for i in range(n):
            num = i + 1
            if num <= 31:
                w[i] *= self.calendar_weight
            if num <= 12:
                w[i] *= self.month_weight
            if num == 7 or (num % 7 == 0):
                w[i] *= self.lucky7_weight
            w[i] *= self.low_number_decay ** (num - 1)
        return w

    def _pattern_multiplier(self, main: tuple[int, ...]) -> float:
        arr = np.asarray(sorted(main), dtype=int)
        mult = 1.0
        diffs = np.diff(arr)
        if len(diffs) and np.all(diffs == 1):
            mult *= self.consecutive_pattern
        elif len(diffs) and np.all(diffs == diffs[0]) and diffs[0] > 1:
            mult *= self.arithmetic_pattern
        # all in one decade band
        if (arr.max() - arr.min()) <= 9:
            mult *= self.same_decade_pattern
        return mult

    def ticket_popularity(self, spec: GameSpec, ticket: Ticket) -> float:
        """Unnormalised relative popularity of a single ticket (higher = more crowded)."""
        main, _star = ticket
        nw = self.number_weights(spec)
        base = float(np.prod([nw[v - 1] for v in main]))
        return base * self._pattern_multiplier(main)

    def popularity_vector(self, spec: GameSpec, tickets: list[Ticket]) -> np.ndarray:
        """Popularity of each ticket, normalised to a probability distribution over the set.

        This is a *relative share of the crowd among these tickets only*. Use it when you genuinely
        want within-set proportions; for expected-ROI math use :meth:`absolute_shares`, which is
        anchored to the whole game and therefore does not change meaning with portfolio composition.
        """
        raw = np.array([self.ticket_popularity(spec, t) for t in tickets], dtype=float)
        total = raw.sum()
        if total <= 0:
            return np.full(len(tickets), 1.0 / max(len(tickets), 1))
        return raw / total

    def reference_mean_popularity(
        self, spec: GameSpec, n_samples: int = 20000, seed: int = 0
    ) -> float:
        """Mean raw popularity of a *uniformly random* legal ticket (the absolute baseline).

        Computed by vectorised Monte Carlo with a fixed seed, so it is deterministic. This is the
        divisor that turns a ticket's raw popularity into an *absolute crowd share* where 1.0 means
        "as crowded as an average ticket", <1 means under-picked, >1 means over-picked.
        """
        rng = np.random.default_rng(seed)
        nw = self.number_weights(spec)
        idx = np.argsort(rng.random((n_samples, spec.main_n)), axis=1)[:, : spec.main_k]
        idx = np.sort(idx, axis=1)  # 0-based sorted number indices
        prod = np.prod(nw[idx], axis=1)
        nums = idx + 1
        diffs = np.diff(nums, axis=1)
        consec = np.all(diffs == 1, axis=1)
        arith = np.all(diffs == diffs[:, :1], axis=1) & (diffs[:, 0] > 1)
        same_decade = (nums.max(axis=1) - nums.min(axis=1)) <= 9
        mult = np.ones(n_samples, dtype=float)
        mult[consec] *= self.consecutive_pattern
        mult[arith & ~consec] *= self.arithmetic_pattern
        mult[same_decade] *= self.same_decade_pattern
        return float(np.mean(prod * mult))

    def absolute_shares(
        self,
        spec: GameSpec,
        tickets: list[Ticket],
        reference: float | None = None,
        *,
        n_samples: int = 20000,
        seed: int = 0,
    ) -> np.ndarray:
        """Each ticket's crowd share relative to an average ticket (1.0 = average crowding).

        This is the quantity the ROI layer needs: expected co-winners scale linearly with it, and it
        is invariant to which other tickets happen to be in the portfolio.
        """
        if reference is None:
            reference = self.reference_mean_popularity(spec, n_samples=n_samples, seed=seed)
        if reference <= 0:
            return np.ones(len(tickets), dtype=float)
        raw = np.array([self.ticket_popularity(spec, t) for t in tickets], dtype=float)
        return raw / reference

    def calibrate_from_counts(self, spec: GameSpec, observed_counts: np.ndarray) -> "PopularityModel":
        """Return a copy whose per-number behaviour is informed by real, normalised pick counts.

        ``observed_counts`` must already be normalised for sales volume / population (see
        ``docs/GEOGRAPHY.md``). This is a placeholder hook: it validates shape and, for now,
        leaves the structural priors intact so behaviour stays deterministic until a real,
        vetted dataset is wired in. Kept explicit so the data contract is visible in code.
        """
        observed_counts = np.asarray(observed_counts, dtype=float)
        if observed_counts.shape[0] != spec.main_n:
            raise ValueError(f"expected {spec.main_n} counts, got {observed_counts.shape[0]}")
        return self
