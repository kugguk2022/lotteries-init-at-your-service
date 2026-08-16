"""lotteries_core: a research framework for coordinated, distributed inference over lottery draws.

This package is the common substrate for the repository's pivot (see ``repurpose.md``)
from a collection of one-off lottery-analysis experiments toward a reproducible framework
for studying whether *coordinating many independent inference strategies* can improve
**combinatorial coverage** and **expected return-per-ticket** under a **fixed ticket budget**.

Read ``docs/SCOPE_AND_ETHICS.md`` before using anything here. The short version:

* Fair number-draw lotteries (EuroMillions, Totoloto, EuroDreams) are *provably random*.
  No model in this package can predict which numbers will be drawn, and none tries to.
* The lottery is a **negative-sum** game. Nothing here produces a guaranteed positive ROI,
  and the framework never claims otherwise.
* The *only* mathematically valid levers this framework studies are:
    1. **Unpopularity / jackpot-sharing** -- for a pari-mutuel (shared) jackpot, choosing
       number combinations that few other players pick raises the *expected payout
       conditional on winning*. This does not change the probability of winning.
    2. **Instant-game remaining-prize EV** -- for finite-pool scratch/instant games with
       officially published remaining-prize inventories, expected value is computable and
       occasionally favourable. This is the honest core of the "Joan Ginther" advantage-play
       story (see ``docs/GEOGRAPHY.md``).
* The framework is **simulation-only**. It never moves money, pools funds, buys tickets,
  or executes wagers.

Public surface:

* :class:`~lotteries_core.protocol.GameSpec` -- describes a lottery's combinatorial shape.
* :class:`~lotteries_core.protocol.InferenceProvider` -- the interface every strategy implements.
* :class:`~lotteries_core.envelope.InferenceEnvelope` -- a reproducible, provenance-carrying
  container for a provider's proposed tickets (the unit of file-based exchange between nodes).
* :mod:`~lotteries_core.aggregation` -- diversity-aware, equal-budget aggregation across providers.
* :mod:`~lotteries_core.evaluation` -- forward-only, equal-budget evaluation.
"""

from __future__ import annotations

from .envelope import InferenceEnvelope
from .protocol import GameSpec, InferenceProvider, Ticket

__all__ = [
    "GameSpec",
    "InferenceProvider",
    "Ticket",
    "InferenceEnvelope",
    "__version__",
]

__version__ = "0.2.0.dev0"
