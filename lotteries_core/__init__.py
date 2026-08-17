"""lotteries_core: a research framework for coordinated, distributed inference over lottery draws.

This package is the common substrate for the repository's pivot (see ``repurpose.md``)
from a collection of one-off lottery-analysis experiments toward a reproducible, prospective
experiment: record fixed-budget portfolios before each draw, settle them against official results,
and test whether the owner's co-occurrence method or coordinated providers outperform a matched
random control over a long horizon.

Read ``docs/SCOPE.md`` before using anything here. The short version:

* Licensed number draws are designed to be independent and fixed-odds; that is the null hypothesis.
  The package nevertheless tests draw-prediction methods prospectively rather than declaring an
  edge—or impossibility—without outcome evidence.
* The baseline game is negative-sum for users. The research objective is to test whether coordinated
  selection can produce a repeatable predictive edge and positive net user outcome. It makes no
  guaranteed or pre-evidence positive-ROI claim.
* The primary question is outcome-based: do preregistered portfolios beat an equal-budget random
  control on holdout hit tiers and tracked payout? Coverage and unpopularity are secondary metrics.
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
    "InferenceEnvelope",
    "InferenceProvider",
    "Ticket",
    "__version__",
]

__version__ = "0.2.0.dev0"
