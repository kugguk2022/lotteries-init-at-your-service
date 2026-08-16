# Scope, ethics, and what is (and isn't) mathematically possible

Read this before using or extending the framework. It exists so the project stays honest, useful,
and defensible.

## The one-paragraph version

The lottery is a **negative-sum** game: operators return less in prizes than they take in sales.
No analysis in this repository changes that, and none tries to predict the outcome of a fair random
draw, because a fair draw is unpredictable *by construction*. What this repository studies is
narrow and real: whether **coordinating many independent inference strategies** can improve
**combinatorial coverage** and **expected return-per-ticket** under a **fixed ticket budget** —
in **simulation only**, with **no money ever handled**.

## What is NOT possible (and is not attempted here)

- **Predicting which numbers will be drawn.** EuroMillions, Totoloto, and EuroDreams use certified,
  audited random draws. Past draws carry no information about future draws. GLM, XGBoost, deep
  learning, HMMs, GARCH — none of them can extract a signal that does not exist. Any module here that
  touches ML is pointed at *player behaviour* or *diagnostics*, never at the draw.
- **Turning a negative-sum game positive by picking "smart" numbers.** Expected ROI on a random
  draw is fixed by the prize structure and the odds. Number choice cannot move the odds.

## What IS possible (the only real levers)

1. **Unpopularity / jackpot-sharing (draw games).** Most jackpots are *pari-mutuel*: the prize is
   split among all winners. Choosing combinations that few other people pick does not raise your
   probability of winning, but it raises your **expected payout conditional on winning**, because you
   expect to split with fewer people. This is a genuine, well-documented effect and the core of the
   `unpopularity` provider and `roi.py`. It typically makes ROI **less negative** — it does not make
   it positive.

2. **Remaining-prize expected value (instant / scratch games).** A scratch game is a *finite deck*.
   Lotteries publish how many top prizes remain. When enough top prizes remain relative to unsold
   tickets, the expected value of a *remaining* ticket can (rarely) exceed its price. This is public,
   auditable advantage play — and it is the honest kernel of the "Joan Ginther" story (see
   [`GEOGRAPHY.md`](GEOGRAPHY.md)), far more than any number-prediction myth. Modelled by
   `roi.InstantGamePool`.

3. **Coverage under a fixed budget.** Given that we cannot predict the draw, how a fixed number of
   tickets *covers the combination space* is one of the few things fully under our control, and it is
   what a coordinated multi-provider run is trying to improve. Measured by `coverage.py`.

## The "Joan Ginther" framing, corrected

Joan Ginther won large lottery prizes multiple times. The popular "a PhD cracked the code" narrative
is mostly myth: fair draws have no code. The defensible parts of such advantage play are
(a) **instant-game remaining-prize EV** on finite decks, and (b) **volume with disciplined EV
gating**. This framework adapts *that* — public-data EV gating and disciplined, coverage-aware ticket
selection — in a distributed way. It does not adapt, because it cannot exist, a method to predict a
fair draw.

## Hard product boundaries

This project will not, in code or docs:

- pool or custody user funds;
- purchase tickets or execute wagers;
- claim guaranteed or positive expected winnings;
- present simulated ROI as achievable profit.

Everything is forward-only simulation and measurement. If a contribution crosses any of these lines,
it is out of scope regardless of how well it performs.

## A note on responsible use

Gambling can cause harm. Nothing here is advice to play. If you or someone you know is struggling
with gambling, contact a local support service (for example, in many countries, a national gambling
helpline). This tool is for research into inference and combinatorial coverage, not for encouraging
play.
