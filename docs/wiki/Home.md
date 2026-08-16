# Project Wiki

Start here. This wiki explains what the repository is, what it honestly does and does not do, how to
run it, and what state it is currently in.

It lives **inside the repository** (`docs/wiki/`) rather than in the GitHub Wiki tab on purpose: it is
versioned with the code, reviewable in pull requests, and cannot silently drift out of date without
the diff showing up.

## Read in this order

| Page | What it answers |
|---|---|
| [Scope and Honesty](Scope-and-Honesty.md) | What this project claims, what it refuses to claim, and why. **Read this first.** |
| [Getting Started](Getting-Started.md) | Install, then a set of commands verified to actually run today. |
| [Repository Map](Repository-Map.md) | Which directories are maintained, which are labs, which are legacy. |
| [Methods and Findings](Methods-and-Findings.md) | Every method in the repo, what it does, and how it has scored so far. |
| [Outcome Tracking](Outcome-Tracking.md) | The prospective ledger: the actual experiment being run. |
| [Current State](Current-State.md) | Honest status: what is broken, what is stale, what the path to green is. |
| [Documentation Standard](Documentation-Standard.md) | The bar this and sibling repositories are being held to. |

## The project in five sentences

This is a **research framework** for studying lottery ticket portfolios, not a betting system. A fair
draw is unpredictable by construction, so nothing here predicts winning numbers and nothing here
claims positive expected return. What it studies is narrower and real: whether coordinating many
independent strategies ("providers") improves **combinatorial coverage** and **expected
return-per-ticket** under a fixed ticket budget, evaluated forward-only. The honest lever it exploits
is **jackpot sharing** — picking combinations the crowd avoids does not improve your odds, but it does
improve the payout *conditional* on winning. It never pools funds, buys tickets, or moves money.

The full argument, including which techniques are mathematically incapable of helping and why, is in
[`docs/SCOPE_AND_ETHICS.md`](../SCOPE_AND_ETHICS.md).

## Current state in one line

The maintained core (`lotteries_core/`) is green and gated in CI. The wider repository is **not** —
the full test suite has two modules that fail to import, repository-wide lint reports 61 errors, and
the bundled draw history is roughly a year stale. See [Current State](Current-State.md) for the
specifics and the prioritized path to fixing it. This is stated up front deliberately: a reader
should learn a repository's real condition from its documentation, not from a failing command.

## Sibling repositories

These are separate public repositories by the same author. They are **not** yet documented to the
standard described here; [Documentation Standard](Documentation-Standard.md) is written to be copied
into them.

- `lotteries-init-at-your-service` — this repository.
- `chainlist`, `ETHEREUMchains` — blockchain chain metadata.
- `Zyntalic_idiom` — constructed-language tooling with a translation GUI.
- `kugguk2022` — GitHub profile configuration.
