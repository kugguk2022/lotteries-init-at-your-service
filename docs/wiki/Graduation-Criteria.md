# Graduation Criteria

Two things about this project are worth waiting for, and both are easy to get wrong by declaring
them early:

1. **The package graduates from alpha to a stable release.**
2. **A provider produces enough prospective evidence to say it beat `uniform_random`.**

Neither should depend on anyone's mood on the day. Both are encoded as machine-checkable gates in
[`scripts/check_graduation.py`](../../scripts/check_graduation.py), run monthly by the
[`graduation-watch`](../../.github/workflows/graduation-watch.yml) workflow, which opens **one**
GitHub issue when a milestone is actually reached and stays silent otherwise.

Run the same check locally at any time:

```bash
make graduation        # or: python scripts/check_graduation.py
```

## Why a workflow instead of a reminder

Watching releases on GitHub tells you a release happened; it cannot tell you whether the release
*should* happen. A PyPI RSS feed has the same problem, and only exists once the production project
does. The evidence milestone is worse still: nothing outside this repository can observe it, and the
number it depends on — realized ROI — is exactly the kind of number that looks exciting for the wrong
reason.

So the watcher checks conditions rather than announcing events. It has one job beyond that: to not
become noise. Every notification embeds a marker, and the workflow searches **all** issues, open and
closed, before opening one. A milestone that was reported and dismissed stays dismissed.

## The stable-release gates

Every gate reports `met`, `unmet`, or `unknown`. Graduation requires all of them `met` — `unknown`
never counts as a pass, because "we did not measure this" and "this is fine" must not render the
same way.

| Gate | What it checks | How |
|---|---|---|
| `pypi_install` | The published package installs on 3.10–3.14 | A matrix job installs `lottobench` from PyPI and imports it. Deliberately **without** `--pre`, so a green result is itself evidence that a stable release exists. |
| `release_history` | Two or more alpha/beta releases exist | Release tags in the clone. |
| `public_api_stable` | No public name was removed | `__all__` in `lotteries_core/__init__.py`, compared against every release tag. Old revisions are `ast`-parsed out of git, never imported. |
| `ledger_schema_stable` | The ROI ledger schema held | `ROI_SCHEMA_VERSION` at each release tag matches the current one. |
| `benchmark_reproducible` | Benchmark output is deterministic | The shipped CLI is run twice over the same draws and the two files must be byte-identical. |
| `settled_roi_observations` | There is enough settled data to say anything | Distinct settled, control-matched draws in the ledger. |
| `no_critical_issues` | Nothing critical is open | Open issues labelled `critical`, **plus** an integrity pass over every ROI ledger record — a ledger row that fails validation is itself a data-integrity problem, whether or not anyone filed it. |

The reproducibility gate replays a short recent window rather than the full history. Determinism is
a property of the pipeline, not of history length, and the full canonical history costs over ten
minutes per run once the `ml` and `transformer` extras are installed — long enough that the check
would get skipped in practice, which is worse than a smaller one that actually runs. It keeps
`--all-providers`, because the optional heavy providers are precisely the ones carrying enough
internal randomness to be worth pinning down.

## The provider-evidence gate

This is the one that has to resist wishful reading. A lottery ledger is dominated by draws where
nobody won anything, punctuated by rare large payouts. In that shape, **positive cumulative ROI is
not evidence of anything** — one lucky ticket produces it with no predictive content behind it.

So a provider must clear all five of these against its matched `uniform_random` control:

| Threshold | Default | Why |
|---|---|---|
| Settled control-matched draws | ≥ 30 | Below this the question is not yet askable. |
| Decisive draws (not a tie) | ≥ 10 | Draws where both sides returned the same amount carry no information, and they are the common case. |
| Cumulative net lift | > 0 | Necessary, nowhere near sufficient. |
| Net lift with its single best draw removed | > 0 | A method whose whole advantage is one payout demonstrated a lucky ticket, not an edge. **This is the gate that exists specifically to stop a jackpot from opening an issue.** |
| Exact one-sided sign test | p ≤ 0.01 | Per-draw wins against the control, tested exactly rather than assumed normal. 0.01 rather than 0.05 because the claim is extraordinary. |

Records without a matched control are excluded rather than compared against nothing, and the control
is never scored against itself.

A notification that does fire says plainly what it is: prospective evidence from a single ledger,
not a claim of an exploitable edge, and not betting advice — see
[Experimental Use and Liability](Experimental-Use-and-Liability.md). The right response is to
replicate on an independent ledger, not to publish.

## Tuning

Every threshold is a flag, so a decision to move a bar is visible in the diff that moves it:

```bash
python scripts/check_graduation.py --min-draws 50 --alpha 0.005
python scripts/check_graduation.py --skip-benchmark      # fast, skips the reproducibility gate
```

The two facts the script cannot observe from a checkout — the live PyPI install matrix and the open
critical-issue count — are supplied by the workflow as `--pypi-report` and `--open-critical-issues`.
Omit them locally and their gates report `unknown`.
