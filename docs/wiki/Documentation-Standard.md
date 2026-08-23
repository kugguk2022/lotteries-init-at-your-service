# Documentation Standard

The bar this repository is being held to, written so it can be copied into the others. It is aimed at
repositories with **human readers and users**, not at private scratch space.

The premise: a reader should be able to learn a repository's real condition from its documentation
rather than from a failing command. Most repositories fail this not by lying, but by describing the
intended state and never revisiting it.

## The rule that does most of the work

> **Nothing goes in the documentation unless it was run.**

Not "should work". Not "works after you install the extras". Run it, paste what happened. If it fails,
document the failure and link to the issue. A README full of aspirational commands is worse than a
short one, because it costs every reader the time to discover which half is real.

## Required, in order of impact

### 1. A truthful first command

Pick one command a newcomer runs immediately after install, make sure it passes, and put it at the
top. If the natural candidate (`make test`, `pytest`, `npm test`) currently fails for unrelated
reasons, say so **and give a working alternative**. A first command that fails is the single most
expensive documentation defect there is — it reads as abandonment regardless of the code's quality.

### 2. A status section that can embarrass you

State what is broken, what is stale, and what is unfinished, with dates. Include a "last verified"
date. This costs credibility once and earns it permanently; the alternative costs it on every visit.
[Current State](Current-State.md) is the worked example.

### 3. Scope: what it does and refuses to do

Especially where a reader may arrive with the wrong expectation. State the refusal plainly and early,
and give the reason rather than the rule. [Project Scope](Scope.md) is the example.

### 4. A tiered repository map

Readers cannot tell maintained code from an abandoned experiment by looking. Label every top-level
directory: **maintained** (tested, gated, safe to build on), **lab** (real but unstable), **legacy**
(kept for provenance, do not extend). One table. See [Repository Map](Repository-Map.md).

### 5. Findings, including the negative ones

If the repository exists to answer a question, record the answers — especially the ones that came out
"no". Negative results are the most easily lost and often the most useful; an undocumented demotion
gets re-litigated by the next person, or by you in six months.
[Methods and Findings](Methods-and-Findings.md) records exactly why one method sits demoted.

### 6. Numbers with their provenance

Every published metric carries the command that produced it, the data it ran on, and the date. A
benchmark number without its command is folklore.

## Structure that scales

Small repository: `README.md` alone, structured as above.

Larger repository: keep the README short — what it is, verified status, install, first command, link to
the wiki — and move the depth into `docs/wiki/` with a `Home.md` index.

Use an **in-repository** wiki (`docs/wiki/`), not the GitHub Wiki tab. In-repo means it is versioned
with the code, reviewable in pull requests, diffable, and works offline. The Wiki tab is a separate
repository that no pull request touches, which is precisely how documentation drifts out of sync
without anyone noticing.

## Mechanics

- **Relative links only**, so they work on GitHub, in editors, and in local checkouts.
- **Link, never duplicate.** Two copies of a fact means one is already wrong.
- **Label the state of each command**: verified / needs network / broken.
- **Date anything that decays**: bundled data, benchmark results, status pages.
- **Fenced code blocks with real output**, not invented output.

## Checklist for a repository intended for readers

- [ ] README opens with what it is, in one paragraph, without marketing.
- [ ] Verified status near the top, with a "last verified" date.
- [ ] Install instructions that were executed on a clean environment.
- [ ] A first command that passes, or an explicit note that it does not and what to run instead.
- [ ] Every top-level directory labelled maintained / lab / legacy.
- [ ] Every published number carries its command, data, and date.
- [ ] Negative results and demoted approaches recorded, not deleted.
- [ ] Known issues listed with causes and suggested fixes, not just symptoms.
- [ ] CI blocks on whatever the documentation claims is green.
- [ ] Licence, and a `CONTRIBUTING.md` that describes the API that actually exists.

## The last one is the one that holds

**CI must block on whatever the documentation claims is green.** Everything above is a snapshot that
decays the moment someone merges. A blocking gate is the only part of the standard that maintains
itself. Where a full gate is not yet achievable, gate the maintained subset, say so explicitly, and
treat expanding that subset as the roadmap — which is exactly the position this repository is in
today.
