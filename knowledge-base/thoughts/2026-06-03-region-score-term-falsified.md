# 2026-06-03 — region value as a score-term: falsified (the lever the last handover named)

The previous session built the region/chunk MVP, got parity, and diagnosed
*why*: the chooser picks by its post-rollout look-ahead score, not by the
candidate's cheap pre-score, so the bias hook (which reorders candidates before
the rollout) gets absorbed. The handover named one untried lever as the most
likely path from parity to lift: **add region value as an additive term in the
chooser's final score** — feed the rollout at the *scoring* layer instead of the
*enumeration* layer. This session built and tested it. **It does not work.**

## What I built

New flag `BASELINE_REGION_SCORE` (default OFF, byte-identical), independent of
the bias hook so it A/Bs in isolation. The region desirability is the *same*
function the bias hook already uses — I extracted `_region_factor` so both share
it, then `_region_desirability_by_id` returns `factor − 1.0` per planet (range
[−0.3, +0.5]). In `choose_trajectory`, after every candidate is scored by the
rollout and before the final sort, I add `weight × desirability × (mean turn Δ)`
to each candidate's score. Scaling to the mean positive Δ makes it scale-free
across game phases; the design intent is a near-equal-Δ **tie-breaker**, never an
override (only candidates already scored Δ>0 are re-ranked, so it can't resurrect
a move the rollout rejected — the reach-frontier guardrail holds).

## The result (clean null)

3-weight sweep vs the table-ON champion, n=32 each, process-isolated `clean_ab`
(hard-set headers — see harness note below):

- weight 0.10 (gentle): **16/31 = 51.6%**, Wilson [0.348, 0.680] — parity
- weight 0.20 (moderate): **13/32 = 40.6%**, Wilson [0.255, 0.577] — regression
- weight 0.40 (strong): **17/32 = 53.1%**, Wilson [0.364, 0.691] — parity

All three Wilson-lows are far below the 0.55 gate. No lift at any strength.

## Why — the mechanism

It's the same wall the whole region family keeps hitting, now seen from the
other side. A **gentle** bonus does nothing because the look-ahead score
differences between real candidates are larger than the tie-breaker, so it almost
never flips the choice. Turn the bonus **up** enough to matter, and it starts
overriding the look-ahead — picking moves the rollout judged worse — which loses
(the 0.20 regression). There's no sweet spot in between: the band where the
bonus is "big enough to change the pick but still right" is empty, because when
the region bonus and the look-ahead disagree, **the look-ahead is the one that's
right.** Region value is not adding information the rollout doesn't already have.

That's the real lesson: region-as-a-signal isn't a *placement* problem (we tried
enumeration layer → parity, scoring layer → parity/regression). The rollout
already prices whatever the region heuristic was trying to say. The region
abstraction is a nice way for *humans* to talk about the board; it is not extra
signal for an agent that already simulates forward.

## Harness note (reusable)

The MVP's `clean_ab` contamination came from `setdefault` headers: two bundles in
one worker share `os.environ`, so the second's `setdefault` is a no-op and it
inherits the first's flags. Fix that stuck: **hard-set** the differing flag in
each bundle header (`os.environ[k]=v`), and read it into a module constant at
import. Each bundle's header runs immediately before its own code, so each reads
its own value regardless of load order. With that, even in-process A/B is clean;
`clean_ab` (one game per subprocess) is belt-and-suspenders. Also: never pipe a
long A/B through `grep` for live monitoring — grep block-buffers and you go blind;
redirect raw to a file and filter on read. `clean_ab` streams per-game lines with
`wall=Xs`, which is the right progress signal.

## Status

Code committed, default OFF, byte-identical champion — kept in-tree as a
documented dead-end. No submission (Rule 42/43: parity/regression doesn't earn a
rolling-pair slot). The region family is parked (mechanism-ledger +
MULTI_BRANCH). The one thing still alive from this line is the **idle-source
finding** (~90% of planets idle/turn even in close games), which reopens the
joint-coordination question on better evidence than the null that closed it.

## Open questions

- The idle capacity is real; is it correct hoarding or a conversion gap? One
  redeploy heuristic (the advance pass) was net-neutral → leans "mostly correct,"
  but that's one data point.
- If the rollout already prices region structure, the lift (if any) has to come
  from making the **rollout itself** see further or cheaper, or from a signal the
  rollout genuinely lacks (opponent intent, multi-turn coordination) — not from
  bolting a hand-built spatial heuristic onto its inputs or outputs.
