# 2026-05-27 — heuristic agent: prevent-vs-enable asymmetry

**Branch:** `claude/heuristics-agent-physics-ZvZIm`. Session
shipped Phase 3 (source-defense reservation), falsified four
Phase 4 variants on the same underlying axis. Decision-quality
postmortem in `audit/2026-05-27-postmortem-heuristics-agent-
physics-ZvZIm.md`.

## The pattern, three runs in a row

Three Phase 4 variants this session were "modeling fixes that
enable more launches":

1. **2-source joint capture** — combat rule 1 stacks same-step
   same-owner arrivals. Sized two fleets at later-eta align.
   Captured targets no solo could afford. 4/32 vs Phase 3's
   7-8/32. Within Wilson noise of baseline.

2. **Multi-wave `time_to_hold` cap** — bound the post-capture
   horizon at the first enemy wave the hold sizing can't beat,
   instead of `EPISODE_STEPS − step − eta`. Should make us prefer
   captures we hold long; in fact made us decline captures we
   would have held by reinforcing later. 6/32.

3. **Idle drain to most-frontier own planet** — keep ships moving
   (user's stated principle). Forward leftover sendable to the
   own planet closest to non-mine planets. 0/32 — catastrophic.

The diagnosis that ties them together: each addition operates by
**spending ships sooner**. Joint commits two sources to one
target where one solo would have under-committed. Multi-wave
cap declines captures the agent's later turns might have saved
by reinforcement; the cap is too pessimistic precisely because
it's static. Drain ships from a safe deep source into a frontier
planet that's frontier *because it's where the fighting is* —
ships in flight to a doomed sink hand themselves to the opponent.

By contrast, the two fixes that did work — Phase 2b (hold-aware
sizing) and Phase 3 (source reservation) — operate by **NOT
spending ships in cases where the agent would have lost them**.
Hold-aware sizing refuses captures sized too thin against the
next in-flight enemy wave. Source reservation refuses launches
from planets that are themselves about to be hit.

## Why I think the asymmetry holds in this design

A greedy single-turn ROI agent has limited information about its
own future moves. When it under-spends, the next turn re-evaluates
and the missed launch is still available. When it over-spends,
the loss is locked in at arrival time. Defensive modeling fixes
(prevent bad launches) inherit the agent's natural turn-by-turn
re-evaluation; aggressive modeling fixes (enable more launches)
commit ships into futures the agent cannot replan for.

This is symmetric to what falsified at Track A: "stacking
analytical layers on a rollout chooser is noise — analytical
needs multi-turn glue OR must replace rollout entirely"
(`state/MULTI_BRANCH.md` Track A status). The fix that needs
multi-turn coordination doesn't work as a one-shot single-turn
addition.

## What I'd say to a depth-1 lookahead agent

If you simulate the next state of the board after one of your
candidates and pick by post-state ship-delta, the "spend ships
sooner" fixes might score correctly because the simulator can
SEE the doomed-sink outcome. Greedy can't. So:

- Idle drain that aimlessly sends to the most-frontier planet
  is a greedy-only failure mode. A lookahead would see the sink
  flip and reject the drain; greedy can't.
- Joint capture's double-commit issue (steals from solo) would
  show up in lookahead as one of two compared states being worse;
  greedy ROI sorts joints LOW (more ships) so they rarely beat
  solo when solo exists — the problem appears only when joint
  uniquely captures, and then the cost is that the involved
  sources can't capture other things this turn, which lookahead
  could weigh.

## Rule 37 axis-naming

The four variants felt different at write-time (joint vs cap vs
drain vs keep-N). At evaluate-time they were a single axis. I
recognised Rule 37 on the 4th, not the 3rd. The fix for next time
is to **name the axis before writing the candidate** — if the
axis already has 2 falsifications, the variant counts toward the
cap regardless of surface differences.

## Open question for whoever picks this up

Phase 3's source reservation could be a useful **predicate** for
Track B's production lineage chooser — independent of the rest of
this heuristic agent. The predicate (`max_sendable(p, wm)`) is
~30 LOC of pure Python and reads only `wm.owner_at` / `wm.ships_at`,
which Track B already builds. Plugging it in as an opt-in launch
gate is a fair experiment.
