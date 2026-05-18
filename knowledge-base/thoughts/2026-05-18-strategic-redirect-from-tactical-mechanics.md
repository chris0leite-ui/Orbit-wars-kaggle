# PI strategic redirect: stop tactical mechanics, start objective-first search

**Date**: 2026-05-18
**Branch**: `claude/ml-competition-strategy-PFhzM`
**Context**: End of Phase C+ session. Bundle had been iterating on
chooser/scorer/opp-model knobs for many sessions. cands=5 fix
landed (vs v7_0 81%), lead-aim regressed it, symmetric lead-aim
didn't recover. PI called for wrap-up + pivot.

## PI's voice-dump (paraphrased, my best reconstruction)

> Okay, so let's wrap up and see how we can improve next time,
> where we can move from here. I think you're too deep into the
> "when I do this, then you do that" and so on.
>
> We have a huge advantage, and we have yet to find out how to use
> it.
>
> And I think one step is a better search of what is really good
> now in the game. We need to understand what are the most important
> moves now — like which planets to capture, how to defend, and
> where to attack. We need to dig deep into that and simply find
> out, can the opponent hinder us from doing that, and what is the
> likelihood also? And then if we found some really good moves or
> joint actions and subsequent trajectories also, then we go for
> that.

## My read of what this means

PI is pointing at a **different abstraction level** for the bundle
agent's decision-making.

### Current state (what we've been doing)
```
1. Enumerate candidate single moves (src × tgt × ratio × launch_turn)
2. For each, simulate trajectory (with opp prediction)
3. Score path-integrated planet + production + ship deltas
4. Pick argmax
```
This is **move-centric**. Every iteration tweaks the enumeration
shape (cands), the geometry (aim), or the prediction (opp model).
Five of seven variants on this axis null/regressed.

### What PI is pointing at
```
1. Analyze current state: identify strategic objectives.
   - "P14 is the highest-prod neutral within 15 turns — capturing
     it would dominate the south quadrant."
   - "P0 is vulnerable to P3's pressure — must defend within 8
     turns or lose."
   - "Opp's P15 is undefended — a 30-ship strike from P8 + P16
     captures it cheaply."
2. For each objective, identify the moves (single or joint) that
   serve it. Each objective has an estimated VALUE if achieved.
3. Filter by feasibility: "Can opp prevent this in reasonable
   play?" If yes, discount the value by likelihood.
4. Commit to the highest expected-value objective AND emit moves
   that serve it (potentially joint, potentially multi-turn).
```
This is **objective-centric**. The search is over OBJECTIVES, and
moves are the implementation of the chosen objective(s). Joint
actions emerge because objectives often need multi-source backing.

### Why this might be the unlock
- **Trajectory layer's strength is path-integrated value over
  long horizons.** Used currently to score single-turn moves —
  wasted. Should score TRAJECTORIES that pursue objectives.
- **Beam search at depth=2 is barely better than greedy** because
  the score function is too noisy at the move level. At the
  objective level, the search space is smaller and the value
  function is more meaningful.
- **Opp model becomes a feasibility filter, not a step-by-step
  prediction.** "Can opp prevent objective X?" is a simpler
  question than "what move does opp make at turn t?".
- **Joint actions are first-class.** A 60+60 ship gang-up emerges
  naturally from "capture opp's P2" objective, not as a beam-
  search miracle.

## Connection to existing work

`agents/baseline/proposer.py` already does something close to this.
It has a `MissionPanel` of mission types (snipe, reinforce, opening,
drain, gang_up) that propose moves with built-in objective values.
The `chooser.py` evaluates missions and picks the highest-EV bundle.

Bundle agent does NOT use this mission framework. It enumerates raw
candidates. **That's the gap PI is highlighting.**

The minimal pivot: have bundle's enumeration consult a mission
proposer first, get a set of "strategically interesting" candidate
bundles, then score them with trajectory-layer's path-integrated
metric. Best of both: bundle's compute headroom + baseline's
strategic judgment.

## What to chase next session

Per Rule 37 (axis exhaustion), pivot off the chooser/scorer axis.
Pivot options:

| Option | Cost | Captures PI's intent? |
|---|---|---|
| Mission-based proposer (port baseline's framework) | 1 session | Yes — directly |
| Value-head learning (NN over states) | 3 sessions + GPU | Indirectly — learned implicitly |
| IL warm-start (clone baseline) | 2 sessions | Yes — by example |

PI's framing ("a better search of what is really good") most directly
maps to **mission-based proposer**. The framework exists in
agents/baseline; we adapt it to feed bundle's trajectory-layer
scorer instead of baseline's `fast_sim`.

## Risk: this is yet another axis

Per Rule 37 we should ALSO be wary of "endless pivots". The
discipline check: mission-based proposer gives us ONE new axis to
explore. If it nulls at n=8 A/B, that's the trigger to escalate
again (or go to value-head as the next pivot).

## Open question

PI's "huge advantage" phrasing — I read it as the trajectory layer
+ compute budget. Worth confirming with PI next session what
specifically they have in mind. The redirect makes sense either way,
but knowing exactly what advantage they see helps prioritize.

## File this entry connects to

- `audit/2026-05-18-phase-c-symmetric-leadaim-wrap.md` (session wrap)
- `/root/.claude/plans/foamy-pondering-floyd.md` (multi-session plan
  has Direction 1.A/1.B/1.C — this redirect maps to 1.C-ish: build
  better strategic-judgment substrate before further scorer tuning)
- `agents/baseline/proposer.py` and `agents/baseline/chooser.py`
  (the existing mission framework to study and adapt)
