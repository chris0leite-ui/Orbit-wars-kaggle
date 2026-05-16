# Chooser family saturation — empirical proof and forward paths

## Date / context

2026-05-16. End of a long session that began with the question
"how do we have the best of both worlds?" (v9-style chooser
strengths + v3.5.1-style opp modeling) and ended with empirical
proof that the v9-family chooser is structurally saturated at
μ~1120.

## The finding

The v9-family chooser — characterized by:
- `_favor` leaf scorer (F1 ship balance + F2 PV-discounted prod share)
- Fresh candidate enumeration (src × nearest-K target × ship-grid)
- `opp_traj` baseline subtraction at horizon K (~30 steps)
- Common-random-numbers (CRN) variance reduction

— exhibits a HEAD-TO-HEAD CEILING within the family. Every variant
we've built (v9, v12, v13, v14, v15) lands at the SAME ladder
μ-band (1099-1123) and is statistically indistinguishable from
v12 in head-to-head: 47% (v13), 50% (v14), 45% (v15 Iter 3).
Best result across 3 chooser variants and 7 sub-iterations: PARITY.

## Why the surface fixes don't work

The three root causes I diagnosed (leaf scorer / action space /
opp model) appear independent but share a common framework
invariance: **the chooser scores leaves at a fixed horizon K
with a FIXED opp_traj that's identical across all candidates**.

Within this framework:

- **Leaf scorer features (F4 vulnerability) are dominated by CRN
  cancellation.** Same threats appear in baseline and candidate
  leaves; F4 discount applies equally; Δ cancels. F4 only
  differentiates via second-order effects (our launch depletes
  a source → source becomes vulnerable in candidate but not
  baseline), and that second-order signal was NET-NEGATIVE —
  it punished aggressive plays.

- **Action-space expansion (dogpile) over-estimates joint value
  without reactive opp.** The K-step rollout shows us "owning"
  a far-away capture without accounting for opp's counter-
  attack on a poorly-defended new acquisition. Joint Δ is
  inflated; joint candidates fire when they shouldn't.

- **Reactive opp models do break the CRN invariance** but the
  per-candidate variation in opp's step-0 doesn't propagate
  enough through the K-step rollout to change leaf rankings
  significantly. Most candidates' worst-case scores end up
  similar to their lite_greedy scores.

The three root causes COMPOSE through the framework. Fixing one
without the others creates invariant shifts that don't change
rankings. The framework itself needs replacing, not patching.

## Why same-family agents converge

Two v9-family agents (v12 and v15) playing head-to-head:
1. Both enumerate the same candidate set on the same board
2. Both score with the same `_favor` leaf scorer
3. Both predict opp's moves with similar opp_traj policies
4. Both produce similar Δ rankings → pick similar moves

Result: NASH-LIKE equilibrium within the family. Neither side
finds an exploitable advantage. Wins are determined by board
geometry (seed) and noise, not by strategic difference.

This is the chess-engine "two engines find the same line"
phenomenon. To win head-to-head, an engine must either:
- VALUE different things (different leaf scorer)
- CONSIDER different actions (different enumeration)
- SEARCH differently (different depth or branching structure)

Surface modifications to a single component don't satisfy any of
these — they're shifts, not divergences.

## What this proves and doesn't

**Proves**: simple modifications to v9-family components (single
hand-crafted leaf scorer feature; dogpile action enumeration;
per-candidate opp_traj step-0 reactivity) cannot break head-to-
head parity. 7 modifications tested across 3 axes; 5 regressed,
2 at parity (one of which is just unchanged v12, the other is
v15 Iter 3).

**Doesn't prove**: a LEARNED value head wouldn't break parity.
That changes the entire scorer in a way that's not "+ one
hand-crafted feature" — it captures features unknown to v12.

**Doesn't prove**: a DIFFERENT chooser family (true depth-2
minimax with α-β pruning; MCTS; RL-trained policy) wouldn't
break parity. That replaces the framework entirely.

**Doesn't prove**: the diagnosis was wrong. The root causes ARE
real — they explain WHY same-family agents draw. What we've
shown is that within the framework, those causes can't be
surface-fixed.

## Forward paths (ranked by tractability × expected lift)

1. **Empirical loss-pattern analysis (Path A — cheapest)**:
   pull 15-20 v12 ladder losses via Kaggle replay API, classify
   by phase/planet-pattern/opp-behavior. If a dominant pattern
   exists (>25% concentration), it's a SPECIFIC feature `_favor`
   misses → hand-craft a targeted fix that BYPASSES the
   framework invariance (because it'd be PARAMETERIZED BY
   GAME-PHASE or BY OPPONENT-OBSERVED-BEHAVIOR, not just by
   leaf-state). Same methodology that produced v9's 4-fix
   stack. 1-2 hours to execute.

2. **Learned value head (Path B — biggest ceiling)**: collect
   self-play game records (state, action, outcome) from v12 vs
   v12 at multiple seeds; train MLP value head; replace `_favor`
   entirely. The learned scorer would inherently differ from
   v12's hand-tuned scorer → head-to-head divergence. Needs
   3-5 sessions: data collection → training → integration →
   panel + h2h validation. Expected lift if it works: +50-150μ.

3. **Different chooser family (Path C — biggest risk)**: MCTS
   with v9 leaf scorer; true depth-2 minimax with α-β pruning;
   RL-trained policy network distilled from v12 via DAgger.
   Each is a multi-session build with uncertain payoff.

My recommendation order: A first (cheap, decisive — answers
"does v12 have a dominant failure mode we can hand-craft for?").
If A finds a pattern → targeted fix (no Path B needed). If A
doesn't find a pattern → B's learned head is the next direction.

## Methodological lesson

Panel results vs one opponent class DON'T predict head-to-head
vs a same-family agent. Three iterations confirmed this (v13,
v14, v15 Iter 3). Going forward, **head-to-head vs v12 at n≥32
is the binding gate**. Panel is necessary but not sufficient.

This finding has been promoted to friction.md as
`panel-misleads-head-to-head` (4th recurrence ⇒ promotion
candidate to .claude/skills/kaggle-comp/improvements.md).

## Don't repeat this iteration loop

This session spent ~6 hours iterating across F4 / dogpile /
reactive-step-0 with diminishing returns. The right strategic
move BEFORE iterating would have been: pull v12 ladder losses
FIRST, then design targeted fixes based on observed patterns.
The "thoroughly address all root causes" directive made sense
philosophically but cost a session of compute that could have
gone toward Path A.

Future-session rule: **for any chooser-family iteration, run
Path A (empirical loss-pattern analysis) BEFORE proposing
specific feature additions.** Surface fixes without empirical
target patterns are likely to be saturated-family no-ops.
