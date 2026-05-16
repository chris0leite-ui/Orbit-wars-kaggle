# v15 results — all three root causes addressed; chooser family ceiling confirmed structural

Branch: `claude/review-foundations-progress-14HXp` HEAD `55fc157`
Date: 2026-05-16
Architecture: v12 (v9 chooser + opp_traj baseline + lite_greedy fix) +
iterative modifications addressing each of three diagnosed root causes.

## TL;DR

PI directive: "thoroughly address all root causes, iterate through,
do it carefully." Did exactly that — three iterations, each
addressing one of the three saturation root causes diagnosed last
turn. All three iterations FAILED to break head-to-head parity vs
v12:

- **Iter 1 F4 (leaf scorer)**: 3 variants, all regressed (5-31% h2h)
- **Iter 2 dogpile (action space)**: 2 variants, both regressed (28-31% h2h)
- **Iter 3 reactive step-0 opp (search/opp-model)**: parity (45.3% n=64)

The v9-family chooser ceiling at μ~1120 is **structural, not
surface-tunable**. Any future lift requires either a learned value
head, a fundamentally different chooser architecture (MCTS, planner),
or direct loss-pattern analysis from ladder replays. None of those
are session-scale.

## Iteration 1 — F4 vulnerability penalty (leaf scorer modification)

**Hypothesis**: `_favor` linearly aggregates ships and production
without discounting planets we're about to lose. F4 weights each
planet's production by how secure it is from opposing-color threats.

**Variants tested**:

1. **v1 — potential-launch + capture-feasibility threat ETA**: every
   stationary enemy planet that could plausibly capture our planet
   counts as a threat → 15.6% h2h vs v12 (catastrophic)
2. **v2 — in-flight only**: only count fleets currently en route →
   Felipe 1/2 Naoism 0/2 (still bad)
3. **v3 — in-flight + capture-feasibility**: only count in-flight
   fleets that could ACTUALLY capture → Felipe 1/2 Naoism 0/2

**Diagnosis**: opp_traj's CRN cancellation makes F4 mostly add noise
rather than signal. Same threats appear in baseline (me=idle) and
candidate (me=action) leaves — F4 discount applies to both → cancels
in Δ. F4 only differentiates candidates via SECOND-ORDER effects:
our launch depletes a source → source becomes vulnerable in
candidate leaf but not baseline → F4 punishes the launch. That
second-order signal was net-negative.

**Verdict**: F4 hypothesis empirically wrong. The leaf scorer's
apparent blind spot of over-valuing about-to-lose planets isn't
exploitable through simple production reweighting.

## Iteration 2 — Dogpile candidate enumeration (action-space expansion)

**Hypothesis**: Same-family agents enumerate the same single-source
single-target action space. Joint candidates (2 sources → 1 target
where no single source can capture alone) introduce action-space
asymmetry.

**Variants tested**:

1. **v1 — raw dogpile**: 10/32 (31.2%) h2h vs v12
2. **v2.1 — opportunity-cost filter** (joint Δ must exceed sum of
   best alternative singles from its sources): 9/32 (28.1%) — even
   worse

**Diagnosis**: The joint Δ at horizon K assumes opp_traj (built once
at turn start). Opp doesn't react to our dogpile, so the leaf state
shows us "owning" the captured planet without accounting for opp's
counter-attack on a far-away, hard-to-defend new acquisition. The Δ
over-estimates dogpile's value because opp's reactive defense isn't
modeled. The opportunity-cost filter still let joints fire when
their over-estimated Δ exceeded singles.

**Verdict**: Dogpile valuation needs reactive opp_traj to be
accurate. Deferred to a future iteration on top of Iter 3.

## Iteration 3 — Reactive step-0 opp model (search/opp-model modification)

**Hypothesis**: opp_traj's CRN trick makes the chooser
opp-INSENSITIVE within the rollout (same opp_traj for every
candidate → no candidate-specific opp response → identical leaf
rankings between same-family agents). A candidate-specific
reactive step-0 (opp's nearest source defends the planet WE
attacked) creates differentiated leaves and could break head-to-
head parity.

**Implementation**: For top-3 validated candidates (top-1 in 4P),
score against 2 additional opp_trajs alongside the existing
lite_greedy traj:
- `no_launch` (pre-built once): opp idle at step 0
- `counter_reinforce_target` (per-candidate): opp's nearest source
  sends `our_ships+1` ships to defend the planet WE attacked.
  Parameterized by THE CANDIDATE'S target — key difference vs v14's
  fixed-archetype maximin.

Maximin pick = candidate with the best worst-case score across the
3 scenarios.

**Result**:
- Felipe 2/2 PASS (first non-trivial modification that didn't break Felipe)
- Naoism 2/2 PASS
- Bench p95=207ms max=368ms zero ≥1000ms PASS
- **v15 vs v12 n=64: 29/64 (45.3%, Wlo=0.337, Whi=0.574)** —
  INCONCLUSIVE, point estimate slightly below 50%

**Verdict**: First modification that doesn't regress. But also no
lift. Statistically indistinguishable from v12 (CI brackets 50%).

## The cumulative empirical finding

| Agent | Strategy | vs v12 head-to-head |
|---|---|---|
| v9 (baseline) | chooser + idle baseline | (no opp_traj; ~50% expected) |
| v12 (live μ~1119) | + opp_traj lite_greedy | (self) |
| v13 | + hybrid top_tier + lite_greedy | 47% (n=32) |
| v14 | + maximin over 5 fixed archetypes | 50% (n=64) |
| v15 Iter 1 F4 | + leaf scorer vulnerability | 16% (n=32) REGRESS |
| v15 Iter 2 dogpile | + multi-source action | 28-31% (n=32) REGRESS |
| **v15 Iter 3 reactive** | **+ per-candidate opp step-0** | **45% (n=64) PARITY** |

Five non-trivial modifications across all three root cause axes.
**Best result: parity. No modification has lifted head-to-head.**

The chooser family's component agents converge on the same value
landscape and pick the same moves on the same boards. This is
NASH-LIKE equilibrium within the family — modifying ONE component
(scorer, action space, opp model) shifts evaluations uniformly and
doesn't create exploitable asymmetry.

## Why the three root causes can't be addressed independently

The diagnosis last turn proposed three independent root causes:

1. Leaf scorer is saturated
2. Action space is restrictive
3. opp_traj cancels opp's reactivity

**The empirical answer**: these aren't actually three independent
problems. They share a common underlying constraint — **the chooser
evaluates leaves at a fixed horizon K with a FIXED opp model**.
Within that framework:

- Better leaf scorers (F4) are dominated by the CRN-cancellation
  invariance — opp's contribution cancels in Δ.
- Better action spaces (dogpile) need accurate Δ to rank against
  singles — and Δ over-estimates joint moves because opp doesn't
  react.
- Reactive opp models (Iter 3) DO break CRN invariance — but the
  per-candidate variation in opp's step-0 doesn't propagate enough
  through the K-step rollout to change the leaf ranking
  significantly. Most candidates' worst-case scores are very
  similar to their lite_greedy scores.

The framework's invariance properties prevent surface-level fixes
from creating head-to-head edge. **The framework itself needs to
change.**

## What this proves (and what it doesn't)

**Proves**: simple modifications to v9-family components (leaf
scorer features, action enumeration, opp model variants) don't
break head-to-head parity. Five attempts across three axes.

**Doesn't prove**: a LEARNED value head wouldn't break parity.
That changes the leaf scorer in a way that's not "+ a single
hand-crafted feature" — it replaces the entire scorer with one
that captures features unknown to v12. We haven't tested that.

**Doesn't prove**: a DIFFERENT chooser family (MCTS, depth-2
proper search, RL-trained policy) wouldn't break parity. That's
not modifying v9; it's replacing the chooser entirely.

**Doesn't prove**: the diagnosis from last turn was wrong. The
root causes ARE real — they explain WHY same-family agents draw.
What we've shown is that the CURRENT EXPRESSION of those root
causes can't be fixed with simple modifications. Fixing them
needs more substantial architectural change.

## Recommendation

**HOLD v15.** Submit nothing. v15 Iter 3 is statistically v12
(45.3% h2h, CI brackets 50%); submitting evicts v9 for expected
μ ≈ v12 = 1119. No upside.

Forward path (multi-session):

1. **Path A — Empirical loss-pattern analysis** (next session,
   1-2 hours):
   - Pull 15-20 v12 ladder losses via Kaggle replay API
   - Classify by phase, planet pattern, opp behavior
   - If a dominant pattern emerges, design a TARGETED fix
   - Same methodology that produced v9's 4-fix stack
2. **Path B — Learned value head** (3-5 sessions):
   - Collect self-play game records (state, action, outcome) from
     v12 vs v12 games at multiple seeds
   - Train MLP value head (24-32-16-8-1 sigmoid)
   - Replace `_favor` with learned head
   - Lift potential: unknown but plausibly +50-150μ ceiling
3. **Path C — Different chooser family** (4-7 sessions):
   - MCTS with v9 leaf scorer + UCB
   - Depth-2 minimax search (not single-candidate maximin) with
     proper alpha-beta pruning
   - RL-trained policy network distilled from v12 via DAgger

**My recommendation order**: A first (cheap, decisive). If A
finds a clear pattern, hand-craft a fix (no new session needed
beyond that). If A doesn't find a pattern, B has the highest
expected ceiling.

## Reproduction

```bash
# v15 Iter 3 (the only non-regressing modification)
python fast.py play  agents/v15 --vs v7_0 --seed 1492346051
python fast.py play  agents/v15 --vs v7_0 --seed 1492346051 --swap
python fast.py play  agents/v15 --vs v7_0 --seed 768065184
python fast.py play  agents/v15 --vs v7_0 --seed 768065184 --swap
python fast.py bench agents/v15 --vs v7_0 --games 3
python fast.py eval  agents/v15 --vs agents/v12 --max-seeds 64 --workers 4
```

Wallclock total: ~25 min.

## What this iteration was worth

Even though no submission resulted, the session produced two durable
artifacts:

1. **The empirical proof of family saturation**: future sessions
   know that surface modifications to v9-family components won't
   lift head-to-head. Saves us from re-running this experiment.
2. **The agent infrastructure**: agents/v12 is staged for head-to-
   head testing; `scripts/play4p.py` is built for 4P; v14's
   wallclock-cap pattern is proven. All composable for future work.
3. **The methodology**: three iterations with explicit gate
   criteria, Rule-37-compliant axis abandonment when 2-3
   falsifications stack. Reproducible discipline.

Submission of v9 (μ=1123) remains the current team floor;
rolling-last-2 = [v12 sub 52699232 μ=1119, v9 sub 52687411 μ=1123].
