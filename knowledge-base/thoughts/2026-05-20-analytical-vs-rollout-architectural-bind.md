# Analytical vs Rollout — the architectural bind

> Session thoughts (Rule 36) — 2026-05-20. Branch:
> claude/strategy-framework-design-OyoYR-rebased.

## The shape of the bind

Across 10 slices and 2 sessions, we tried to push the chooser
from rollout-shaped toward analytical-native. None landed at
ladder-parity.

The deepest finding isn't "the analytical primitives are wrong."
The W1/W2 bounds, the differential leaf eval, the migration
solver, the joint LP — they all pass unit tests, behave
sensibly in introspect, return mathematically clean values.

The finding is that **the analytical pieces don't compose into
a winning chooser when plumbed into a per-candidate scoring
loop**. The rollout chooser is implicitly doing PLANNING —
its leaf-favor evaluates a leaf state that encodes the joint
consequences of the whole turn's move-set. No per-candidate
analytical score reproduces that.

Even the joint LP (Slice 10) lost. That was the architectural
fix the deep diagnosis prescribed: replace greedy per-candidate
emit with bipartite assignment over the whole turn. It didn't
recover. The remaining hypotheses for why are well-articulated
but each is a multi-slice undertaking:

- Single-turn horizon is too short — trajectory's rollout
  simulates 25-30 ticks; LP optimizes only "this turn given
  current state."
- Value calibration mis-tuning — capture / reinforce / migration
  values came from independent formulas, may not be
  comparable on a common scale.
- Candidate space still incomplete — multi-source coalitions,
  time-shifted joints, pre-positioning... all absent from the
  proposer + migration solver enumeration.

## Why the analytical vision still might be right

The math we built isn't wrong. The differential's Δ-favor
correctly says "this move doesn't change anything" when it
doesn't. The migration solver correctly identifies
ship-repositioning opportunities. The joint LP correctly finds
the best one-turn allocation. The W1 bounds correctly prove
captures are holdable.

What we don't have is the GLUE — a control structure that
takes the analytical pieces and produces winning play. The
rollout's glue is "simulate 30 ticks, score the leaf, repeat
per candidate." It's noisy and policy-dependent, but it
**does the planning** that the analytical pieces by themselves
don't.

Two paths forward from here:

1. **Build the analytical glue** — multi-turn planning via DP
   or rolling LP, with the analytical primitives as input. Big
   architectural lift; not a single-session deliverable.
2. **Use the rollout for glue, analytical pieces for
   acceleration** — keep the trajectory chooser's planning role,
   replace its per-step substrate (leaf eval, opp policy) with
   analytical primitives. This is essentially: differential's
   leaf state as the rollout's leaf evaluator, with rollout
   still simulating the path.

(2) might be cheaper. The rollout exists; we'd swap its leaf
evaluator from `lite_greedy + favor` to `differential
projection`. The simulation remains as the planning mechanism.

## What this session taught me (durably)

- Stacking analytical commits on top of a rollout chooser is
  noise. Always. 7 attempts demonstrate.
- "Closed-form zero" is honest, not broken. Loosening the gate
  is an anti-pattern.
- Single-game introspect's full per-source candidate
  distribution matters MORE than the top-3 listing. Lesson
  paid in 1 slice's worth of A/B wallclock.
- Joint optimization over a static one-turn state isn't
  enough to replicate what a 25-turn rollout implicitly plans.
  The horizon matters.

## Where I'd start next session (if PI directs)

Probably (2) above — analytical leaf eval INSIDE the rollout,
not replacing it. The differential's leaf state is faster than
rebuilding World + WorldModel from a fast_sim leaf obs. That's
a wallclock improvement that COULD also be a quality
improvement if `lite_greedy` is replaced with a closed-form
"opp does its best one move per tick" inside the rollout.

But: production stays at μ=1118.8. The marginal hour might be
better spent on submission strategy or opp-class fingerprinting
than on more architectural experiments. PI's call.
