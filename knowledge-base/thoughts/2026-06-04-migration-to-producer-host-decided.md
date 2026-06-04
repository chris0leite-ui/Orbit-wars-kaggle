# 2026-06-04 — migration to Producer-host decided

After parking the circulation triplet (3 falsifications of pressure-
gradient as a post-pass), the diagnosis crystallised: Producer's regroup
works because his ENTIRE planner thinks in pressure / exact flow diff.
Our scoring is per-`(source, target)` trade ROI. Any thin post-pass
borrowed from him will misalign with our scoring lens.

Two unblock paths existed:
1. Rewrite our chooser to share his scoring lens (port his scoring INTO
   our agent).
2. Use his engine as host, port our pieces into him.

PI decided on path 2 after seeing the n=32 A/B result (we lose ~60 %
to Producer locally; live μ gap ≈ 15 in his favour).

## Why path 2, not path 1

Path 1 means porting `sparse_launch_flow_delta` — a torch per-step combat
simulation across 18 turns — into our numpy/Python stack. Multi-week
build, high risk of subtle drift, throws away Producer's already-
co-tuned engine.

Path 2 means we adopt his engine wholesale and add our mechanisms as new
candidate-generation patterns INSIDE his planner. They get scored by his
flow-diff alongside everything else. Coherence by construction. Smaller
net code change.

## Critical constraint — Producer is NOT submittable

Per PI: "It's not our work. We won't submit it."

The vendored `agents/producer/` is a sparring partner and engine
substrate. Only the hybrid (`producer_plus` with our additions) is
submittable. Documented in `state/MIGRATION_PLAN.md` § Ethics.

## What we keep building from our agent

In order of expected lift × effort:

1. Multi-source coalitions — biggest expected lift; Producer is
   explicitly single-source.
2. Adaptive K horizon — trivial port, phase-aware K.
3. Multiple sizes per (src, tgt) — small port, may already be implicitly
   optimal under his scoring.
4. Wait-then-fire — medium port; needs ledger if turn-to-turn fickle.
5. Comet aim — only if archetype split says we need it.

## What dies

- `chooser_trajectory.py`, `cheap_marginal_value`, `launch_rules.py`,
  `joint_solver/`
- All post-pass drains (idle_rear, stagnant_rear, combat_stack, sniper,
  idle_stockpile, frontier_circulation)
- All `BASELINE_*` env vars

Roughly 5,000+ lines deleted, ~500-1,000 added. Net code shrink.

## What gates each step

n=32 Wilson-lo ≥ 0.55 vs the previous step's bundle (Rule 45 + tighter
than the floor for parity claims). Smoke (Rule 46) before any submit.

## When this becomes a submission

Only when the hybrid clears live-μ-equivalent locally AND maintains
wallclock under 1000 ms max for seed 7. Until then,
`champ_computeByShips_on.py` stays as backstop in the rolling pair.
