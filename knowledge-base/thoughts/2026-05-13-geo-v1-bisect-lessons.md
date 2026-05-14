# 2026-05-13 — geo v1 bisect: which "obvious" heuristics regress

Negative-result session. Built the `geo` agent (lib/geo/{sense,posture,allocator}.py +
agents/geo/main.py) targeting top-10's geometric / multi-launch / source-emptying
fingerprint. Every "obviously helpful" heuristic regressed against v3.5.1.

## Bisect table (vs v3.5.1 bundle, n=32 unless noted)

| Iteration                                       | winrate      | Δ vs base |
| ----------------------------------------------- | ------------ | --------- |
| **bisect-2** v3.5.1-exact source pipeline       | 46.9%        | 0         |
| + greedy-multi allocator (replaces settle_plan) | 15.6%        | **-31pp** |
| + posture multipliers (DEFEND ×2 reinforce ...) | 9.4%         | -37pp     |
| + `_aggressive_for(DEFEND) = False`             | 25.0%        | -22pp     |
| **final**: aggressive=True, mults 1.0 (n=64)    | 48.4%        | +1.5pp    |

The final state is at parity with v3.5.1. The architecture (sense → posture →
settle_plan → realize) is wired and unit-tested (16/16) but adds no value yet.

## What broke and why

### 1. Multi-launch allocator (-31pp)
`allocate_greedy_multi` sorts all missions globally by score, then lets one
source fire multiple times if budget allows before the next source gets a turn.
Source A's top 2 missions (scores 10, 9) BOTH fire before source B's top
(score 8) — A overcommits, B's surplus sits idle.

settle_plan's per-source-then-break gives every source one launch. That
spreads attention better than concentrating force at the top-scoring source.

**Top-10 signal (1.9× launches / turn, multi-launch turns) is real, but
implementing it as "let high-score sources keep firing" is exactly wrong.**
The right design likely involves a per-source launch cap that scales with
garrison (large sources get more, small get 1), with the cap derived from
defensive threat budget. v1.5 work.

### 2. Posture × mission-class multipliers (-37pp)
Setting `DEFEND.reinforce = ×2`, `DEFEND.snipe = ×0.5 enemy-only`,
`BREAK.recapture = ×2.5` etc. distorted settle_plan's per-source ranking
enough that the WRONG mission won the per-source slot in many turns.
v3.5.1 wins precisely because every source picks its best aggressive snipe
unless reinforce strictly dominates.

The mistake was thinking posture multipliers were a *soft prior*. They're not —
they scale scores by 2-3×, which is huge relative to the natural variance in
mission scores. Settle_plan picks per-source best; multiplying a class crushes
the others.

**Lesson: don't multiply scores across classes. Posture should affect
either FILTER (drop wrong-posture missions) or SHIPS (vary sizing), not
score magnitudes.**

### 3. Non-aggressive snipe in DEFEND (-22pp)
`_aggressive_for(DEFEND) = False` called `propose_snipe(aggressive=False)`,
which is v3_snipe's `target_min+1` sizing instead of v3.5.1's
`max(target_min, min(0.7 × src.ships, src.ships - 5))`.

v3_snipe loses to v3.5.1 by 56.6%. Calling non-aggressive in DEFEND
turns is calling v3_snipe-sizing for the most pivotal moments of the game.
Even if DEFEND is only 10-20% of turns, those turns matter
disproportionately.

**Lesson: never use the non-aggressive sizing formula in any posture.
It's strictly dominated. Defensive shaping happens via SHIP RESERVE in
the allocator (keep X garrison home), not by sending smaller fleets.**

## The architectural ceiling problem

The bisect-2 result (46.9% with `snipe(aggressive=True) + reinforce +
settle_plan`) is approximately v3.5.1's source-pipeline performance vs its
own bundle. That's a CEILING for "v3.5.1 wrapped in new scaffolding."
To beat v3.5.1 we have to *strategically* differ from it, but every
strategic difference I tried regressed.

The candidates I haven't tried yet (deferred to v1.5; substrate is in
place via lib/geo/sense.py):

- **Voronoi-aware target filter** in OPENING — drop neutrals where enemy
  reaches us first (sense.voronoi[pid] != my_cluster). This sharpens the
  opening grab without changing scoring.
- **Front-pressure reinforce bonus** — multiply reinforce.score by
  `(1 + 0.3 × front_pressure)` where front_pressure is enemy threat at
  the planet. This is a SMALL bias, not a 2-3× domination.
- **Multi-launch with surplus-aware cap** — `allocate_greedy_multi` but
  bounded `max_per_source = max(1, garrison // 15)`. Doesn't over-concentrate.
- **Comet-claim gate** — drop snipe missions targeting comets where
  `sense.comet_claims` doesn't list our cluster (we'd lose the race).

Each of these is a SMALL, surgical change. The lesson is that competitive
agents are won/lost in the margin, not by sweeping architectural redesigns.
The geo SCAFFOLDING (sense/posture/allocator) is the right place to put
small surgical changes — but the changes themselves need to be small.

## What we should ABSOLUTELY NOT try again

- Cross-class score multipliers > 1.5×.
- Replacing settle_plan with global score-sort.
- Non-aggressive snipe sizing in any posture.
- Adding new mission classes (recapture, opening) without first measuring
  their contribution in isolation against v3.5.1 source.

## What this session DID produce

- `lib/geo/sense.py` — clustering, Voronoi, front, threat, comet-claim.
  Pure-function, unit-tested. Free to import from any future agent.
- `lib/geo/posture.py` — 4-way posture decision tree. Pure-function.
- `lib/geo/allocator.py` — both `allocate_greedy_multi` and `allocate_lp`
  exist as primitives. Don't use them in EXPAND yet.
- `agents/geo/main.py` — clean scaffold at v3.5.1 parity. The seam for
  v1.5 surgical changes.
- `tests/test_geo.py` — 16 tests covering all three layers + e2e.
- `fast.py` — eval harness extended with agents/<name>/main.py auto-resolution.
- `audit/friction.md` entry (to be added) documenting the bisect.
