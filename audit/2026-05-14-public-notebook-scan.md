# Public-notebook scan — Orbit Wars (Rule 22)

**Date**: 2026-05-14
**Trigger**: Phase 0 of the geometry-conditional strategy EDA (plan file: `you-are-a-senior-woolly-nest.md`).
**Notebooks scanned**: 4 (kaggle CLI pull on 2026-05-14; copies in /tmp/orbit-notebooks/, not committed).

## Per-notebook one-liners

| Author | Slug | Approach | Verdict |
|---|---|---|---|
| sigmaborov | orbit-wars-2026-tactical-heuristic | Production-weighted heuristic with sun-avoid + orbital prediction + Voronoi frontline + indirect wealth + multi-phase | **Serious contender.** Closest in spirit to our v7 family. |
| melccoro | orbit-wars-step-by-step-agent-dev-ablation | Nearest-target greedy + angle-based fleet inference, with systematic ablation of three knobs | **Mid-effort.** Useful for the ablation numbers, not the agent. |
| rahulchauhan016 | orbit-wars-target-score-2000-4 | MCTS + Beam Search + CFR + opponent model + neural-net value head + diplomacy, 14 subsystems | **Most sophisticated by far.** Claimed *target* μ=2000.4 (aspirational, not measured — to verify on LB). |
| adilshamim8 | orbit-wars-101 | V1→V4 progressive build (sniper → sun-aware → reinforce → orbit-intercept) with local benchmarks | **Mid-effort tutorial.** Confirms "orbit prediction matters" but no LB number. |

## Key technique inventory across the four

- **Sun avoidance**: sigmaborov (1.5-margin segment-hit), adil (closest-point-on-segment), rahul (implicit). Melccoro none.
- **Orbital lead prediction**: sigmaborov (5-iter fixed-point), adil (angular velocity step). Others none.
- **Frontline / Voronoi**: only sigmaborov (nearest-distance-to-set). Rahul has centroid-based border pressure (similar but coarser).
- **Multi-turn search**: only rahul (MCTS budget 420 ms/turn, 10-turn rollout, beam K=3).
- **Opponent modelling**: only rahul (per-enemy archetype).
- **Comet policy**: sigmaborov *conditional* (grab if profit ≥ 1 and turns-to-expire positive). Adil light. Melccoro ignore. Rahul implicit.
- **Game-phase switching**: sigmaborov (4 phases), rahul (3-tier domination thresholds), melccoro (binary "early-off until turn N"), adil (late consolidation).
- **Fleet speed formula**: all four reproduce `1 + 5·(log ships / log 1000)^1.5`.

## Findings relative to our prior work

- **Confirmation**: our top-10 fingerprint observation that "winners avoid comets" matches sigmaborov's *conditional* policy and contradicts melccoro/adil (who effectively ignore). The conditional gate (`profit ≥ 1`) is a cleaner formalisation than our current per-mission heuristic.
- **Tension to resolve in Mine 3 (opening atlas)**: melccoro's ablation says `early_off_until=50` gives **+2-5% winrate over its own baseline**, but our top-10 fingerprints say winners fire by step ~4. Two readings: (a) melccoro's baseline is so weak that suspending its bad early actions is a Pareto improvement — i.e. the question is "fire early *correctly*" not "fire early at all"; (b) the public ladder distribution differs from the top-10 corpus, so "early-off" is right against weak opponents and wrong against strong ones. **Mine 3 must check both.**
- **Gap nobody covers** (vs. our prior work): σ-equivariance, PV target valuation with explicit discount, hierarchical spatial abstraction beyond Voronoi, temporal discounting curves, production-cascade modelling. These remain ours to claim.
- **New idea worth porting**: rahul's "neutral denial" term — production of neutrals close to enemy but not to us. We do not currently penalise letting the opponent expand into uncontested neutrals; this overlaps with our border-pressure intuition but is more directly actionable.

## Watchlist

- **rahul's μ=2000.4 claim — resolved as aspirational.** Public LB top on 2026-05-14 is `bowwowforeach` at μ=1650.6; no `rahulchauhan016` in top 20. So μ=2000.4 is the notebook's *target*, not a measurement. Gap from top to us (μ=1064) is ~590μ — large but not catastrophic; EDA plan stands.
- **sigmaborov's indirect-wealth term** — closest cousin to our PV target valuation. Worth a side-by-side diff in a later session.

## Decisions feeding the mines

1. Mine 1 (board taxonomy) — add **distance from each planet to the home-pair perpendicular bisector** (sigmaborov-style Voronoi) and **comet-arrival-window count** as features.
2. Mine 2 (planet importance) — add **neutral denial** (rahul's term: production of neutrals closer to enemy than to me) as a candidate feature.
3. Mine 3 (opening atlas) — explicit "fire early vs early-off" comparison to resolve the melccoro tension. Bucket top-10 first-launch turn against our submitted agents.
4. Mine 4 (endgame) — keep as planned; no new ideas from notebooks.
5. Mine 5 (sun-shadow) — sigmaborov's 1.5-margin already standard in our `lib/geometry.path_clears_sun`; no change.

## Files

- Raw notebooks (uncommitted): `/tmp/orbit-notebooks/{sigmaborov,melccoro,rahul,adil}/*.ipynb`
- Detailed extraction notes: kept in this session's working memory; only this audit note is committed.
