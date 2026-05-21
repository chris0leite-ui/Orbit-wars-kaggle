# 2026-05-21 — drain/sniper iteration → baseline_full submitted

## Session outcome

Submitted `baseline_full` as sub **52893236**. Replaces
`baseline_joint_aggr` (sub 52874528, μ ≈ 1135) in the rolling pair.
Live champion `baseline_joint_aggr_consolidated` (sub 52882014)
remains. PI explicit signoff: "submit baseline full now anyway."

## Features stacked in baseline_full

1. **Orbital arrival safety** (correctness fix in
   `lib/world_model.py:time_to_enemy_threat`). When scoring a
   capture's post-arrival hold, predict BOTH target and enemy
   positions at our arrival via `predict_relative`. The old code used
   current positions, silently scoring orbiting targets that rotate
   into enemy territory as safe. PI directive: "never drop fixing
   bugs!" — kept regardless of A/B parity result.

2. **Stagnant-rear drain** (`drain_stagnant_rear` in
   `agents/baseline/main.py`). Dynamic-reserve replacement for the
   2026-05-18-falsified `drain_idle_rear`. Trigger: src.ships > 2 ×
   `max(production*5, 10)`, zero inbound enemy fleet ETAs, friendly
   target with `d_action` improvement ≥ 8 board units. Each launch
   physics-filtered via `predict_fleet_fate`. n=16 alone: 6/16 =
   37.5% (Wilson [0.185, 0.614]), possible lift vs 25% baseline.

3. **Combat-stack drain** (`drain_combat_stack`). Drain target =
   NON-OUR planet with friendly fleet already inbound. Stacks excess
   onto attacks-in-progress. PI directive: "our large planet sits
   fleets away from combat, we do not cluster at combat."

4. **Sniper** (`emit_sniper_strikes`). When total reserve > 300 and a
   source has ≥80 idle ships, fire a sized strike (margin 1.2× over
   predicted garrison) at the enemy's biggest planet (production ≥
   +4). Sort candidate sources by predicted ETA, not ship count
   (close + big = fast both fold into ETA). Follow-on reinforcements
   from remaining idle sources arrive after capture to bolster the
   new garrison.

## Bugs caught + fixed this session

- `step_of_hit` AttributeError (sniper crashed entire 4P game at
  step 81; all 4 agents → ERROR). Cause: docstring at
  `lib/trajectory.py:25` lied about the field name. Fixed at
  `agents/baseline/main.py:798` + docstring. Code review agent
  located it in ~5 min.

## A/B trail

| Variant | n | Wins | Pct | Wilson |
|---|---|---|---|---|
| stagnant drain alone | 16 | 6 | 37.5% | [0.185, 0.614] |
| orbital fix alone | 16 | 4 | 25.0% | [0.102, 0.495] |
| sniper v1 (ship-count) | 4 | 2 | 50.0% | [0.150, 0.850] |
| sniper v2 (eta-sort) | 4 | 1 | 25.0% | [0.046, 0.699] |
| **baseline_full vs consolidated** | 4 | 2 | 50.0% | [0.150, 0.850] |
| **baseline_full vs v3.5.1** | 4 | 2 | 50.0% | [0.150, 0.850] |

n=16 confirmation of baseline_full was started multiple times,
killed each time per PI direction to pivot. Ladder will be the
real truth.

## What I'd do next

1. **Watch sub 52893236 climb.** Track μ over 12–24h. If μ < 1000
   after 12h ladder games, `baseline_full` is a regression and we
   should not iterate on it further.
2. **n=16 vs consolidated for any next variant** before re-submit.
   The two killed n=16 runs left us blind on `baseline_full`'s true
   gap vs current champion. Don't repeat that.
3. **Drain target choice** is the underexplored axis. PI's image
   showed central planets idle even with `BASELINE_COMBAT_STACK=1`.
   Hypothesis: chooser is over-conservative on AGGR's TOP_K=5
   source enumeration for high-prod enemy targets. Investigate
   raising TOP_K when target.production ≥ 4.
4. **n=4 baseline.** 4-player self-play has a 25% null hypothesis
   for focal wins, NOT 50%. Update the A/B harness output to print
   "vs 25% baseline" so future readers don't make the same misread.

## PI second-brain notes

- PI strongly preferred coherence-over-ablation when six mechanisms
  were floating: "the thing maybe is not to switch somewhat of off,
  but simply to consolidate it, to use all the powers we gained."
  Today's `baseline_full` is exactly that.
- PI insisted bug fixes stay regardless of A/B: "do not drop
  orbitfix! never drop fixing bugs!"
- PI's mental model for sniper: "gather ships from close planets,
  the bigger, the faster." Implemented as ETA-sort.
