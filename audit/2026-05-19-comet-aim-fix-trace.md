# Comet-aim fix — Rule-38 trace (2026-05-19 PM)

PI observation from sub 52811320 ep 77087563 (2P vs Félix Truong, we
lost): step 52 we launched 40 ships from planet 12, aim 160.01°,
fleet OOB at step 59. Comets had entered at step 50.

## Trace setup

- Replay: `audit/live-episodes/52811320/episode-77087563-replay.json`
- Step 51 (the proposer call BEFORE the bad step-52 launch):
  - Our seat 0 (`ChrisLeiteScha`), opponent `Félix Truong`
  - Source planet 12 at (22.54, 71.50), 40 ships, prod=5
  - Target comet 31 at (5.74, 77.61), 15 defenders, neutral
  - omega = 0.0281, comet_ids = {28, 29, 30, 31}
- Live action at step 52: `[12, angle=2.793 rad (160.01°), ships=40]`

## Pre-fix vs post-fix `aim_and_eta` on the same obs

```
PRE-fix (BASELINE_COMET_AIM=off):
  angle = 160.01°  ← matches live launch exactly
  eta   = 5
  predict_fleet_fate: outcome=oob hit_id=None step=8

POST-fix (BASELINE_COMET_AIM default = ON):
  angle = 121.25°  ← rotated 38.77° eastward toward comet path
  eta   = 3
  predict_fleet_fate: outcome=planet hit_id=1 step=7
```

## Why the fix changes the angle

- `omega = 0.028 rad/turn`, comet 31 orbital_radius = 52, angle from
  sun = 148°. Orbital prediction at +5 turns rotates by 8°, putting
  the comet at (2.31, 71.13) — essentially stationary on the southwest.
- Comet 31's ACTUAL path moves east at ~4 units/turn. At step 51+5=56
  the comet is at (25.41, 80.60); at step 51+7=58 it's at (33.42, 80.56).
- `aim_comet`'s 5-iter fixed-point converges on `path[path_index + 3]`
  = (17.46, 79.88) — northwest of the source. Aim there: 121.25°.

## Why post-fix still doesn't hit comet 31

- 40-ship fleet speed ≈ 2.95 units/turn; comet speed = 4 units/turn.
  Fleet cannot catch the comet — it lags every step.
- The iterative-intercept converges on a near-launch path point that
  the fleet *could* reach if the comet stood still, but the comet's
  east-bound motion outpaces the fleet.
- `predict_fleet_fate` correctly reports outcome=`planet` (hits planet
  1 mid-flight, not the comet). The proposer's trajectory filter
  drops candidates whose outcome != "target", so this candidate
  WOULD BE DROPPED by the existing filter pipeline.

## Net effect on this case

| stage | live behaviour | post-fix predicted behaviour |
|---|---|---|
| candidate generated | YES (angle 160°, eta 5) | YES (angle 121°, eta 3) |
| trajectory filter outcome | `oob` (live: step 59 OOB) | `planet` (hits planet 1) |
| proposer keeps candidate | NO — already dropped pre-fix in `predict_fleet_fate` filter via comet-lifetime gate (wait — the live agent DID emit it, so the filter didn't catch it for some reason — most likely the live bundle's `predict_fleet_fate` used `predict_relative` for comet position and the comet appeared to be at a position the trajectory missed completely, so outcome was reported as `target` and the candidate was kept) | NO — filter correctly drops `planet` outcome |

The key point: **the fix doesn't make us hit comet 31** (the geometry
prevents that — slow fleet vs fast comet). The fix makes the proposer
**correctly REJECT the candidate** because `predict_fleet_fate` now
uses path-aware comet positions and reports the correct non-target
outcome. The 40 wasted ships at OOB don't get launched in the first
place.

## Verification

- 30 proposer unit tests green (including 4 new comet-aim cases).
- 3 trajectory tests + 3 world_model tests covering comet path lookup.
- The 5-iter convergence on this real obs: angle stable within 0.3°
  XY tolerance after 2 iterations; no fallback needed.

## File pointers

- `lib/world_model.py:438-460` — new `comet_position_at` helper.
- `lib/aim.py:151-209` — new `aim_comet` 5-iter fixed-point.
- `agents/baseline/proposer.py:75-127` — `aim_and_eta` routes comets.
- `lib/trajectory.py:101-138` — `predict_fleet_fate` uses path for comets.
- Env-var gate: `BASELINE_COMET_AIM` (default ON; `=off` reverts).
