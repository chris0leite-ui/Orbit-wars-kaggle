# Session 2026-05-15 — wrap-up

Branch: `claude/bootstrap-competition-setup-6uU6k`
Final commit: `5972247` (v7 lead-aim chooser)
Local A/B vs v7_0 (n=48): **43.8 %** (Wilson lo 30.7 %, verdict TIE)

## TL;DR

Started with a blank-slate agent and after 12 variants reached v7 at
43.8 % vs v7_0. The session's single recurring lesson: **every
breakthrough was about removing a hand-rolled heuristic and trusting
the proven foundations** (`lib/fast_sim`, `lib/aim`, `lib/orbit`,
`lib/fleet`, `lib/geometry`, the v1 favor formula). Every place I
re-implemented something `origin/main` already had, I introduced a
bug; every breakthrough was un-introducing one of those bugs.

## The arc

| # | Variant | Change | vs nearest | vs v7_0 | Verdict |
|---|---|---|--:|--:|---|
| v1 | analytic 1-step Δfavor | F1+F2 chooser | 87.5 % | 0/24 | LOSS |
| v2a | rollout chooser | 10-turn fast_sim, nearest opp | 100 % | 0/24 | LOSS |
| v2b | rollout chooser | 10-turn, n_turns=20 | — | 0/12 | LOSS |
| v2c | rollout chooser | favor-greedy opp | 100 % | 0/12 | LOSS |
| v2d | rollout + F3 leaf | F3 hold-time discount | 100 % | 0/24 | LOSS |
| v3 | analytic only + F3 | drop rollout, keep F3 | 79.2 % | 0/24 | LOSS |
| v4a | exact-physics + v1 favor | drop F3, fast_sim leaf | 100 % | 0/24 | LOSS |
| v4b | + multi-size | 3 sizes per (src,tgt) | 100 % | 0/24 | LOSS |
| **v5** | **baseline-by-horizon fix** | **score_action Δ vs idle leaf** | 100 % | 4/48 = 8.3 % | LOSS |
| **v6** | **drop heuristic guards** | drop _incoming_threat, drop +5 gate, MIN_HORIZON=15 | 100 % | 9/48 = 18.8 % | LOSS |
| **v7** | **lead-aim** | recover lib.aim + lib.orbit + lib.fleet + lib.geometry | 100 % | **21/48 = 43.8 %** | **TIE** |

## The three bugs that mattered

All discovered by following the PI's "play the game step-by-step"
methodology — running `trace_game.py` / `diagnose_chooser.py` on
specific seed × turn pairs and finding moments where the chooser
behaved weirdly.

### Bug 1 — wrong baseline in `score_action` (v5)

`score_action` returned `favor(leaf_after_action_at_horizon_N) − favor(now)`.
The leaf is N turns ahead → favor naturally grows from production
during those N turns regardless of any action. The "Δ" was almost
entirely natural growth credited to whichever action was being
scored. Discovered at seed 1003 turn 2: launching a redundant fleet
at P10 (already targeted) scored Δ = +2873; the truth at the same
horizon is +0.0 (both leaves identical, 143 ships either way).

Fix: build `_build_idle_baseline(snap_base, me, num_seats, horizon)`
once per turn, returning `favor(leaf_after_idle_for_k_turns)` for
`k = 0..MAX_HORIZON`. `score_action` now compares its leaf against
`baseline_favors[arrival + SIM_SETTLE_TURNS]`. Result: 0/24 → 8.3 %.

### Bug 2 — heuristic guards over-rejecting candidates (v6)

Three small guards, each second-guessing the simulator:

- `_incoming_threat` reserved garrison against ANY enemy fleet within
  ±17° of the bearing to my planet. Over-counted (fleets passing
  through the quadrant) and ignored my production accumulating during
  the threat's arrival window. At seed 1003 turn 20 it gated out
  P8 → P22 ×26 (Δfavor = +2320) because P8's launch_budget was capped
  at 20 < 26.
- The `+5` gate in multi-size enumeration blocked `launch_budget` as
  a candidate size when budget was close to capture_size. At seed 1003
  turn 0, ×10 (launch_budget) scored +1453 vs ×6 (min-cap) +1447 —
  small but real lift from faster speed = earlier arrival.
- The sim horizon was `arrival + 2` only. With `_incoming_threat`
  removed, short-distance offensives (arrival=4) wouldn't see
  long-distance threats (arrival=15) land at the home planet.

Fix: drop `_incoming_threat`, drop the `+5` gate, floor the horizon
at `MIN_HORIZON=15`. Result: 8.3 % → 18.8 %.

### Bug 3 — orbital aim miss (v7)

The chooser computed `angle = atan2(tgt.y − src.y, tgt.x − src.x)` —
bearing to the target's CURRENT position. Orbital targets rotate
during the fleet's flight; by arrival the planet has moved 10–15
units along its orbit and the fleet sails into empty space.
Diagnosed at seed 1003 turn 16: P12 → P22 ×26 MISSED (P22 had only
25 ships, capture-ready) because P22 drifted 13.1 units during 11
turns of flight.

The simulator was correctly *predicting* this miss (score_action
returned −inf for the orbital-aim version), but the action the agent
sent to the env had the same wrong angle, so neither the prediction
nor the actual fleet ever escaped the bias.

Fix: recover `lib/aim.py` (5-iter fixed-point lead + safe-intercept
fallback) + `lib/orbit.py` + `lib/fleet.py` + `lib/geometry.py` from
`origin/main`. All parity-tested via `tests/test_orbit.py +
test_fleet.py + test_geometry.py` (39 tests pass). Use
`aim_orbiting()` in BOTH `score_action`'s simulated action AND the
final action submitted to the env. Result: 18.8 % → 43.8 %.

## Recovered from `origin/main` (foundations we built ON)

| File | Purpose | Tests |
|---|---|--:|
| `lib/fast_sim.py` | parity-tested forward simulator | 62/62 |
| `lib/game/interpreter.py` | game engine rebuild | (covered by parity) |
| `lib/aim.py` | 5-iter fixed-point lead-aim + safe-intercept | (covered) |
| `lib/orbit.py` | predict_relative + is_orbiting | 17 |
| `lib/fleet.py` | speed function | 8 |
| `lib/geometry.py` | Point + dist + helpers | 14 |

Not yet recovered (still hand-rolled in our minimal branch):

| Hand-rolled in main.py | Main has | Likely next-session lift |
|---|---|---|
| `favor.py` (F1+F2) | `lib/value_heads.composite_capture_value` | medium |
| `_capture_size_guess` | `lib/missions/snipe.py` proper sizing | small-medium |
| `_enumerate_candidates` | `lib/v7_search._enumerate_drop_or_add_one` | medium |
| `score_action` (idle baseline) | `lib/v7_search.score_candidate` | small (covered by argmax) |
| greedy non-dogpile | `lib/planner.settle_plan` | small |

## Diagnostic scripts written this session

- `diagnose_v7_0.py` — turn-by-turn ledger of one game vs v7_0
- `diagnose_chooser.py` — per-(src, tgt, ships) Δfavor dump at a
  specific seed × turn
- `diagnose_formula.py` — compare v1 vs v2 (F3) Δfavor side-by-side
- `trace_game.py` — full event trace with per-fleet outcome inference

These were the workhorses for finding the three bugs.

## What was tried and failed

- **F3 (defensibility) in favor.** Closed-form `_expected_hold_time`
  estimating "when does this planet fall." AUC dropped 0.945 → 0.912
  on saved snapshots; chooser regressed (79.2 % vs nearest). Reverted
  in v4.
- **v2 rollout choosers** with various opp policies (nearest,
  favor-greedy) at various depths (10, 20). Cascade noise dominated
  signal; all 0/24 vs v7_0.
- **Port to main's v7 stack.** Created a worktree, built
  `agents/v7_0_idle/main.py` (drop_one + idle followup, no cascade).
  A/B vs `agents/v7_ablations/v7_0_drop_one`: 12/24 = 50 %, no lift.
  Main's argmax-over-fixed-K-candidates already handles cross-candidate
  comparison correctly; our v5 baseline-fix doesn't directly translate.
  Worktree torn down. Branch `claude/baseline-fix-port` left on origin
  pointing at `e242099` (main's tip — no commits landed; signing
  infrastructure failure). **Safe to delete remote branch.**

## Did we touch `origin/main`?

**No.** Verified by `git rev-parse origin/main = origin/claude/baseline-fix-port = e242099`. The empty branch on origin has zero new commits. The signing failure prevented any commit from landing during the worktree experiment.

## Commits this session (on `claude/bootstrap-competition-setup-6uU6k`)

```
5972247 v7: lead-aim via lib.aim.aim_orbiting (43.8% vs v7_0)
d99c88e v6: drop redundant heuristic guards on the chooser
5f6a231 diagnose_chooser.py: update signature for v5 score_action
a7a851d v5: fix score_action baseline (compare to idle-same-horizon, not current)
1dbad2e diagnose_v7_0.py: single-game post-mortem dump
cd7b28f v4: exact-physics chooser + multi-size + revert F3 in favor.py
9f111da main.py v3: formula-consistent analytic chooser (drop rollout)
9baae84 favor.py: F3 defensibility — replace F2 with prod × expected_hold_time
53574b2 main.py: rollout opp policy = favor-greedy (instead of nearest)
9d7e96d main.py: multi-step rollout chooser (v2) + recover fast_sim
98441bd main.py: favor-driven chooser (v1)
a0d4d8c Add favor.py + validate_favor.py: decouple world-evaluation from action selection
```

12 commits. Working tree clean. Ahead of `origin/main` by 12, behind by 6.
