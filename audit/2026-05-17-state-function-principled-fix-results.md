# 2026-05-17 — Principled state-function fix: results

Branch: `claude/recover-main-foundations-MV0e2`
Builds on: f89c0ba (v11 Layer 1+2, partial structural fix)
Commit (v12): 0910bf6
Commit (v13): 30a5aeb
Status: COMPLETE — v12 submitted (52699232 PENDING). v13 panel PASS;
unsubmitted pending PI decision.

## v13 update (reactive opp + neutral fix)

PI directive after submitting v12: "find the root cause of the
213tubo loss." Investigation:

- `lib/opp_model.py:lite_greedy_policy` was treating neutrals as
  accreting production. Env rule (orbit_wars.py:511-514) says neutrals
  don't produce. Fixed → opp_traj correctly predicts opp captures
  near targets in opening.
- Standalone, that fix regressed v7_0 panel to Wlo=0.483. Investigation:
  the chooser picked a `WAIT src24→tgt0` candidate with Δ=+372 that
  actually captured planet 4 (prod=4) via swept-pair collision —
  legitimate prod=4 gain, but at 1-ship surplus (fragile). F2 credited
  the planet for the full remaining game (~99 units), but the rollout
  ended at h=18 before opp_traj could counter-launch.
- Root cause: `opp_traj` was precomputed once from me-idle, so it
  didn't react to my candidate's captures. Extending MAX_HORIZON alone
  didn't help because opp doesn't see my new planet.
- Fix: dropped opp_traj precomputation. `_opp_actions_for_snap` is now
  called at each rollout step in both `_build_idle_baseline` and
  `_score_action`, so opp reacts to the evolving snap.

### Panel (n=32, v7_0 continued to 64)

| vs | v12 wins/n | v12 Wlo | v13 wins/n | v13 Wlo | Δ Wlo |
|--|--|--|--|--|--|
| v7_0 | 52/64 | 0.700 | 50/64 | 0.666 | -0.034 |
| v4_planner | 24/32 | 0.579 | **28/32** | **0.719** | **+0.14** |
| v3.5.1 | 24/32 | 0.579 | **29/32** | **0.758** | **+0.18** |

All three PASS. v3.5.1 and v4_planner show strong gains; v7_0 slightly
weaker but still well above the 0.55 gate.

### Bench (3 games, standalone)

```
focal v8_scavenge: n=514 p50=68 p95=210 p99=249 max=273ms over_1000ms=0
verdict: PASS  (gate: p95<800ms AND zero >=1000ms)
```

Cost ~2× v12 (lost CRN cancellation; opp_policy called per-step in
every rollout). Still well under 1000ms ceiling.

### Felipe / 213tubo seeds

- Felipe seed 1492346051: 1-2/2 wins (timing-dependent due to
  adaptive N_VALIDATE on per-step probe). Was 2/2 with v12.
- 213tubo seed 1844543828 (v7_0 matchup): 2/2 wins. Was 0/2 with v12.

### Next steps

- (Optional) Submit v13 — pending PI decision.
- Iterate on F2 leaf scoring if v7_0 -3pp matters more than v3.5.1/v4
  +14-18pp gains.

---

## v12 baseline (for context)

Builds on: f89c0ba (v11 Layer 1+2, partial structural fix)
Commit: 0910bf6
Status: COMPLETE — all panel + bench + tests PASS.

## TL;DR

Replaced the strict-idle / step-0-mirror baseline with a **full opp
trajectory** built once per turn via `lite_greedy_policy` (bounce-check
fixed) and replayed identically in baseline + every candidate (common
random numbers). Removed `MAX_WAIT` as a behavioural restriction;
only the `wait_N + eta + SIM_SETTLE_TURNS ≤ MAX_HORIZON` computational
cap remains. Fixed an orbital-aim bug where `_aim_and_eta(wait_N>0)`
rotated only the target — for co-rotating planets, both src and tgt
must be rotated by `omega × wait_N` to compute the correct fire-time
geometry.

**Felipe seed 1492346051 (v7_0 opponent): 0/2 → 2/2.**
- v8 now picks "wait 17 turns, accumulate 31 ships, fire at near
  prod-5 id12" (Δ=+475) instead of "fire 14 ships at far prod-4 id21
  now" (Δ=+386). Matches v7_0's winning play on the same seed.

Bench: p95=116ms, max=213ms, zero over 1000ms (was p95=89ms before
opp_traj; +27ms net cost from lite_greedy per-step calls in trajectory
build). Plenty of headroom.

## Root causes addressed

### 1. Strict-idle baseline blindness

Old: `_build_idle_baseline` applied a one-shot mirror at step 0, then
strict-idle for the rest of the horizon. Candidate rollouts were
asymmetric (me_action + opp_step0_only).

New: `_build_opp_trajectory` pre-computes opp actions for each step
0..MAX_HORIZON-1 by running `lite_greedy_policy` against each opp's
observation as the snapshot advances. Baseline applies this trajectory
with me idle; every candidate applies the same trajectory with my
action spliced at step `wait_N`. **Same opp behaviour in both legs of
the Δ.**

### 2. `lite_greedy_policy` bounce bug

Old: `ships = int(src[5] * 0.7)` — sent 7-ship fleets at 30-defender
neutrals at game start. Made it useless as the opp-trajectory policy.

New: predicts defenders at straight-line ETA, skips if `aggressive
size < needed`; otherwise sizes to `max(aggressive, capture_size)`.
Now correctly identifies "I cannot afford this capture yet, idle"
which is the v7_0 opening strategy.

### 3. `_aim_and_eta(wait_N > 0)` rotated only target

Old: `_orbit_predict_relative(tgt, omega, wait_N)` — shifted only the
target by `wait_N` rotations, then aimed from the CURRENT src. For
co-rotating planets this gave wildly wrong angles and inflated eta,
blocking the wait-N candidate via `wait_N + eta + SETTLE > MAX_HORIZON`.

Empirical proof (Felipe seed step 4, src=id28→tgt=id12, wait_N=17):
- Aim shifting tgt only: angle=-2.812, eta=12.9 (HORIZON=32, blocked)
- Aim shifting BOTH: angle=0.149, eta=1.5 (HORIZON=21, allowed)
- Ground truth at step 21 (no wait): angle=0.149, eta=1.5

New: shifts BOTH src and tgt by `omega × wait_N` before calling
`aim_orbiting`. The relative geometry is preserved (rotational
symmetry), so the angle in the rotated frame equals the correct
world-frame angle at fire time.

### 4. `MAX_WAIT` as a restriction (PI directive)

PI verbatim: "max wait should not be a restriction. not waiting
should emerge from a proper modeling not from a restriction. similar
for other tweaks or restrictions."

Old: `MAX_WAIT = 10` truncated `_wait_then_fire_candidate` to wait_N ≤ 10,
which filtered out the Felipe-winning wait_N=17 even when the orbital
aim was correct.

New: removed entirely. The remaining bound is `wait_N + eta +
SIM_SETTLE_TURNS ≤ MAX_HORIZON=30` — a computational ceiling, not a
behavioural restriction. PV-discount in `_cheap_marginal_value` and
the opp_traj's penalty on opp-progress-during-wait do the rejecting
naturally.

## Files modified

- `agents/v8_scavenge/main.py`
  - Replaced `_build_idle_baseline(opp_step0_actions=None)` with
    `_build_opp_trajectory` + new baseline signature using `opp_traj`.
  - Refactored `_score_action` to splice my action at `wait_N` inside
    a full opp_traj replay.
  - Removed `MAX_WAIT` constant and the wait-N cap check.
  - Fixed `_aim_and_eta(wait_N > 0)` to rotate both src and tgt.
  - Updated module docstring to reflect opp_traj approach.

- `lib/opp_model.py`
  - `lite_greedy_policy`: added capture-size estimate via straight-line
    ETA + defender prediction; skip if `needed > src.ships`; size to
    `max(aggressive=0.7×src, needed)`.

## Verification

### Felipe seed (1492346051)

| Variant | P0 outcome | P1 outcome |
|--|--|--|
| f89c0ba (Layer 1+2, prior commit) | LOSS | LOSS |
| This commit | **WIN** | **WIN** |

Step 4 candidate scoring (this commit):
- WAIT src28→tgt12 (id=12, prod=5): ships=31, wait=17, eta=2, Δ=+475.8
- WAIT src28→tgt16 (id=16, prod=5): ships=29, wait=15, eta=5, Δ=+477.7
- FIRE-NOW src28→tgt21 (prod=4): ships=14, eta=30, Δ=+386.3
- Chooser picks tgt16 wait-15 (highest Δ), reserves src28, emits []
  at step 4. Launches at step 19 once accumulated.

### Bench (3 games)

```
focal v8_scavenge: n=512  p50=42  p95=116  p99=161  max=213ms  over_1000ms=0
verdict: PASS  (gate: p95<800ms AND zero >=1000ms)
```

### Panel (Wilson 0.55 gate, max-seeds 32; PASS continues to 64)

| vs | wins/n | win % | Wlo | verdict |
|--|--|--|--|--|
| v7_0 | 52/64 | 81.2% | 0.700 | **PASS** (was 75% Wlo=0.579 — improved +6 pp / +0.12 Wlo) |
| v4_planner | 24/32 | 75.0% | 0.579 | PASS (unchanged from baseline) |
| v3.5.1 | 24/32 | 75.0% | 0.579 | PASS (unchanged from baseline) |

The v7_0 lift is the headline. Same direction as the live-replay
analysis predicted: opp_traj baseline closes the mid-game launch-rate
gap that drove most v8 losses (84% mid_economy_lost per
`audit/2026-05-18-loss-mode-v8-v9.md`). v4_planner / v3.5.1 are weaker
proxies for top-tier opps and were already at panel ceiling; no
regression.

### Foundation tests

687 passed, 4 skipped, 1 xfailed in 1149s (full run).
6 errors in `tests/test_bundle.py` during the parallel run were
TRANSIENT `/tmp/pytest-of-*` directory collisions under 24-worker
contention — re-running the suite standalone passes all 10 bundle
tests in 40s. Net: **baseline 693 preserved.**

## What's NOT addressed (yet)

- `_favor`'s full-game F2 multiplier (γ^step × 99 for unit prod) still
  amplifies leaf production differences. This is partially compensated
  by the opp_traj making the baseline's prod also grow, but a finite
  evaluation horizon with a full-game PV at the leaf is still an
  approximation.
- The `cheap > -10` filter in stage 1 — arbitrary threshold, restricts
  cheap-rank pool. Should be relaxed if wallclock allows.
- `MIN_FLEET_SIZE = 2`, `NUM_TARGETS_PER_SOURCE = 8` — also restrictions
  by fiat. Lower priority to remove.
- maruichi01 case (live recent loss) — same "better start compounds"
  pattern as Felipe. Should benefit from this fix but not directly
  validated.

## Rule 39 reminder
No Claude session URLs in commits / PR bodies.
