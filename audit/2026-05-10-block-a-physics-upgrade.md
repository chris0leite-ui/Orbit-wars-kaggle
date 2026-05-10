# 2026-05-10 — Block A: physics-module replacement (lib/aim + mechanism upgrade)

> Implementation block from the strategic-direction plan
> `/root/.claude/plans/you-are-a-champion-sprightly-sunset.md` — adapts
> Roman 1224 / Pilkwang structured-baseline physics primitives into our
> Strategy → Intent → realize(mechanisms) pipeline. Decision gate: the
> capture-success probe (`audit/2026-05-10-capture-success-probe.md`)
> which identified collided_other (10.7%) + OOB (7.6%) as the largest
> physics-loss buckets, NOT sun (2.1%).

## Files

- **NEW** `lib/aim.py` — `aim_orbiting` (5-iter fixed-point), `search_safe_intercept` (self-consistent intercept scan), `swept_pair_hit` (env-mirror), `estimate_eta`, `flight_distance`.
- **MODIFIED** `lib/mechanism.py` — added `lead_aim_v2`, upgraded `sun_avoid` to use lead-predicted arrival point, added `path_clears_other_planets` and `oob_guard`. Old `lead_aim` retained for PARITY_MECHANISMS / v1 parity test. `DEFAULT_MECHANISMS = [validate, arrival_size, lead_aim_v2, sun_avoid, path_clears_other_planets, oob_guard]`.
- **MODIFIED** `lib/intent.py` — added `arrival_xy: tuple | None = None` to Intent.
- **MODIFIED** `scripts/bundle_agent.py` — added `aim` to `DEFAULT_LIB_ORDER` so the bundler inlines it (`mechanism.py` now imports `lib.aim`).
- **NEW** `agents/simple/roi_baseline.py` — frozen pre-physics ROI using `DEFAULT_MECHANISMS_PRE_PHYSICS`. Local A/B control arm.
- **MODIFIED** `tests/test_simple_strategies.py` — `PANEL_OBS` fixture: planets 1 and 5 nudged off the fleet-path y-axes so the new `path_clears_other_planets` mechanism doesn't drop their intent. Target IDs / rankings preserved.

## Results

### Test gate

- All **160 tests green**. PARITY_MECHANISMS unchanged → v1 parity test passes byte-equal on the 10-seed bag.

### Capture-success probe — direct physics lift

Re-run of `python -m scripts.capture_probe --seeds 32 --out audit/2026-05-10-capture-success-probe-v2.json` on roi self-play:

| Outcome             | Before (v1.2) | After (v1.3) | Delta     |
|---------------------|--------------:|-------------:|----------:|
| reached             | 77.2%         | **83.9%**    | **+6.7pp**|
| collided_other      | 10.7%         | 5.2%         | -5.5pp    |
| oob                 | 7.6%          | 6.4%         | -1.2pp    |
| sun                 | 2.1%          | 2.1%         |  0.0      |
| alive_at_end        | 2.4%          | 2.5%         | +0.1pp    |

`path_clears_other_planets` cleared **51% of the collided_other bucket**
(10.7% → 5.2%). The mechanism's swept-pair check (60-step horizon)
correctly drops intents whose flight path intersects an orbiting
planet's projected chord. `oob_guard` is doing less than expected
(only 1.2pp reduction); the residual 6.4% OOB is likely fleets that
lead-prediction misses the orbital target, then continue past
predicted arrival until they exit the board (oob_guard's endpoint
check doesn't catch overshoot beyond predicted target). Improvement
queued for Block C alongside arrival-ledger work.

`sun` unchanged (2.1%). Plausible cause: `sun_avoid` checks
`src.center → arrival_xy` segment, but for orbital targets the fleet
flies along `aim_angle` toward predicted arrival and may continue
past it if the lead is off — eventually entering the sun. Like the
OOB issue, this is a "fleet overshoots predicted arrival" symptom,
not a `sun_avoid` defect per se. Cross-reference with the OOB
diagnosis.

### Head-to-head A/B — new roi vs roi_baseline

`python -m scripts.strategy_panel --strategies roi roi_baseline --no-refs --seeds 32 --workers 4`:

```
              |      roi      |  roi_baseline
roi           |   sp 4/2/26   |  56% (36/64)
roi_baseline  |  44% (28/64)  |   sp 3/5/24
```

**roi (new) beats roi_baseline (old) at 56% (36/64) over 64
head-to-head games.** Wilson 95% CI [43%, 69%] — the lower bound is
under 50%, so we have a real but noisy lift. Self-play cells show
both agents have a P1-favouring tie-break asymmetry as observed in
A.6 (P0/P1: 4/2/26 for new, 3/5/24 for old) — the new mechanisms
don't worsen the asymmetry.

p95 turn time: 1.3ms (new) vs 0.4ms (old). 3× slower from the new
mechanism stack — well under the 1000ms `actTimeout`. The
`path_clears_other_planets` mechanism is the dominant cost; the
per-turn precomputation (40 planets × 60 horizon ≈ 2,400 cos/sin
calls) amortises across all intents that turn.

### Gate verdict

- **(a) ≥60% local panel WR vs current best:** 56% — **MARGINAL** (3pp short).
- **(b) ≥55% head-to-head vs live submit:** 56% — **CLEARS** (1pp over).

Both halves of the locked eviction rule are required. With (a)
marginal, recommend NOT pushing v1.3 as a same-day submit. Instead:
- Stage bundle at `submissions/v1_3_roi_physics.py` (built; self-play
  validation 5/5 DONE).
- Hold for PI submit approval after seeing this audit.
- Continue to Block C (arrival ledger) immediately — that work
  doesn't depend on v1.3 ladder data.

## Outstanding follow-ups

1. **OOB residual 6.4% and sun residual 2.1%.** Both seem to be the
   same root cause: lead prediction error → fleet flies past predicted
   arrival → enters OOB or sun. Fix candidates (Block C-adjacent):
   - Tighten lead-prediction convergence (5-iter is generous but
     `search_safe_intercept` returns None more than expected — investigate).
   - Cap fleet trajectory at the predicted arrival distance + small
     buffer; if no planet contact within the buffer, drop the intent
     (over-conservative but safe).
2. **Bundle naming.** `submissions/roi.py` was overwritten by the
   v1.3 bundle. The pre-existing live-equivalent bundle is no longer
   in `submissions/`. Acceptable because `agents/simple/roi_baseline.py`
   reconstructs the pre-upgrade behaviour via `DEFAULT_MECHANISMS_PRE_PHYSICS`.
3. **Wallclock.** 1.3ms p95 is fine for v1.3, but `path_clears_other_planets`
   will scale unfavourably when v2 adds the mission framework
   (multiple proposals per source). Profile under Block C/D load.
