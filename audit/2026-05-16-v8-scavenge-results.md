# Session 2026-05-16 — v8_scavenge chooser: from 0/32 to 71.9% PASS vs v7_0

Branch: `claude/recover-main-foundations-MV0e2`
Final state: **PASS** — Wilson lo 0.599 ≥ 0.55 gate at n=64 vs v7_0.
Files: `agents/v8_scavenge/main.py` (~270 LoC), `scripts/diag_v8.py`,
`scripts/diag_outcomes.py`.

## TL;DR

The plan called for a depth-0 analytic chooser using `WorldModel`
predictions for end-state scoring. That FALSIFIED at 0/32 vs v7_0.
Two structural fixes recovered the result:

1. **Fix A** (commit `2a93062`): replaced analytic depth-0 with a
   per-candidate `lib.fast_sim` K-step rollout (K = max(eta+2, 15))
   + F1+F2 favor at the leaf + idle-baseline subtraction at the same
   horizon. Got to **12/32 = 37.5%** (matched the bootstrap session's
   level; CIs overlapped 43.8%).
2. **M2 reinforce** (commit `82b5526`): extended the candidate target
   pool to include MY OWN threatened planets (via
   `WorldModel.incoming_enemy_eta`). Sized to beat the predicted
   strongest enemy wave at predicted arrival. **46/64 = 71.9%, Wilson
   [0.599, 0.814] → PASS.**

Diagnostic finding that unlocked M2: 13/13 losses vs v7_0 in n=16 were
ELIMINATION at mid-game (we end with 0 planets, opp has 20-38). Not
outscored on production — wiped out. Pattern: capture a planet with
1-2 surviving ships, leave it undefended, opp recaptures, repeat
until home falls too. Reinforce directly addresses it.

## The arc

| Phase | Change | vs nearest | vs v7_0 | Verdict |
|---|---|--:|--:|---|
| Plan-Phase-1 v8.1 | analytic depth-0 + custom weights | 62.5% (20/32) | 0/32 | FAIL |
| v8.2 | + multi-launch + eta-discount | 53.1% (17/32) | n/t | FAIL |
| v8.3 | revert eta-discount | 53.1% (17/32) | n/t | FAIL |
| v8.4 | composite_capture_value weights | n/t | 0/32 | FAIL |
| v8.5 | + defense reserve | n/t | 0/32 | FAIL |
| **eta fix** (194d995) | use `aim_orbiting`'s eta (not naïve dist/speed) | n/t | 0/32 | FAIL (correct fix, wrong scoring) |
| **Fix A** (2a93062) | fast_sim K-step rollout + F1+F2 favor + idle baseline | 100% (32/32) | **37.5% (12/32)** | improved |
| **M2** (82b5526) | + reinforce targets (own threatened planets) | 100% (16/16) | **71.9% (46/64), Wlo=0.599** | **PASS** |

## What went wrong with the analytic depth-0 approach

**Failure 1 — orbital eta mismatch.** `_arrival_eta = distance / speed`
based on target's CURRENT position. For orbiting targets,
`aim_orbiting` aims at the FUTURE position (lead prediction), and
the fleet flies an intercept arc that's 3-4× longer than naïve.
Marginal-value queried `WorldModel.ships_at(tgt, naïve_eta)` — the
wrong turn. Symptom: agent fires at tgt=16 with naïve eta=5, scores
+24 from "Δfavor", but fleet's actual eta is 21. Captures (when
they happen) are at the wrong future state from what was scored.
**Fix**: combined `_aim_and_eta` returning both from one
`aim_orbiting` call. Closed Failure 1 but didn't lift the score.

**Failure 2 — WorldModel ray-cast doesn't predict orbital captures.**
`lib.world_model.fleet_target_planet` uses a NON-orbital straight
ray-cast (documented limitation). For an orbital fleet/target combo,
the ray-cast from current fleet position to current planet positions
MISSES the orbital target whose CURRENT position is ahead of the
fleet's straight-line direction (the fleet is AIMED at where the
target WILL be). So WorldModel predicts "fleet hits no planet"
instead of "fleet captures target T at eta". The chooser's prediction
of in-flight captures is unreliable for the dominant case (most
non-comet planets orbit).

**Failure 3 — strategic.** Even with perfect predictions, depth-0
can't compare "fire now (capture prod=1 at eta=21)" against "wait
N turns then fire (capture prod=3 at eta=12)". The K-step rollout
naturally captures this: idle-baseline at K=15 grows production
linearly, and a higher-prod capture leaves the leaf at a much
better F2.

## Fix A — fast_sim K-step rollout

Per candidate (src, tgt, ships, angle):
- Clone snap_base.
- Apply my action [src.id, angle, ships] for step 1 (rest idle).
- Idle for K-1 more turns where K = max(eta + SIM_SETTLE_TURNS=2,
  MIN_HORIZON=15).
- Evaluate F1+F2 favor at the leaf.
- Δ = leaf_favor − baseline_favors[K] where baseline is precomputed
  all-idle for horizons 0..MAX_HORIZON=50.

The simulator handles orbital motion, swept-pair collisions, sun
crossings, and combat resolution EXACTLY (62/62 parity tests vs the
env). No more ray-cast attribution issues — the leaf's planet
ownership directly reflects whether my fleet captured.

Performance: p50=57ms, p95=265ms, max=792ms (after the
`WALLCLOCK_BUDGET_MS=750` deadline check between sources).

## M2 reinforce — what unlocked the PASS

Failure mode from `scripts/diag_outcomes.py` on n=16 vs v7_0:
- 13/16 losses, all by ELIMINATION (my_planets=0 at episode end).
- 12 losses MID-game (turn 100-200), 1 LATE, 0 EARLY.
- Median opp end-state: 25 planets, 2500-4500 ships.
- The 3 wins were also by elimination (we wipe opp).

Implication: this is a snowball game. Once one side starts losing
planets, they can't recover. The losing side's captures are
undefended (1-2 ships post-combat); opp's next wave reclaims them
trivially.

M2: extend the target pool to MY OWN planets that have a predicted
incoming enemy fleet (via `WorldModel.incoming_enemy_eta`). The
reinforce branch of `_capture_size` sizes the defensive fleet to
beat `enemy_ship_sum − (my_garrison_at_eta + prod × enemy_eta) + 1`.
fast_sim downstream confirms whether the defense actually holds; the
idle-baseline subtraction means defensive launches only fire when
their Δfavor is positive (i.e., the planet would have fallen without
the reinforce AND the fleet successfully holds).

Side effect: the chooser now has a richer action space (capture
new + reinforce existing), so even on early-game turns with one
home + one captured planet, it can split production between
expansion and defense. The 100%-vs-nearest sanity check confirms
no regression from the new behaviour.

## Recovered from `origin/main` (foundations we built ON)

| File | Purpose | Use |
|---|---|---|
| `lib/fast_sim.py` | parity-tested simulator | per-candidate K-step rollout |
| `lib/aim.py::aim_orbiting` | 5-iter lead-aim + safe-intercept | angle + eta jointly |
| `lib/orbit.py::is_orbiting` | predicate | branch on aim path |
| `lib/fleet.py::speed` | ship-count → speed | eta estimation |
| `lib/intent.py::World.from_obs` | frozen state view | input to WorldModel |
| `lib/world_model.py::WorldModel` | analytic timeline | sizing + threat detection |

Not yet pulled (deferred to later phases):
- `lib/planner.settle_plan` — would replace greedy non-dogpile emit.
- `lib/missions/snipe.propose_snipe_missions` — Mission proposers.
- `lib/value_heads.composite_capture_value` — alternative leaf scorer.

## Diagnostic scripts written this session

- `scripts/diag_v8.py` — per-turn ledger of v8 vs any opponent.
- `scripts/diag_outcomes.py` — per-game win/loss classification with
  endgame stats. Surfaced the elimination pattern that unlocked M2.

## Open questions / next iterations

Per `/root/.claude/plans/curried-gliding-prism.md`:
- **Phase A — panel calibration** (RUNNING): vs v4_planner +
  v3.5.1 at n=32 each. PASS criterion: Wilson lo ≥ 0.50 vs EACH.
  Closes the `local-overpredict-2x` friction
  (state/current.md: v3.5.1 5/12 and geo v3.1 5/14 both passed
  single-opp A/Bs but regressed live).
- **Phase B (optional)**: replace greedy non-dogpile with
  `settle_plan` for the joint emission.
- **Phase C (optional)**: opponent-aware rollout (mirror instead of
  idle) — only if Phase A reveals brittleness.
- **Phase D (optional)**: explicit scavenge ship sizes (the
  original Phase 2 idea — was deferred because M2's reinforce
  addresses the defensive half of the same prediction).
- **Phase E**: submission decision (PI gate; rolling-last-2 in
  `state/current.md` means a submit evicts geo and keeps v7_pv).

## Compute spent / wallclock budget

This session arc:
- Phase 0 (sanity + pytest): ~15 min.
- Phase 1 falsification (5 variants, ~6 A/Bs): ~30 min.
- Fix A development + A/Bs: ~20 min.
- Diagnosis with `diag_outcomes.py`: ~5 min.
- M2 implementation + A/B n=64: ~10 min.
- Total to PASS: ~80 min.

p95 turn-ms with M2: 530ms; max 1139ms with one outlier (likely from
a turn where many planets are threatened and reinforce enumeration
runs into the wallclock guard).

## NO Claude session URLs in commits or audit (Rule 39).
