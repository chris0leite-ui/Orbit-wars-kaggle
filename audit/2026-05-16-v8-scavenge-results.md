# Session 2026-05-16 — v8_scavenge analytic → fast_sim K-step → reinforce; 0/32 → 71.9% PASS

Branch: `claude/recover-main-foundations-MV0e2`
Status: **PASS** — v8_scavenge beats v7_0 at Wilson lo 0.599 (n=64, 71.9%).

## TL;DR

Built `agents/v8_scavenge/main.py` per the PI-approved plan ("macro
moves on origin/main, no K-step rollout, idle-baseline scoring with
WorldModel as the analytic predictor"). Phase 1 falsified at 0/32 vs
v7_0 due to `WorldModel.fleet_target_planet`'s non-orbital ray-cast
mis-attributing in-flight fleets. Two structural fixes later, the
chooser passes the 55% gate decisively (Wilson lo 0.599 at n=64).

The arc:

| # | Variant | Change | vs nearest | vs v7_0 | Verdict |
|---|---|---|--:|--:|---|
| 1a | analytic | depth-0 WorldModel; `composite_capture_value`-style marginal | 62.5% (n=32) | 0/32 | FAIL |
| 1b | + eta fix | use `aim_orbiting`'s eta, not naïve dist/speed | — | 0/32 | FAIL |
| **2** | **Fix A — fast_sim K-step** | K=max(eta+2,15) rollout; F1+F2 favor at leaf; idle-baseline subtraction | 100% (n=16) | **12/32 = 37.5%** | INCONCLUSIVE |
| **3** | **M2 — reinforce** | extend target pool to my OWN threatened planets via `WorldModel.incoming_enemy_eta` | 100% (n=16) | **46/64 = 71.9%, Wlo=0.599** | **PASS** |

## Root cause analysis — three layered failures

The Phase 1 falsification (0/32) wasn't one bug — it was a stack of
three:

**Failure 1: eta off by 3-4× for orbital targets.** `_arrival_eta`
computed `distance/speed` based on target's CURRENT position. For
orbiting targets, `aim_orbiting` jointly solves angle AND eta —
the fleet flies an intercept arc to the target's FUTURE position,
which takes 3-4× longer than the naïve distance. On seed 0 turn 0,
naïve eta=5 vs `aim_orbiting`'s eta=21. Score functions were
querying `WorldModel` at the wrong turn, so capture predictions
were against a state that doesn't match actual arrival. **Fixed**
by replacing separate `_aim_angle` + `_arrival_eta` with one
`_aim_and_eta` call that returns both from a single
`aim_orbiting` solve.

**Failure 2: WorldModel's straight ray-cast misattributes
orbital fleet trajectories.** `WorldModel.fleet_target_planet`
documents: *"this is a non-orbiting ray-cast — it doesn't
account for target planets moving while the fleet is in
flight."* For an orbiting fleet aimed at the target's eta-
position, the straight-line ray-cast from the fleet's current
position to all planets' CURRENT positions doesn't predict that
the fleet will hit (the target has rotated out of the ray's
path). My agent's marginal-value scoring thus couldn't predict
its own captures. **Not fixable in WorldModel directly** — same
limitation that affected my Phase 1 marginal_value. Worked
around by switching to fast_sim K-step rollout (Fix A): the
simulator runs the EXACT physics, so by the leaf state the
fleet has either arrived (and combat resolved) or remains in
flight at a known position. No ray-cast prediction needed.

**Failure 3: undefended captures are wiped at mid-game.** With
Fix A in place we reached 37.5% but couldn't pass the gate.
Diagnosed via `scripts/diag_outcomes.py` on 16 games vs v7_0:
**13/13 losses were eliminations at mid-game** (turn 100-200,
ending with 0 planets vs opp's 20-38). Pattern: capture a
planet with 1-2 surviving ships (clean win on combat but tiny
garrison), leave it undefended, opp recaptures with their next
wave, my home falls next. The chooser never sent defensive
fleets to its own planets — there was no `targets` entry for
my own planets, so `_enumerate_candidates` never proposed
reinforcement. **Fixed by M2**: extend the target pool to
include MY planets that `WorldModel.incoming_enemy_eta`
predicts have an incoming enemy fleet. Reinforce sizing
in `_capture_size` sized to beat the predicted enemy wave at
predicted arrival.

## What turned the corner

M2 was the single biggest lift (37.5% → 71.9%, Δ+34pp). It
addressed exactly what the failure-mode diagnosis surfaced:
captures need defenders. The reinforce mechanic falls naturally
out of the existing chooser pipeline — `_capture_size` and
`_score_action` extend to my-target without architectural
changes; the fast_sim rollout confirms the defense actually
holds; the idle-baseline subtraction prevents over-firing
when no real threat exists.

## What WorldModel can and can't do for this chooser

**Can:**
- Predict per-planet ownership + garrison at any future step
  assuming current in-flight fleets resolve and no new launches.
- Surface incoming enemy ETAs per my planet
  (`incoming_enemy_eta`) — load-bearing for M2 reinforce.
- Size capture fleets via `ships_at(tgt, eta)` (production
  growth + existing reinforcement accounted for).

**Cannot:**
- Reliably attribute orbiting in-flight fleets to their target
  (straight ray-cast misses orbital intercepts).
- Predict opp's NEW launches (no opp speculation).

For the chooser this means WorldModel is the right tool for
sizing + threat detection but NOT for the leaf scorer's combat
prediction. The combat prediction MUST go through `fast_sim`
(K-step rollout) to handle orbital cases correctly.

## Recovered from `origin/main` (foundations we built ON)

| File | Purpose | Role in v8_scavenge |
|---|---|---|
| `lib/fast_sim.py` | parity-tested forward simulator | per-candidate K-step rollout |
| `lib/aim.py::aim_orbiting` | 5-iter lead + safe-intercept fallback | angle + eta in `_aim_and_eta` |
| `lib/orbit.py::is_orbiting` | predicate | branch in `_aim_and_eta` |
| `lib/fleet.py::speed` | comp-spec fleet speed | (transitively via fast_sim) |
| `lib/intent.py::World.from_obs` | frozen state view | input to WorldModel |
| `lib/world_model.py::WorldModel` | analytic timeline + `incoming_enemy_eta` | sizing + threat detection |

Not yet integrated (post-PASS optional lifts):
- `lib/planner.py::settle_plan` — joint Intent emission with
  arrival ledger (replaces greedy non-dogpile).
- `lib/missions/snipe.py::propose_snipe_missions` — global ROI
  ranking with priority-weighted scoring.
- `lib/opp_model.py::make_opp_policy` — opponent prediction
  for K-step rollout (currently strict idle).

## Diagnostic scripts written this session

- `scripts/diag_v8.py` — per-turn planet/ship trace + emitted
  action. Used to verify M2's emission patterns.
- `scripts/diag_outcomes.py` — win/loss classifier by game-
  length + endgame stats. The workhorse for diagnosing
  Failure 3 (elimination pattern).

## What was tried and failed

- **Depth-0 analytic chooser with marginal_value formula.** Three
  variants (composite-style weights, eta-discount, defense reserve).
  All 0/32 vs v7_0. Falsified before M2 was identified.
- **Multi-launch per source.** Marginal regression vs nearest
  (62.5% → 53.1%). Reverted; greedy non-dogpile stays for now.
  Will revisit when `settle_plan` is integrated.
- **eta-discount on capture credit** (penalize distant captures
  by `1 - 0.02 × eta`). Slight regression. Reverted; the natural
  `time_remaining` factor in F2 favor is sufficient.

## Next-session optional iterations

The PASS gate is cleared at n=64. Further lifts (if time permits)
and PI-decided priority:

1. **Panel calibration** — DONE. Results:
   - vs v7_0:       47/64 = **73.4%**, Wilson [0.615, 0.827] → PASS
   - vs v4_planner: 46/64 = **71.9%**, Wilson [0.599, 0.814] → PASS
   - vs v3.5.1:     42/64 = 65.6%,    Wilson [0.534, 0.761] → INCONCLUSIVE (1.6pp under 0.55 gate; passes audit's "Wlo ≥ 0.50 vs each" criterion)

   Panel verdict by strict 0.55 gate: INCONCLUSIVE (worst Wlo=0.534).
   By relaxed 0.50 criterion: PASS (all three Wlo > 0.50).

   **WALLCLOCK CONCERN:** p95=812ms / max=3116ms over 192 games.
   The max exceeds the 1000ms actTimeout — that turn would time out
   on the live ladder.

   **Re-run with MAX_HORIZON 50→30 (2458e85):**
   - vs v7_0:       24/32 = **75.0%**, Wilson [0.579, 0.867] → PASS
   - vs v4_planner: 25/32 = **78.1%**, Wilson [0.612, 0.890] → PASS
   - vs v3.5.1:     44/64 = **68.8%**, Wilson [0.566, 0.788] → PASS

   **Panel verdict: PASS (worst Wlo=0.566).** All three opponents
   clear the strict 0.55 gate. Wallclock tightened: p95=618ms,
   max=1494ms across 128 games — still has occasional outliers >
   1000ms but no longer the 3-second spikes from H=50.

2. **settle_plan emission**. Replaces greedy non-dogpile; allows
   useful same-turn follow-on launches.
3. **Opp-aware rollout** (single-step mirror or
   `lib.opp_model.make_opp_policy`). Currently strict idle —
   may have a remaining failure mode at the panel level.
4. **Scavenge ship sizes** (the original Phase-2 idea). Time
   fleets to arrive at predicted enemy-capture eta + δ.

## Submission state

`state/current.md` shows team rank 125/2667, best live μ=1064.4
(v7_pv), rolling-last-2 = geo + v7_pv. v8_scavenge submission
WOULD evict geo (older of the two). Expected μ on submit:
~1100-1180 (extrapolating from local 71.9% vs v7_0's μ=1081).
Decision NOT made — explicit PI sign-off required before
submit.

## Commits this session (on `claude/recover-main-foundations-MV0e2`)

```
82b5526 v8_scavenge M2: reinforce candidates — 37.5% → 71.9% vs v7_0 (PASS)
2a93062 v8_scavenge Fix A: fast_sim K-step rollout with idle-baseline subtraction
194d995 v8_scavenge: fix orbital-aim eta mismatch
8fe2840 v8_scavenge phase 1: analytic event-horizon chooser — FALSIFIED vs v7_0 (0/32)
```

4 commits. Working tree clean. Ahead of `origin/main` by 4, behind by 2.

## Did we touch `origin/main`?

**No.** All changes on `claude/recover-main-foundations-MV0e2`. No
modifications to `lib/`, `submissions/`, or `fast.py`. The 4-commit
chain documents the full arc including the falsification — useful
record for next session's "what we learned" review.
