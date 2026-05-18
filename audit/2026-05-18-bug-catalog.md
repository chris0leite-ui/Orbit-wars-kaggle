# Known bugs (2026-05-18 session — not yet fixed)

Catalog of bugs discovered while diagnosing the chooser's
under-emission and 4P-regression patterns. Each bug includes:
location, symptom, root cause, fix sketch, and current status.

**IMPORTANT — bugs #4, #13, #14 share a single root architectural
limitation** (single-step + me-static rollout). See concept doc
`knowledge-base/concepts/coordination-oracle-testing.md` for the
foundational framing: many bugs here are symptoms of ONE structural
issue. Oracle-based test methodology proposed there.

---

## #1 — "feasible-now still gets wait-N=1" speculative variant

**Location**: `agents/baseline/proposer.py:wait_then_fire_variants_forward`, line ~144
```python
if shortfall <= 0:
    wait_N = 1  # feasible-now still gets a wait-1 variant
```

**Symptom**: When src can already fire-now affordably, proposer
ALSO emits a wait_N=1 candidate (capture_size + 0 extra ships). The
chooser's Δ slightly favors wait_N=1 over fire-now (extra production
buffer), picks the wait, reserves src+tgt, emits NOTHING this turn.
Repeat → unbounded hoarding (Roman game: 59% idle turns).

**Status**: **FIXED** in backward grid (commit 533caca). Forward
path preserved via `BASELINE_WAIT_GRID=forward` env for rollback.

---

## #2 — Backward grid emits bare-capture (wastes accumulated ships)

**Location**: `agents/baseline/proposer.py:wait_then_fire_variants`
(backward path), original v2 implementation.

**Symptom**: Backward grid v2 launched only `cap_final` (defenders+1)
ships per wait variant. Combat residue: 1 ship. Trivially recaptured
by any small opp launch. 4P regression at 4/32 = 12.5%.

**Root cause**: We waited N turns to accumulate `src.ships + prod*N`
ships, then launched the bare minimum. Accumulation wasted.

**Status**: **FIXED** in v3 (commit f6d7eb2). Emits `final_fleet =
budget` (full accumulated). 4P retest: 5/32 = 15.6% — marginally
better but still FAIL. Indicates other 4P factors at play (see #4).

---

## #3 — `capture_size` reinforce: opp force is static, ours accrues

**Location**: `agents/baseline/proposer.py:capture_size` (reinforce
branch, lines 78-100)

```python
my_garrison = float(tgt.ships) + float(tgt.production) * enemy_eta  # accruing
enemy_strength = best_enemy_planet_ships  # STATIC
shortfall = enemy_strength - my_garrison + 1
```

**Symptom**: Asymmetric prediction — OUR planet accumulates
production over `enemy_eta`, but opp's "potential launch" strength
is treated as the OPP planet's current ship count without growth.
Result: shortfall is usually negative → 0 reinforce candidates → we
don't preemptively defend planets that will eventually be attacked.

**Root cause**: Speculative-launch math is asymmetric.

**Fix sketch**: enemy_strength should accrue too:
```python
enemy_strength_at_eta = best_enemy_ships + best_enemy_prod * enemy_eta
```

**Status**: **NOT YET FIXED**. Identified in asdf game step 22-31
trace. (Note: in the asdf game specifically, the lethal threat was
in-flight not potential — see #11 for that bug.)

---

## #4 — Drain-frontier: chooser depletes exposed defensive planets

**Location**: `agents/baseline/chooser_trajectory.py:score_candidate_v4`
(but really a leaf-scoring + horizon issue)

**Symptom**: Chooser uses any planet as a launch source, regardless
of whether that planet is exposed to opp attack. After launching
FROM the planet, ship count drops dangerously low; if opp's threat
arrives 10-30 steps later, we can't defend. Asdf game: P15 had 25
ships → launched 18 (step 22) → 7 ships → eventually captured. Pattern
recurs 11 times in that 127-step game.

**Root cause**: chooser's leaf at horizon=25 doesn't see threats that
materialize at step 40+. Single-step Δ blind to "this source is also
our only defender of itself / nearby planets."

**Fix sketch**: Multiple angles possible:
1. Longer horizon (cost: wallclock)
2. Cheap-rank penalty for launches from planets with `time_to_enemy_threat`
   within (eta + buffer)
3. Reserve floor: never launch more than `src.ships - reserve_for_defense`
4. Better leaf valuation that weights post-capture residue at SOURCE

**Status**: **NOT YET FIXED**. Identified in asdf game analysis.

---

## #5 — Banding dedup collapses ship-count variants within wait_N

**Location**: `agents/baseline/proposer.py:propose`, line ~255-261

**Symptom**: banding key `(src_id, tgt_id, wait_band)` keeps only
one entry per band. Cheap_delta for capture-success is IDENTICAL
regardless of ship count, so banding picks whichever variant is
inserted first. Multiple ship-count variants at same wait_N
collapse to 1.

**Implication**: For wait variants, can't have both "bare capture"
and "full budget" candidates. We had to pick one in v3 (chose budget).

**Status**: **WORKED AROUND** in v3 by choosing budget. Not a true
fix; if we need finer ship-count breadth, need to change banding
key.

---

## #6 — Tier 1 (lite_greedy vulnerability term): rollout-opp too smart

**Location**: `lib/opp_model.py:lite_greedy_policy`, attempted in
commit 3bba9c4 (reverted in 038a957)

**Symptom**: Adding `defenders_at_eta - 2` divisor to opp scoring
made the rollout's simulated opp correctly punish OUR over-extension.
But our chooser was over-fit to a weak opp model — with smarter opp,
EVERY candidate scored Δ ≤ 0 (we expected opp to counter-attack
every weakness). Chooser emitted ~0 launches. A/B: 7/32 = 21.9%.

**Root cause**: smarter opp + thin-residue launches (#2) =
catastrophic conservatism. The two changes are coupled — need both
or neither.

**Status**: **REVERTED**. Note: not a "wrong direction" — exposes
the chooser's wait-bias (#1) and thin-residue (#2). Worth retrying
AFTER those are fixed.

---

## #7 — Joint v3 4P regression (over-commitment with 3 opps)

**Location**: `agents/baseline/chooser_trajectory.py:choose_trajectory`
(joint enumeration, gated by num_seats <= 2)

**Symptom**: Joint candidates commit 2 sources to capture 1 target.
In 2P (1 opp), 59% winrate. In 4P (3 opps), 12.5% first-place —
catastrophic. The 3 opps simultaneously exploit our 2 weakened
sources + 1 thin-residue capture.

**Status**: **WORKED AROUND** with 2P-only gate (commit f14eb46).
Shipped as 52766596. The structural 3-opp exploit issue is not
fixed; would need #4 + #6 fixed first to retry.

---

## #8 — Spatial leaf 4P regression

**Location**: `agents/baseline/value.py:favor_hybrid_spatial`
(opt-in, default off)

**Symptom**: positional pull in leaf broke A2 weakness-exploitation
in 4P. A/B: 3/32 = 9.4% first-place.

**Status**: **WORKED AROUND** with 2P-only gate (commit 558bd61).
Default off. Same root cause as #7 (3 opps × any aggression =
catastrophic).

---

## #9 — H1 post-chooser idle-drain: forced emissions break calibration

**Location**: `agents/baseline/main.py:drain_idle_rear` (opt-in,
default off)

**Symptom**: Force-emitting reinforce launches from idle rear sources
locked ships in flight, removed defensive optionality. A/B:
11/32 = 34.4%.

**Status**: **DEFAULT OFF**. Code preserved; not the right axis.

---

## #10 — Banding dedup loses orbital-target wait variants

**Location**: `agents/baseline/proposer.py:propose` (interaction
with `aim_and_eta` for orbital targets)

**Symptom**: `aim_and_eta` pre-rotates orbital targets for wait_N >
0 (to handle orbital position at fire time). But banded dedup
collapses these.

**Status**: **NOT INVESTIGATED**. Theoretical risk; not observed in
data yet.

---

## #15 — 🚨 CRITICAL — composite_capture_value doesn't credit captures it CAUSES

**Location**: `lib/value_heads.py:composite_capture_value` line 266-270:

```python
pred_owner = model.owner_at(target.id, eta)
pred_ships = model.ships_at(target.id, eta) or 0.0
if pred_owner == my_id:
    # "Already ours — reinforcement; no extra credit (already in base)."
    continue
```

**Symptom**: when our fleet IS in flight to a target it will capture,
`model.owner_at(target.id, eta)` returns `my_id` because the model's
simulation includes our fleet's arrival. The composite then treats
this as "over-reinforcement" (target already ours) and skips the
capture-bonus credit. Result: composite returns ONLY the base
ship-delta term; captures the agent makes get ZERO credit at the
leaf.

**Discovered**: 2026-05-18 while debugging the trivial 100-vs-5
sanity oracle. Even with 100 ships vs 5 opp ships in a clean
synthetic setup, agent emits nothing. Trace:
```
step 0: my=1pl, opp=1pl, my_sh=100, opp_sh=5, favor=95.0
step 1: my=1pl, opp=1pl, my_sh=1 (post-launch), in_flight=100, favor=95.0
step 11: my=2pl, opp=0pl (captured), my_sh=95, opp_sh=0, favor=95.0, done
```

Favor is 95.0 at ALL steps including post-capture. The capture
delivers 1 production gain over 489 remaining steps (~24 expected
bonus at capture_weight=0.05) but composite credits ZERO.

**Root cause**: chicken-and-egg with predictive simulation. The
model SIMULATES with the fleet included; the composite then asks
"will the target already be ours at arrival?" — yes, BECAUSE OF
THIS FLEET. The check inverts the causal direction.

**Why this devastates the chooser**:
- baseline (idle, no launch): favor = N
- candidate (launch + capture): favor = N (capture not credited)
- Δ = 0 → emit gate (`> 0`) rejects ALL captures

The chooser only emits when Δ > 0, but captures give Δ = 0 from
composite, so the ONLY positive Δ moves are pure ship-balance
plays (reinforce-against-threat — when our planet would otherwise
fall WITHOUT our action). Anything offensive scores 0.

This is THE root cause of bug #13 (can't finish), and a major
contributor to bugs #4 (drain-frontier), #7 (joint 4P fail), and
#14 (asymmetric rollout). Many "we don't emit launches that
should obviously emit" symptoms trace here.

**Fix sketch**: check the COUNTERFACTUAL — what would happen if
THIS fleet weren't in flight?

```python
# Build a ledger WITHOUT this specific fleet
counterfactual_arrivals = [a for a in model.ledger.get(target.id, [])
                           if a != (eta, int(f.owner), int(f.ships))]
# Predict ownership without us
counterfactual_owner = simulate_planet_timeline(
    target, counterfactual_arrivals, eta + 1
)["owner_at"][eta]
if counterfactual_owner != my_id:
    # Without this fleet, target wouldn't be ours → this fleet
    # CAUSES the capture → bonus
    delta += capture_weight * production * time_remaining
else:
    # Even without this fleet, target ends up ours (someone else
    # already in flight or production-capture) → no bonus
    continue
```

Or simpler heuristic: if target.owner != my_id at observation time
AND our fleet's ships > raw `target.ships + production * eta`
(non-modeled prediction), credit the bonus.

**Status**: **NOT YET FIXED**. Discovered late in 2026-05-18
session via synthetic oracle testing (the methodology PI proposed
worked — the trivial sanity test surfaced this immediately).

**Severity**: 🚨 CRITICAL. Likely THE single biggest fixable
issue. Estimated impact: could enable proper capture-driven Δ
across all coordination scenarios. Higher priority than #14.

---

## #14 — ⭐ ROOT — Asymmetric rollout: opp plays, we don't

**Location**: `agents/baseline/chooser_trajectory.py:score_candidate_v4`
rollout loop (line ~370-376):

```python
for t in range(horizon):
    if snap.fake_env.done:
        break
    actions = opp_actions_for_snap(snap, me, num_seats)  # opp REACTS each step
    if t == int(wait_N):
        actions[me] = [[int(src.id), float(angle), int(ships)]]  # we move ONCE
    snap = fs_step(snap, actions, in_place=True)
```

**Symptom**: in the leaf rollout, opp is simulated REACTIVELY (calls
`lite_greedy_policy` every tick), but OUR agent stands still after
our single launch. The chooser is essentially asking "if I make this
ONE move AND THEN STAND STILL FOR 25 TICKS, what's the worst opp
can do?"

This produces a systematic worst-case bound. The chooser refuses
any launch that opp could exploit, because in the rollout we agreed
not to defend.

**Evidence**:
1. Dekaineko step 150: launching P0→P8 exposes P0 to opp counter.
   Real game we'd reinforce P0 from one of 22 other planets. Rollout
   shows us doing nothing → P0 falls → chooser scores Δ ≤ 0.
2. Asdf game pattern: chooser drains planet P15 because rollout
   doesn't show us defending P15 later either.
3. Joint v3 4P regression: joint commits 2 sources; rollout shows
   3 opps exploiting both while we do nothing.

**Root cause**: classic asymmetric self-play simulation. The
opponent's reactive policy is implemented; ours is not.

**Fix sketches**:

1. **Mirror lite_greedy for me in the rollout**: at each tick,
   also call `lite_greedy_policy(snap.state[me].observation)` and
   inject the result. We become reactive too. Honest game-theoretic
   simulation.
   - Concern: our chooser is meant to be SMARTER than lite_greedy.
     Using lite_greedy as our rollout policy under-rates our skill.
     But: it's still better than "we do nothing."

2. **Mirror the proposer+chooser stack for me** (heavier): in the
   rollout, run a cheap version of our own decision pipeline at each
   tick. Very expensive — each leaf eval becomes recursive.

3. **Predict our future captures via composite_capture_value**: at
   the leaf, INCLUDE the value of OUR in-flight fleets' captures
   beyond what's already credited. Captures the "we'll defend"
   intuition without simulating it.

4. **Heuristic overrides for clear oracle scenarios** (bug-class
   patches): see `coordination-oracle-testing.md`.

**Status**: **NOT YET FIXED**. THE structural fix. Implementing #1
or #2 above would dissolve bugs #4 and #13 simultaneously.

**Severity**: critical. Likely the single biggest architectural
improvement available. Roman/dekaineko/asdf losses all trace here.

---

## #13 — Chooser stalls in dominant positions ("can't finish the game")

**Location**: `agents/baseline/chooser_trajectory.py:score_candidate_v4`
emit gate (`score > 0.0` strictly) interacting with leaf scoring in
overwhelming-advantage states.

**Symptom**: When we're winning by a large margin (we hold 23 of 24
planets, 3000+ ships; opp holds 1 planet, ~100 ships), the chooser
emits NOTHING for many turns. Opp's last planet accumulates ships
unmolested until eventually we DO emit (often after opp has
accumulated 100+ ships).

**Evidence**: dekaineko game (episode 76951352) steps 130-160:
- We have 23 planets, ~3000-4000 ships
- Opp has 1 planet (P8) with 52→112 ships
- 4 candidates aimed at P8 each turn with positive cheap_delta
- Chooser scores in v4: P0→P8 Δ=-32, P16→P8 Δ=0, P4→P8 Δ=-1200
- All ≤ 0 → emit gate rejects → EMIT NOTHING
- 30 turns of idle while opp accumulates
- Finally emit at step 161 (we still win, but unnecessary delay)

**Root cause**: leaf favor at horizon=25 dominated by base ship
balance (3274 my - 92 opp). Any launch costs ships from my source
(temporarily reduces my_ships count via ship-loss-to-bounce-or-
combat). Capture bonus (`0.05 × prod × time_remaining ≈ 35`) is
small relative to baseline magnitude. Net Δ ≤ 0 because:
- ship-balance loss from launch (combat) > capture bonus
- OR fast_sim's `done` flag fires early when we capture and game
  appears "decided" — terminal leaf has no marginal benefit

**Fix sketch** (multiple angles):

1. **Emit ANY positive-prod capture in dominant positions**: gate
   could be "Δ > 0 OR (we win and capture has any prod)". Cap at
   1-2 per turn to avoid runaway.

2. **Re-weight capture bonus when we dominate**: scale capture_bonus
   by `(opp_planet_count == 1) * eliminate_bonus`. Captures that
   would ELIMINATE opp are worth far more than incremental.

3. **Tempo-pressure term in favor**: explicit "time to game end"
   reward — game ending sooner is better when we're winning, so
   capturing the last opp planet quickly is positive even if leaf
   ship-balance is unchanged.

4. **Force-emit cleanup**: when opp has only 1 planet AND we have
   overwhelming force, emit one launch per turn from the closest
   sufficient src.

**Status**: **NOT YET FIXED**. Cosmetically we still win this game,
but the stall is exploitable: a smarter opp could counter-launch
during our hesitation and recapture lost ground.

**Severity**: medium. Doesn't lose easy games but wastes 20-30 turns
on close-to-finished games. Could lose IF opp counters effectively
during the stall.

---

## #12 — `capture_size.enemy_inflight` window too narrow for multi-wave attacks

**Location**: `agents/baseline/proposer.py:capture_size` (reinforce
branch, line ~83-87):

```python
enemy_inflight = sum(
    ships
    for (eta_arr, owner, ships) in model.ledger.get(int(tgt.id), [])
    if owner != me and eta_arr <= enemy_eta + 1
)
```

**Symptom**: When an opp launches MULTIPLE fleets at the same target
(staggered wave attack), capture_size only counts those arriving
within `enemy_eta + 1` of the EARLIEST. The post-fix ledger for the
asdf game at step 37 correctly contains both `(eta=2, ships=42)`
and `(eta=4, ships=65)`. enemy_eta = 2 (earliest). The window
`enemy_eta + 1 = 3` excludes F23 at eta=4. shortfall computed only
against the 42-ship fleet → returns 0 → no reinforce candidate.

**Discovered**: AFTER fix #11 landed. The ledger improvement
exposed this asymmetry that was previously masked.

**Fix sketch**: widen the window to include staggered waves:

```python
# Sum all opp ships arriving in the next ~10-20 steps, not just
# enemy_eta + 1. Or: sum ALL opp ships in ledger (let the chooser
# decide whether the late ones are reachable).
enemy_inflight = sum(
    ships for (eta_arr, owner, ships) in model.ledger.get(int(tgt.id), [])
    if owner != me and eta_arr <= enemy_eta + WAVE_LOOKAHEAD  # 10? 20?
)
```

Or a more principled fix: simulate the timeline at the planet
including ALL inbound fleets, find the MAX shortfall over time.

**Status**: **NOT YET FIXED**. Critical follow-on to #11.

---

## #11 — ⚠️ `fleet_target_planet` ray-cast misses ORBITING targets

**Location**: `lib/world_model.py:fleet_target_planet` (ray-cast
attribution)

**Symptom**: A fleet aimed at an orbiting planet ray-casts in a
straight line. The orbiting planet might not be at its expected
straight-line position; ray-cast returns NONE. The ledger then
doesn't include the fleet's threat. By the time the fleet is close
enough to be re-attributed correctly, we can't defend.

**Evidence**: asdf game (76947663) step 37:
```
F23: owner=0 ships=65 from=P18 → target=NONE eta=None
```
F23 was launched at step ~36 toward P15 (orbiting). At step 37, ray-cast
returns no target. At step 40 (when fleet is 1 step away from arrival),
ray-cast finally catches P15 → ledger updates → but too late to react.

**Root cause**: Static ray-cast doesn't model planetary orbits.
Fleet's trajectory crosses where the target WILL BE at arrival time.

**Fix sketch**:
1. For orbital targets, use lead-aim model to predict where the
   planet will be at the fleet's straight-line ETA
2. Test if the fleet's trajectory + planet's predicted position
   intersect at some step
3. Specifically: iterate t in [0, MAX_HORIZON], compute fleet's
   position at t and each orbital planet's position at t, check
   collision

**Status**: **NOT YET FIXED**. Critical bug. Causes systematic
late detection of in-flight threats targeting orbiting planets.
This is the SPECIFIC bug that caused us to lose P15 in the asdf
game (and likely many other "fall-then-recapture" events).

**Likely-implicated friction tag** (existing):
`predict-fleet-fate-orbital-target-mis-attribution` (search
audit/friction.md for related entries).

---

## Cross-bug priority ranking

For fixing in order of (expected impact × cleanness):

1. **#11** orbital ray-cast (clean bug, high impact — many
   fall-then-recapture events)
2. **#4** drain-frontier (strategic depth fix — biggest architectural
   win)
3. **#3** asymmetric reinforce sizing (clean math fix)
4. **#6** retry lite_greedy AFTER #2 + #11 are clean
5. **#10** orbital wait variants (not observed yet)

Bugs #1, #2 are already fixed. Bugs #5, #7, #8, #9 have workarounds.

## When to revisit each

- After 52766596 settles (~6h / 50 games), have ladder feedback
  on joint v3.
- Before next submission attempt, fix #11 (cheap, high impact).
- #4 needs careful design — could combine with #6 retry.

This file is the bug-tracking source of truth. Update as bugs are
discovered/fixed.
