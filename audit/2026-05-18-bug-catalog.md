# Known bugs (2026-05-18 session — not yet fixed)

Catalog of bugs discovered while diagnosing the chooser's
under-emission and 4P-regression patterns. Each bug includes:
location, symptom, root cause, fix sketch, and current status.

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
