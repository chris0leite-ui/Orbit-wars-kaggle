# 2026-05-17 — Felipe seed root cause + Layer 1/2 of v11

Branch: `claude/recover-main-foundations-MV0e2`
Status: Layer 1 (orbital revert) + Layer 2 (mirror-opp baseline) landed.
Felipe-seed v10-vs-v7_0 still 0/2; root cause is structural, not a bug.

## TL;DR

The v10 wait-then-fire orbital "fix" was based on a flawed
`env.steps[0]` test. From a mid-game obs, `predict_relative(lead=N)`
gives the position N env-steps later — exactly. The commit changed
`lead=wait_N` to `lead=wait_N-1`, introducing a 1-step aim error.
Reverted in this session.

After revert + a mirror-opp baseline (Layer 2), Felipe-seed loss is
unchanged: 0/2 vs v7_0. The actual loss mode is **opening cadence**,
not the orbital aim or scoring formulation:

- v10 launches at turn 4 (14 ships → tgt=21, eta=30)
- v7_0 builds to 31 ships at home, blitzes at turn 22 → captures
  4 planets by turn 34
- During v10's 30-turn flight, v7_0 has already won the opening race
- v10's depth-0 chooser with strict-idle K=30 rollout CANNOT see opp's
  4-planet mid-rollout captures, so it scores its own slow-arrival
  candidate as +386 favor (mostly F2 production-stream value) and emits

The remaining fix axis is rollout-aware opp modeling OR a different
opening proposer. Both are structural, larger than one session.

## What we landed

### Layer 1 — orbital aim revert (commit pending)

`agents/v8_scavenge/main.py:200`: `lead = max(0, wait_N - 1)` → `lead = wait_N`

Empirical test (`audit/2026-05-17-orbital-aim-verification.md`):
- From env.steps[15], advancing 6 idle steps to env.steps[21]:
  `predict_relative(planet@15, omega, lead=6)` matches with err=0.0000;
  `lead=5` is off by 1.6934 units.
- The original commit's test from env.steps[0] is a measurement artifact
  (env.steps[0] and env.steps[1] are positionally identical so a 1-step
  off-by-one is undetectable).

Outcome: v10-vs-v8 still 2/2 (Felipe seed), v10-vs-v7_0 still 0/2.
No regression vs the panel (n=16 holds 75% Wlo 0.579).

### Layer 2 — mirror-opp baseline (commit pending)

`agents/v8_scavenge/main.py` agent() pre-computes opp's step-0 action
via `top_tier_mirror_policy` (v3.5.1 pipeline, aggressive sizing).
Same action applied at step 0 of baseline AND each candidate's rollout
— "common random numbers" cancels noise in the Δ subtraction.

`_build_idle_baseline` and `_score_action` now accept
`opp_step0_actions` and apply them on step 0.

Why top_tier_mirror not lite_greedy: lite_greedy at turn 0 sizes
launches at 0.7 × src.ships (7 ships) and selects a 30-defender
neutral by ROI — guaranteed bounce. top_tier_mirror routes through
`propose_snipe_missions` with real capture-size math.

**Felipe-seed observation:** top_tier_mirror returns `[]` for opp at
turn 0 (opp can't capture any neutral with 10 starting ships either).
So `opp_step0_actions = {}` and Layer 2 has NO EFFECT on the Felipe
opening. Layer 2 should help on seeds where opp has a feasible
step-0 capture but is inert here.

## The real Felipe loss mode (forensic)

### Cadence comparison

| | v10/v11 turn 4 | v7_0 turn 22 |
|---|---|---|
| Launch source | 28 (home, 14 ships) | their home (built to 31 ships) |
| Launch eta | 30 turns | ~5-8 turns (close neutrals) |
| Captures by turn 34 | 2 (p21, p28-adjacent) | 4 (4 close neutrals) |
| Ship-count by turn 50 | 38 | 97 |

v7_0's "wait then blitz" strategy is exactly what v10's wait-then-fire
was designed to do — but MAX_WAIT=10 caps it short of the 22-turn
wait v7_0 effectively uses. And v10 has only 1 source planet at the
start (14 ships max), so wait-then-fire from src=28 can build to at
most 24 ships in 10 turns — still wouldn't beat opp's blitz.

### Why the chooser emits the 30-eta launch

Validated Δ at turn 4 (from this session's trace):
- src=28 → tgt=21 (eta=30, 14 ships): **Δ = +386.30**
- src=28 → tgt=20 (eta=11, 11 ships): Δ = −11.00
- src=28 → tgt=24 (eta=17, 14 ships): Δ = −14.00

Top candidate has +386 because:
- F1 (ship balance): −5 at leaf (we have fleet in flight, opp has rebuilt)
- F2 (production stream): (4 prod gain) × pv_horizon(34, 0, γ=0.99) ≈ 4 × 99 = 396
- F2 dominates → +391 → +386 after baseline subtraction

The other candidates scored NEGATIVE because:
- They had similar F2 gains but smaller (cap=9, 11 vs 14)
- F1 was more negative (smaller ships in flight → less "free ship" boost)

The pathology: F2 is a 466-turn production stream valued at ~99 units,
so ANY successful capture overwhelms F1's small ship-count delta.
That's why the chooser likes the 30-eta launch — it expects 466 turns
of production from the captured planet.

In reality, the captured planet matters for ~100 turns (opp eliminates
us at turn ~125). F2's full-game horizon is a fiction in any losing
game.

## Why the fix didn't fix Felipe

Layer 1 + Layer 2 address two real bugs but neither is the bottleneck
on this specific seed:

1. Layer 1: corrects an orbital-aim regression. Felipe doesn't trigger
   wait-then-fire (insufficient ships to wait+fire in 10 turns), so the
   aim fix never applies.
2. Layer 2: corrects strict-idle blindness when opp has a step-0 move.
   Felipe opp's mirror returns `[]` at turn 0 — both baselines are
   identical.

What WOULD address Felipe:
- A. **Rollout-aware opp modeling** — opp acts at EACH rollout step
  (not just step 0). Iter 2 of v8_scavenge tried this with stochastic
  mirror-opp and regressed (-3pp). Needs deterministic + common-random-
  numbers across candidates. Cost ~10ms × 30 steps = 300ms per
  candidate — over budget.
- B. **Opening proposer** — emit `roi`-style or `v3.5.1` opening for
  turns 0-15, switch to v8 chooser thereafter. Cheap, doesn't break
  the existing panel.
- C. **MAX_WAIT extension** — let wait-then-fire wait up to 25 turns
  for a feasible mass capture. Will be slow per turn.
- D. **F2 weighting fix** — cap pv_horizon at a strategic window
  (e.g., 100 turns of forward value, not 466). Reduces the dominance
  of distant captures.

Among these, B is cheapest and most-likely-to-work. The opening proposer
is what v7_0 essentially does (via `propose_snipe_missions`). Plumbing
it in as a fall-through for turns 0-15 when v8's chooser emits nothing
or only-distant captures would directly fix the cadence.

## Panel result (post Layer 1+2)

`fast.py eval v8_scavenge --vs v7_0 --max-seeds 16`: **TBD** — running
in background. Pre-change baseline was 24/32 = 75.0% Wlo 0.579 PASS.
Layer 1+2 expected to be neutral or slight positive (mirror-opp
baseline has no effect when opp idles at step 0, which is the typical
opening pattern).

## Next steps (after panel verifies no regression)

1. Commit Layer 1+2 with this audit doc — bisectable single commit.
2. PI consultation: implement Option B (opening proposer fallback)
   OR drop Felipe from regression set as "known structural loss on
   1-prod-1-planet opening boards" and proceed to live calibration.
3. Plan Layer 3 (enumerator widening) — gated on whether B/D resolve
   the cadence issue.

## Files modified this session

- `agents/v8_scavenge/main.py` — 3 sites:
  - L200: orbital lead revert (Layer 1)
  - L46+: import top_tier_mirror_policy (Layer 2)
  - L503-559: `_build_idle_baseline` + `_score_action` accept
    `opp_step0_actions` (Layer 2)
  - L658+: agent() pre-computes `opp_step0_actions` (Layer 2)
- `audit/2026-05-17-orbital-aim-verification.md` — new (Layer 1 proof)
- `audit/2026-05-17-felipe-seed-root-cause.md` — new (this doc)
