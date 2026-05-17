# 2026-05-17 — v8_analytic value-head pivot to fast_sim + lite_greedy

Branch: `claude/space-fleet-physics-engine-lrLE6`. Session goal:
diagnose why v8_analytic loses to nearest, then either fix or kill.
PI verdict: **keep the architecture as a baseline for future
ambitious ideas**.

## Load-bearing facts

### Root cause of the 50% vs nearest ceiling

Diagnosed via `/tmp/diagnose_losing_game.py` (macro-trace) and
`/tmp/micro_trace.py` (per-state value-head dump). Seed 1 vs nearest,
agent loses at turn 229. Trace shows v8_analytic launching on
**19% of turns** vs nearest's **49%** — passive play.

At turn 80 (5 mine / 6 opp / 13 neutral — a state that obviously
calls for expansion), the JAX K=8 value head produced:

- 40 candidate atoms post-cap
- **0 atoms score better than no-op**
- **38 of 40 atoms score EXACTLY equal to no-op** (delta = 0.0000)
- 2 atoms score worse than no-op (the only two with ETA ≤ K, bounces)

Mechanism: `value_with_future_production` (`analytic_score.py:80-135`)
sums `my_ships + my_prod × γ_h`. In-flight fleets count as `my_ships`
(lines 123-126), so a launch with ETA > K moves ships from "planet's
ships" to "in-flight" with zero net delta. Capture hasn't landed by
turn K=8 → `my_prod` unchanged. Leaf state is **bit-identical to
no-op** for any candidate whose ETA exceeds K.

Median launch ETA in mid-game ≈ 10-30 turns. K=8 fundamentally cannot
see most candidates' payoffs. All 7+ prior tuning attempts this
session (cap=64/128, single-wave mirror, multi-wave mirror, width=3/4,
dedup, defensive window, fragility-aware cheap-rank) operated on the
candidate pool BEFORE the value head or on the opp simulation INSIDE
it. None changed the leaf representation.

### The fix

Replace the JAX vmap value head with `lib/fast_sim.py`'s Python
rollout + reactive lite_greedy opp + v8_scavenge's `_favor` leaf.
New module `lib/foundation/strategies/analytic_fastsim.py` (~190
lines). Architecture pivot landed in commits:

- `abcb77b` — replace JAX value head with fast_sim + lite_greedy
- `7e511a0` — bump fast_sim K=8 → K=15, N=40 → N=25

### K-sweep on the diagnosed-broken state (seed 1 turn 80, 40 atoms)

Demonstrates the new value head's discrimination scales monotonically
with K — a tunable knob with predictable effect:

| K  | atoms beating no-op | timing (single state) |
|----|---|---|
| 8  | 1 | 65 ms |
| 15 | 5 | 140 ms |
| 25 | 8 | 302 ms |
| 40 | 9 | 625 ms |

Selected K=15 N=25 as the budget-fitting balance.

### Probe 1 quality bench (n=8 side-balanced)

| Variant | vs nearest | vs v7_0 | Note |
|---|---|---|---|
| JAX baseline (width=3, single-wave mirror) | **4/8** (50%) | **2/8** (25%) | committed `c89eb71` |
| fast_sim K=8 (regressed) | 2/8 (25%) | 0/8 (0%) | uncommitted intermediate |
| fast_sim K=15 N=25 | **4/8** (50%) | **0/8** (0%) | committed `7e511a0` |

**vs v7_0 regressed (2/8 → 0/8)** while vs nearest held flat (4/8 →
4/8). Two explanations consistent with the data:

1. **K=15 is still too short for the longer-ETA captures v7_0
   doesn't punish.** v7_0 is a lookahead agent that anticipates
   far-future moves; my K=15 leaf doesn't see those moves. The
   value head systematically misjudges actions that pay off after
   K, and v7_0 exploits that gap more than nearest does.
2. **`lite_greedy` is a poor opp model for v7_0.** lite_greedy is
   nearest-style; v7_0 plays differently. The reactive baseline
   predicts opp's response wrong, so my candidates' delta-favor
   isn't calibrated to the real opponent.

Both are addressable via the listed "Where to go next" extensions
(wallclock-guarded K extension lifts (1); `top_tier_mirror_policy`
swap addresses (2)).

Same point estimate vs nearest (4/8). Wilson 95% LB = 21.5%; n=8 too
small to clear the strict 40% LB threshold from the original plan.

**Pattern change:** the agent swapped which seeds it wins.

| Seed | Side | JAX | fast_sim K=15 |
|---|---|---|---|
| 0 | p0 | win (499) | loss (485) |
| 1 | p0 | loss (229) | **WIN (499)** |
| 2 | p0 | loss (159) | loss (262) |
| 3 | p0 | win (454) | win (499) |
| 0 | p1 | win (109) | loss (461) |
| 1 | p1 | loss (251) | **WIN (475)** |
| 2 | p1 | loss (257) | loss (237) |
| 3 | p1 | win (454) | win (493) |

Seed 1 went 0/2 → 2/2 (genuine progress predicted from the K-sweep —
at K=15 we got 5 positive atoms vs 1 at K=8). Seed 0 regressed
2/2 → 0/2; loss profile changed character (was 109-499 turn wins,
now is 461-485 turn losses — full games, slowly outproduced rather
than eliminated).

### Timing

p95 across all 8 nearest trials: 154-705 ms. One over-1000ms turn out
of 4,500+ warm turns total. Well within the 1000 ms Kaggle budget.

p50 mostly 72-294 ms (compare JAX path at 411-798 ms). Fast_sim is
~3× cheaper per turn than the JAX vmap path. Substantial budget
headroom for layering on more ambitious work (see "Where to go next").

## Architecture verdict: KEEP

Three pieces of evidence support "this is a viable substrate, not a
dead end":

1. **K-sweep monotone.** 1 → 5 → 8 → 9 positive atoms as K rises.
   The value head's discrimination capability has a clear, tunable
   gradient. Contrast: JAX path produced 0 positive at any tuning.
2. **K change moved real outcomes.** Seed 1 win pattern flipped
   exactly as the K-sweep predicted. The pipeline is reactive to
   interventions in the right places.
3. **Timing headroom.** ~300 ms unused at K=15 N=25 mid-game. Room
   for beam-over-fastsim, longer K with wallclock guards, mirror
   opp, or learned head.

## Where to go next (ambitious ideas the substrate now supports)

Listed in order of expected EV per implementation hour.

1. **Wallclock-guarded K extension.** Bail mid-rollout when cumulative
   cost > 700 ms; targets K=25-40 in light states, K=15-20 mid-game.
   v8_scavenge does this exact pattern. ~30 LOC.
2. **Beam search over fast_sim.** ~12 ms per candidate at K=15 N=25
   leaves room for depth=2 width=3 ≈ +110 ms. Could find compound
   actions singleton-greedy misses.
3. **Wait-then-fire candidates.** Direct port of v8_scavenge
   `_wait_then_fire_candidate` (lines 316-375). Action-space
   expansion, not infrastructure work. ~80 LOC.
4. **Stronger opp model.** Swap `lite_greedy_policy` for
   `top_tier_mirror_policy` (already in `lib/opp_model.py`). Single
   line change; cost ~10× per call so requires K or N reduction.
5. **Learned value head.** Use fast_sim rollouts as training data;
   small NN predicts K-step `_favor` from state. Replaces explicit
   rollout with constant-time inference. Enables effective K=80+.

## Critical paths

- `lib/foundation/strategies/analytic_fastsim.py` (new this session;
  cf commit `abcb77b`). `_favor` is v8_scavenge port; pipeline is the
  v8_scavenge shape with our enumerate_capped + mission framework.
- `lib/foundation/strategies/analytic_score.py` — `enumerate_capped`
  now returns `(atom, target_id)` tuples when `return_targets=True`.
  Cheap-rank formula unchanged; defensive enumeration unchanged.
- `lib/foundation/strategies/analytic.py` — `emit` calls
  `score_and_select_via_fastsim` at the two former `beam_search`
  call sites. Width/depth knobs kept for API compat but unused.
- `lib/foundation/strategies/beam_search.py` — still on disk, no
  longer imported from analytic.py. Used by no live agent. Safe to
  delete in a follow-up if confirmed orphan.

## Reproduction commands

```bash
# Diagnostic trace of a losing game (root-cause artifact)
python /tmp/diagnose_losing_game.py > /tmp/losing_seed1_trace.log

# Micro-trace: value-head ranking at a specific mid-game state
python /tmp/micro_trace.py   # seed=1, turn=80, 40 atoms

# Probe 1 bench (n=8 side-balanced)
OPP=nearest SEEDS=0,1,2,3 python /tmp/probe1_bench.py
OPP=v7_0    SEEDS=0,1,2,3 python /tmp/probe1_bench.py
```

## Decisions taken without PI override

- Cap=128 → 64 → 128 (multiple variants, none moved metric)
- Single-wave mirror at turn 0 added (no quality movement)
- Multi-wave per-step mirror added then reverted (`2d69d1f` →
  `8893e9b`) due to v7_0 regression 1/4 → 0/4
- Width=4 → width=3 (timing fix; quality unchanged)
- Value head replaced with fast_sim + lite_greedy
- K=8 → K=15, N=40 → N=25 after K-sweep showed K=8 was the
  architectural floor

## Decisions where PI intervened

- "Continue to Probe 2" → "Fix the timing tail first" (PI override
  on plan's Wilson-LB strict reading; right call, decoupled timing
  from quality).
- "Plan it first" before fast_sim implementation (forced re-entry
  to plan mode; tighter implementation as a result).
- "We do not need to win, we just need to know if we can use the
  architecture as a strong baseline" (reframed the kill criterion
  from win-rate to substrate-viability).
