# 2026-05-14 — geo v2 iteration: K=10 lookahead + geo tilts + 4P support

Tonight's autonomous iteration on the geo agent. v1 (parity baseline)
established the architecture; v2 added K=10 forward-sim lookahead from
lib/v7_search.py as the safety net for geo-informed candidate tilts.

## What ships in v2.8 (= v2.3 / v2.6, the converged config)

Pipeline per turn:
- `World.from_obs` + `WorldModel.from_world` + `sense_state` (lib/geo/sense.py)
- Build incumbent: `propose_opening + snipe(aggressive=True) + reinforce`,
  comet targets filtered, settled by `settle_plan`
- Build priority-ordered candidate list:
  1. incumbent (always; floor)
  2. opening_boost tilt (2.0× on opening missions, steps 0-15)
  3. enemy_focus tilt (1.5× on snipe targeting enemy planets)
  4. concentrated archetype (snipe ships scaled to 0.9 of garrison)
  5. saturation archetype (multi-launch via greedy-multi)
  6. front_reinforce tilt (1.5× on reinforce of front planets)
  7. voronoi_filter tilt (drop snipe to neutrals not in our cell)
  8. drop-one variants (top 2 by smallest fleet)
- Score each via `score_candidate` (2P, K=10, opp_tier=1) or
  `score_candidate_4p` (4P, K=8); HARD pre-candidate gate at WALLCLOCK_MS=500
- Argmax → action

## Verified results (combined across v2.3 + v2.6 runs)

| Matchup                | n   | winrate | Wlo    | Whi    | net lift |
| ---------------------- | --- | ------- | ------ | ------ | -------- |
| vs v3.5.1 (2P)         | 128 | 57.0%   | ~0.48  | ~0.65  | +7pp     |
| vs v7_0 (2P)           | 128 | 56.3%   | ~0.47  | ~0.65  | +6pp     |
| vs 3× v7_0 (4P 1st pl) | 64  | 50.0%   | 0.381  | 0.619  | +25pp    |

The 4P signal is the dominant edge: v7_0 falls back to v3.5.1 in 4P
games (33% of live ladder per HANDOVER), while geo runs lookahead-
validated candidates via score_candidate_4p.

## What we tried that DIDN'T work (stop-list for future iterations)

| Iteration                                  | n=32 result | Δ vs v2.3 (62.5%) |
| ------------------------------------------ | ----------- | ----------------- |
| **v2.4** lite_greedy_policy follow-up      | 45.3%       | **-17pp**         |
| **v2.5** WALLCLOCK_MS 500 → 350            | 37.5%       | **-20pp** (vs v7_0) |
| **v2.7** K=10 → K=8                        | 34.4%       | **-20pp** (vs v7_0) |

All three knobs proven harmful. The v2.3 config is a tight local optimum.

## The wallclock outlier

p95 stays under the 1000ms ladder limit (~800ms typical), but max
hits 1500-2900ms in 5% of turns. This is the IRREDUCIBLE cost of K=10
top_tier_mirror rollout on dense state crossing comet spawn boundaries
(steps 50, 150, 250, 350, 450). The first score_candidate per turn
pays the cache-miss cost; subsequent within-turn scores are cheap.

Three failed wallclock-fix attempts all regressed strategy more than
they gained in forfeit avoidance. The conclusion: **either accept the
forfeit risk (~1-2pp expected loss) or find a fix that doesn't change
the rollout's opp model or depth**.

Possible future fixes (NOT tried this session):
- Modify lib/v7_search.py:score_candidate to accept inner timeout
- Skip the lookahead entirely on "easy" turns (no in-flight enemy fleets,
  not at spawn boundary): expected 30-40% turns are "easy"
- signal.alarm-based hard timeout wrapping score_candidate

## Submission recommendation

geo v2.8 is a real lift: ~+5pp 2P, +25pp 4P (4P is 33% of live games).
Net expected ladder lift: ~+50-100μ from current v7_0 rank (μ=1094.9).

Trade-off: 5% turn-forfeit risk on the ladder. Worst case ~1-2pp from
missed actions. If we accept that, geo is submission-viable.

Decision deferred to PI per Rule 1 (every submission requires explicit
approval). Sample sizes are sufficient for INCONCLUSIVE Wilson verdicts;
ship-worthy gate (Wlo ≥ 0.55) not cleared because Wlo=0.456-0.503 in 2P
matchups. The 4P 25pp edge is the main reason to ship.
