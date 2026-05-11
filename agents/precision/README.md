# Precision agent — Orbit Wars

A deterministic, physics-precise agent for the Kaggle [Orbit Wars](https://www.kaggle.com/competitions/orbit-wars) competition. Every shot the agent fires is guaranteed to land on its intended target; every plan is scored against a multi-hypothesis enemy model. No learned components — pure search + ROI heuristics.

Ladder reference: submission **#52552139** (precision_v3, single-file 102 KB). Local results vs the previous lib-based bot: **83 % win-rate, ship-ratio 2.04** over 6 mirror games.

---

## Architecture

Five layers, each in one file. Strict bottom-up dependency.

```
sim.py            deterministic physics: planet/comet/fleet kinematics + combat
prediction.py     reference forward rollout (event-driven via fast_sim.py)
fast_sim.py       event-driven rollout — same semantics, ~24× faster
intercept.py      shot solver: every emitted Shot is verified to land
scoring.py        ROI for shots/waves; defense-reserve table
enemy_model.py    project enemy launches under multiple hypotheses (depth-1, depth-2)
bundling.py       multi-source wave synthesis (≤2 sources)
planner.py        global ROI-greedy + swap, robust min-max scoring
main.py           Kaggle entry point — agent(obs) with 0.85 s deadline
```

Run the agent directly:

```python
from agents.precision.main import agent
# agent(obs) returns list[[src_id, angle, ships]]
```

Or submit the bundled single-file:

```bash
./scripts/pack_precision_single.sh   # builds submissions/precision_v3.py
./scripts/pack_precision.sh          # builds submissions/precision_v2.tar.gz (alternative)
```

---

## Strategies

These were added one iteration at a time. Each iteration is recorded in commit history with the head-to-head result that justified shipping it.

### 1. Precision intercept (the foundational claim)

The engine's fleet speed depends on ship count: `v = 1 + 5·(ln S / ln 1000)^1.5`, clamped to `[1, 6]`. To hit a moving target at a chosen step `T`, the solver iterates direction-and-spawn to a fixed point (5 iterations), then picks the smallest ship-count whose fleet speed lands at the target's predicted position. Every candidate shot is verified via `_verify_intercept` (sun safety, OOB, intermediate-planet collisions); shots that fail any check are dropped.

`intercept.find_shot_for_arrival(src, tgt, T, ...)` is the inverse solver — "give me a shot that arrives at step T" — and is the building block for wave synthesis, strike-window timing, and enemy projection.

**Property:** if `find_shot` returns a Shot, executing it in the live engine lands the fleet on the target. Verified at 100 % across 156 random (src, tgt, motion) combinations (`tests/test_intercept_*`).

### 2. ROI-greedy planner with robust min-max

`planner.plan_turn` builds a candidate pool (single shots + 2-source waves + strike-window shots), then runs greedy add with a swap-improvement pass. Each trial plan is scored via `fast_sim.plan_score` under **both** enemy hypotheses:

```
robust_score = 0.7·plan_score(greedy_theirs) + 0.3·plan_score(worst_for_us)
```

The weighted blend prefers plans realistic against typical opponents while still resisting worst-case projection. The greedy add only commits a candidate if it improves the robust score; the swap pass then tries replacing each pick with an alternative.

### 3. Event-driven forward simulator (`fast_sim.py`)

`prediction.rollout` walks every step 1..horizon iterating every planet — fine for small horizons, slow for the 200-step rollouts the planner needs. `fast_sim.rollout` leapfrogs between actual events (arrivals + comet exits), batching production accrual between events. ~0.012 ms/call vs 0.30 ms baseline (**24× speedup**). Bit-parity verified (`tests/test_fast_sim_parity.py`).

In-flight enemy fleet projections are cached on the world dict (`_inflight_cache`) so they're computed once per turn instead of once per rollout — this alone cut real-game turn time from ~800 ms to ~140 ms.

### 4. Wave bundling with retention gate

Two-source wave: each source launches with a different ship count so their fleets land on the same engine step (engine sums same-owner same-tick arrivals into one attacker pool). Bundling lets us crack targets out of reach to any single source.

The wave gate is the trickiest tuning — past attempts that gated only on "wave ROI > best single-source ROI" hurt win-rate vs same-aggression opponents. The current gate combines:

1. **Strict no-single-source-captures**: emit a wave only if no single-source shot has positive ROI for that target.
2. **Source-retention floor**: each participating source must retain at least 3× its wave contribution post-firing (≤ 25 % spend per source). Prevents the "double-commit collapse" where a same-aggression opponent captures our newly-thin sources on the next turn.

### 5. Strike-window timing

For each projected enemy capture (turn T), schedule a candidate shot arriving at T+δ where δ ∈ {1, 2, 3, 5}. The just-captured planet has its garrison reduced to `E−N`; a shot landing one tick after enemy capture exploits this minimum-defender window, scoring **2×** the value (enemy denied + we acquire).

ROI of strike-window candidates is α-damped (`α = 0.7`) since the projected enemy capture might not happen; the full rollout-based `robust_score` then decides if the damped candidate is still worth picking.

### 6. Post-commitment enemy re-projection (waves)

After we fire a wave, two of our sources are weakened. A same-aggression opponent will see this depleted state and pivot. To price this, `_post_wave_threats` projects the enemy's response from the *post-wave* world (sources debited via `_world_after_wave`) and folds the projected arrivals into the worst-case scoring for that candidate. Cached by `(src_id, ships_after_wave)` signature — multiple waves over the same source-pair reuse the projection.

### 7. Depth-2 enemy minimax

`enemy_model.project_two_turns(world, end_step)` projects the enemy's response across **two** turns:

1. `t+1`: project enemy's worst-for-us actions from current world.
2. `_apply_arrivals`: shallow-clone world; resolve `t+1` combat via `sim.combat_resolve`; bump step.
3. `t+2`: project from the post-t1 world.

Combined arrivals become the `enemy_worst` in `robust_score`. Catches cascades: enemy strike at `t+1` weakens us → enemy on `t+2` finds a new opening. Applied to ALL candidates (not just waves) so the wave-vs-single asymmetry disappears.

---

## Numeric defaults (tuned, not arbitrary)

| Constant | File | Value | Why |
|---|---|---|---|
| `EPISODE_STEPS` | `sim.py` | 500 | Engine max. |
| `MAX_SHIP_SPEED` | `sim.py` | 6.0 | Engine clamp on fleet speed. |
| `SUN_RADIUS` | `sim.py` | 10.0 | Engine. Use `margin=1.0` for intercept verification. |
| `SPAWN_OFFSET` | `sim.py` | `0.1` | Verified against live engine source. |
| `WAVE_DEFENSE_HORIZON` | `planner.py` | 30 | Defense reserve looks ahead 30 ticks of enemy launches. |
| `STRIKE_WINDOW_ALPHA` | `planner.py` | 0.7 | Confidence damping on strike-window ROI (the projected enemy capture may not happen). |
| `STRIKE_WINDOW_DELTAS` | `planner.py` | (1, 2, 3, 5) | Ticks after projected enemy capture to schedule our follow-up shot. |
| `WAVE_RETENTION_MULTIPLE` | `planner.py` | 3 | Source must retain ≥ 3× its wave contribution post-firing. |
| `K_SHOTS_PER_PLAYER` | `enemy_model.py` | 1 | Enemy projection assumes one launch per player per turn. |
| Time budget | `main.py` | 0.85 s | Hard deadline; engine allows 1.0 s + accumulating overage. |
| Robust-score weights | `planner.py` | 0.7 / 0.3 | Greedy-realistic / worst-case blend. |

Increasing `K_SHOTS_PER_PLAYER` or the strike-window delta count is the cheapest way to spend more compute if profile permits.

---

## Testing

```bash
# Run the full gauntlet (matches what was green at merge time).
python3 -m pytest agents/precision/tests/ -q          # if pytest is configured
# Or directly:
for t in agents/precision/tests/test_*.py; do python3 -W ignore "$t"; done
```

Critical tests (each independently runnable):

- `test_intercept_landing.py` / `test_intercept_combinations.py` — 100 % land rate every (src × tgt) motion combination (orbit/static/comet).
- `test_inverse_intercept.py` — `find_shot_for_arrival` exact-step at 100 %.
- `test_fast_sim_parity.py` — bit-parity vs `prediction.rollout` + benchmark.
- `test_strike_window.py` — strike-window candidate generation + capture-value owner correction.
- `test_post_commitment.py` — `_world_after_wave` + wave-cost-aware scoring.
- `test_depth2_minimax.py` — `_apply_arrivals` + `project_two_turns` + within-budget planner run.
- `test_packaged_submission.py` — imports + runs from a clean cwd (catches packaging breaks).
- `test_4player.py` — runs a 4-player game cleanly.
- `test_planner_v2_vs_v1.py` — head-to-head benchmark vs frozen v1 baseline.

---

## Extending the agent

Where to add things, in order of expected impact:

1. **Better enemy modeling.** `enemy_model.py` currently assumes ROI-greedy enemies. A non-greedy opponent (e.g., comet-rusher) will surprise us. Add a third hypothesis or learn a posterior over opponent classes from observed behavior.
2. **Deeper minimax.** `project_two_turns` does two enemy turns; the natural next step is `project_three_turns` with alpha-beta pruning. Cost will need careful budget management.
3. **3+-source waves.** `bundling.py` is capped at 2 sources. Some heavy targets need 3-source synchrony.
4. **Commitment continuation across turns.** Currently the planner re-plans from scratch each turn. Caching last turn's plan-tree as a search seed (with re-validation) could trim search cost or improve consistency.
5. **Comet-aware planning.** `sim.py` and `prediction.py` track comet exits, but the planner doesn't actively use comet arrival timing to schedule attacks against comet-mounted planets while they're still in flight.

What NOT to change without strong evidence:

- The 0.7 / 0.3 robust-score weighting — tuned across iterations; aggressive shifts have regressed wins.
- The 3× wave-source retention multiple — verified empirically against same-aggression opponents.
- `MIN_KEEP_FRACTION = 0.0` (disabled) — every variant ≥ 0.1 hurts win-rate.
- The fast_sim event-driven structure — `prediction.rollout`'s step-loop is the parity reference; don't drop it.

---

## Submission format

Kaggle's validator for this competition rejects `.tar.gz` bundles (submission #52551945 failed validation with this format despite local tests passing). Use the **single-file `.py` bundle** (`scripts/pack_precision_single.sh`) — matches the convention used by all successful submissions in this comp.

The single-file pack concatenates modules in dependency order and rewrites qualified imports (`prediction.Arrival` → `Arrival`). The build script validates syntax post-concatenation; the `test_packaged_submission.py` test validates runtime behavior.

---

## Iteration history

| Iter | What | Δ vs prior | Commit |
|---|---|---|---|
| 1 | Precision intercept + greedy planner | foundation | 97d4a14, f00cf09, 129a33e |
| 2 | ROI scoring + inverse intercept + enemy projection | enables waves + post-commitment | fc75b77 |
| 3 | Swap pass + 4-player audit + packaging | submission-ready | ce63741 |
| 4 | Strike-window timing on enemy captures | 2× capture value | 15ede38 |
| 5 | Event-driven fast_sim + cache | ~24× rollout speedup, max 250 ms turn | 79770b2 |
| 6 | Post-commitment wave projection + 3× retention gate | closes wave-cascade collapse | e9425db |
| 7 | Depth-2 enemy minimax (all candidates) + ladder submission | catches enemy cascades | 1ff3443, 3de6204 |

Head-to-head reference: 83 % vs `main_v2` (the prior lib-based agent at μ=1014.7), 100 % vs `random` and `starter`.
