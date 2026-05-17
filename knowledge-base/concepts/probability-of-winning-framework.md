# Probability-of-winning framework for the chooser

> Filed 2026-05-17 after PI critique on the trajectory chooser's 3
> consecutive 0/32 failures vs v15. Sister doc to `trajectory-first-
> architecture.md` and `trajectory-chooser-v2-sketch.md`. This is the
> reframe that follows from those negatives.

## The reframe

PI's exact words: *"how does taking this trajectory affect my
probability of winning, or how does this set of trajectories, even in
a sequence, help me position myself better and harness compounding
and helps to improve my probability of winning?"*

The chooser should not be a **per-fleet outcome scorer**. It should
be a **`ΔP(win)` optimiser**:

| current paradigm | reframed |
|---|---|
| `score(action) = production × time` if captured | `score(action) = E[P(win | state·action) − P(win | state·idle)]` |
| per-candidate independent | joint over my actions + opp's reactions |
| single-turn | multi-turn sequence (this turn's capture funds next turn's launch) |
| measures resource gain | measures probability-of-eventual-victory gain |

The `production × time_remaining` term in `composite_capture_value` is
the crudest possible compounding surrogate — it assumes captures are
held forever and resources convert linearly to winning. Neither is
true.

## Three compounding mechanisms (currently captured poorly)

1. **Production → ships → captures → more production.** A
   production-1 capture early might fund ~50 extra ships by mid-game
   (linear term). It can also fund 2–3 more captures along the way
   (re-investment loop). Composite captures only the linear term;
   missing the multiplier.
2. **Positional → opportunity.** Owning a planet next to a contested
   neutral lets us strike there cheaply (short ETA, fewer ships
   needed). The neutral's eventual capture probability depends on
   which side reaches it first, which depends on our adjacency. None
   of our current value heads represent this.
3. **Tempo → opp limitation.** A launch now forces opp to react this
   turn (defend) instead of expand. The cost to opp is the value of
   moves they cannot make. Composite never sees this.

## Three operational directions

### A. Richer leaf eval ("compounding-aware favor")

Smallest change. Keep `fast_sim`-along-trajectory (v3's rollout
method); replace the binary owner-check leaf with `favor`-style
scoring that includes:

- `(my_ships − opp_ships)` — immediate strength
- `(my_prod − opp_prod) × pv_horizon(step, gamma)` — discounted
  future production (existing `lib.scoring.pv_horizon` captures the
  geometric compounding)
- `(my_planet_count − opp_planet_count) × planet_weight` — board
  control
- (deferred) `Σ neighbor_value(p) per owned planet` — positional bonus

Subtract idle baseline (`Δ = leaf_with_action − leaf_idle_at_same_horizon`)
to get a v15-style preference relation.

**Cost:** ~40–60 LOC. **Hypothesis tested:** "v3's binary leaf was
the killer." If A reaches v15 parity → information collapse was the
issue; v3's fast_sim rollout was already structurally correct.

### B. Joint action evaluation (set-of-trajectories)

Score candidate COMBINATIONS, not individuals:

- Greedy beam: pick top-k one at a time, re-evaluating the world
  after each commit. Each subsequent pick sees the world updated
  with previously-committed launches.
- Captures interaction effects ("if I send fleet A, my home is
  empty — defending move B becomes valuable").

**Cost:** ~100–150 LOC. **Hypothesis tested:** "Per-candidate
independent scoring misses joint effects." Even v15 evaluates
candidates independently — this would be a genuine architectural lift
beyond v15.

### C. Sequential planning (k-turn lookahead with opp model)

Plan 2–3 turns ahead:
- Turn t: pick action; simulate opp response (`lite_greedy_policy`)
- Turn t+1: re-pick; simulate opp again
- Turn t+2: estimate `V(state)` via cheap heuristic
- Pick turn-t action maximising `V` at t+2

**Cost:** ~200–300 LOC with aggressive pruning. **Hypothesis tested:**
"Single-turn rollout misses multi-turn strategy." Heaviest direction;
reserved for after A and B are exhausted.

## Why A first

1. **Decisiveness.** A's outcome is a clean fork — either v3's leaf
   scoring was the binding constraint (A passes, ship it) or it
   wasn't (A also fails, retire trajectory direction).
2. **Cheapness.** ~1 hour to build + A/B vs ~half-day for B,
   1–2 days for C.
3. **Compositionality.** If A succeeds, B and C can layer on top of
   it. If A fails, B and C inherit the same broken leaf and don't
   help on their own.

## What "P(win)" actually means here

Three operational definitions, increasingly principled:

1. **`favor` proxy** (cheapest): treat `(ships_diff + prod_diff × pv)`
   as a P(win) proxy. Works if the proxy ranks states in the same
   order as actual win-rate. Bias-prone but no training cost.
2. **Learned value function**: train a tiny model `V(state) →
   P(win)` on self-play or replay data. ~5k params (cf. konbu17's
   shot validator). 1–2 days of training infra + data prep.
3. **Monte Carlo rollout to game end**: drop the leaf eval; play
   each candidate forward via `fast_sim` until the game terminates;
   read the actual reward. Exact but expensive (~500 ticks per
   sample × N samples per candidate). Probably infeasible at
   per-turn wallclock.

Direction A uses (1). If we want true P(win) we move to (2) post-A.

## Cross-references

- `knowledge-base/concepts/trajectory-first-architecture.md` — the
  v1 architectural reframe that started this thread.
- `knowledge-base/concepts/trajectory-chooser-v2-sketch.md` — v2
  attempt; defense + opp lookahead + multi-launch.
- `agents/baseline/chooser_trajectory.py` — v3 implementation with
  fast_sim along trajectory; loses 0/32 vs v15.
- `agents/baseline/chooser.py` — v15's K-step rollout chooser
  (`build_idle_baseline` + `score_action` with `Δ = leaf − baseline`).
- `agents/baseline/value.py` — `favor` / `composite` / `hybrid`
  value heads (the leaf-eval functions A reuses).
- `lib/scoring.py:pv_horizon` — geometric compounding factor used by
  `favor`. Already captures the discount-future-production term.
