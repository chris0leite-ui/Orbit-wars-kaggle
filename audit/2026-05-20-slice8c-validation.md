# Slice 8c validation — 2026-05-20 — wait_N filter made things WORSE

> Commit `20caa6b` — one-line filter dropping wait_N>0 candidates
> from differential's input. Hypothesis: wait_N plans were locking
> sources without emitting. Verification per Rule 41.

## Single-game introspect (seed 0 vs trajectory)

**Outcome: TIE (1, 1)** — both alive at episode end (499 turns).

```
totals: cands=10746 positive-delta=1988 emits=357
per-turn avg: cands=21.5 positive-delta=4.0 emits=0.72
```

Emit rate barely moved vs Slice 8: 0.72 vs 0.75/turn. The wait_N
filter wasn't the dominant cause of under-emit.

### Deeper finding: late-game stalemate

Detailed trace inspection reveals the actual failure mode.
Late-game step pattern (480-498):

```
480-498:  ~14 candidates per turn, ALL at Δ=+0.0, 0 emits
```

The differential's closed-form Δ-favor projection correctly says
"this capture-of-expired-comet doesn't change favor; Δ=+0.0
exactly." The chooser's `delta > 0` strict gate filters all
of them. **Source idles for 20+ turns straight.**

Trajectory chooser doesn't have this stalemate because its
rollout's noisy lite_greedy policy rarely produces exactly Δ=0
— there's always some stochastic variation, so SOME move always
fires. Differential's mathematical precision is actively hurting
us in late game.

## Small A/B (n=16, vs trajectory baseline)

```
n=16  wins=3/16  (18.8%)  Wlo=0.066  Whi=0.430  FAIL  (Whi<0.55)
focal turn-ms  p50=44  p95=269  max=704
total elapsed 319.5s
```

| | Slice 8 (no filter) | Slice 8c (filter) |
|---|---|---|
| Wins | 6/16 (37.5%) | **3/16 (18.8%)** |
| Wlo | 0.185 | **0.066** |

**Slice 8c is strictly worse than Slice 8.** Dropping wait_N
candidates eliminated even more sources of action, accelerating
the stalemate. Wallclock improved slightly (704 vs 810ms max).

## Diagnosis (REVISED 2026-05-20 PM per PI directive)

The Δ=+0.0 verdict is a **feature, not a bug**. Closed-form
correctness saying "this move doesn't change favor" IS the
analytical clarity we want. The differential is right to refuse
to emit those moves.

The actual missing piece: **the candidate space is incomplete**.
The proposer only emits:
- Fire-now captures
- Wait-then-fire captures
- Defensive reinforces (when threatened)

It does NOT emit:
- **Offensive repositioning** (move ships from rear-line own
  planet to a front-line own planet to enable future attacks).
- **Concentration moves** (mass ships at a designated
  launchpad planet).
- **Strategic migration** (send ships toward zones of future
  conflict).

When the chooser correctly says "no positive-Δ launches this
turn," what it should be doing instead is **repositioning ships
to where they'll be useful in 5-10 turns**. That class of move
doesn't exist in the candidate space — proposer doesn't emit
them, chooser can't evaluate them.

The previous (rejected) hypothesis was to relax the `delta > 0`
gate. That would just inject noise back in. The right answer
is to GENERATE the missing candidate class analytically.

## Decision

**STOP on the differential-as-chooser axis.** Per plan §14
preservation strategy:

- `chooser_differential.py` stays as opt-in research code
  (`BASELINE_CHOOSER=differential`).
- The wait_N filter (`20caa6b`) stays applied — it doesn't help
  win rate but it makes the chooser's behavior less confusing
  (no high-Δ candidates that never fire).
- The closed-form Δ-favor projection (`_projected_state_at`,
  `_favor_from_state`, `score_candidate_differential`) is the
  REUSABLE SUBSTRATE for future analytical work — Slices 10-12
  in the full-analytic roadmap.

## What's missing — the analytical positioning solver

The puzzle's missing piece is closed-form **ship migration**:
given our planet graph + ship distribution + target/threat
positions, where SHOULD ships be? When a planet has nothing
to attack, the closed-form answer is to MOVE its ships to a
planet that does.

This is a network-flow problem with a clean closed-form solution:

- **Demand** at each of our planets = sum of nearby capture-
  attractive targets weighted by feasibility, minus current
  ship count. High demand = "this planet is well-positioned
  for an attack but doesn't have the ships."
- **Supply** at each of our planets = excess ships above the
  threat-defended baseline. High supply = "this planet has
  ships but no nearby target."
- **Solve**: bipartite assignment / min-cost flow from
  high-supply to high-demand planets. Output: a list of
  migration moves (src → own_tgt, ships) that minimize total
  positional misalignment.

This is the analytical analogue of what trajectory's rollout
accidentally finds via "the leaf state with closer-to-target
ships scores higher." Differential needs to model it
explicitly because closed-form leaf eval doesn't credit
"ships near targets" the way the rollout's transient state
does.

## Vision intact (revised)

The differential SUBSTRATE remains valuable. The new roadmap:

- **Slice 9 (NEW priority)**: **Ship migration solver**.
  Closed-form network-flow over our planet graph emits
  "migration" candidates (own→own moves to reposition ships).
  Differential chooser scores them via the same Δ-favor
  projection — and they'll show positive Δ because moving
  ships toward action increases future capture capacity.
- **Slice 10**: L0 commits (W1/W2) wired under differential.
- **Slice 11**: endgame solver — small-state exact 3-step
  lookahead at step ≥ 400.
- **Slice 12**: bounded-interval scoring with differential.

Slice 9 is the load-bearing missing piece. Without it, the
differential's "Δ=0 = no action" honestly reports no available
moves — because the available move set is incomplete.

## Recommendations

1. **Don't iterate further on differential's scoring gate.**
   The Δ=0 verdict is correct. Don't relax it.

2. **Build Slice 9 (ship migration solver) next.** This
   completes the candidate space. With migration moves
   available, differential's Δ-projection will find positive-Δ
   options in the late-game stalemate turns.

3. **Production unchanged**: `BASELINE_CHOOSER=trajectory`
   remains default. Rolling-pair floor μ=1118.8 preserved.
