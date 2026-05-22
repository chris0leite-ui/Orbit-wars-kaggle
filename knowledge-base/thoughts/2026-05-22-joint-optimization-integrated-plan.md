# 2026-05-22 — Joint optimization, integrated plan

> Branch: `claude/strategy-axis-decision-3437` (session-EqJuT)
> Origin: PI directive — "We really need the joint optimization. We
> have all the trajectories encoded. We can backwards optimize. We
> can run simulation. So we have a lot of speed. How would we best
> integrate it with what you're suggesting now?"
>
> The "what you're suggesting now" is the Lagrangian / dual-decomposition
> formulation of the joint LP, sketched in the prior turn.

## 1. The substrate, restated

Four primitives, all closed-form or byte-exact:

1. **Trajectories** — `lib/trajectory.py::predict_fleet_fate` returns
   the fate of any (src, tgt, angle, ships, launch_step) tuple (sun /
   oob / target / comet / planet-block / timeout) + the arrival tick.
   With `KINEMATIC_TABLE_ENABLED=1` (default-on per commit `c48e143`)
   the inner-loop position lookup is O(1).
2. **Backwards-induction predicate** — `lib/joint_solver/predicate.py`
   gives closed-form `is_winning_state`, `is_winning_state_if_owned`,
   `is_winning_state_if_lost`. The underlying quantity
   `W(s) := prod_advantage(s) · remaining_turns(s) − opp_pool(s)`
   is signed and **smooth**: positive ⇒ we're in winning state,
   magnitude ⇒ how much margin.
3. **Per-planet game-end forward sim** — `outcome_table.enumerate_outcomes`
   walks any subset of arrivals to game-end horizon T=500 and returns
   `(owner_T, ships_T, prod_stream_by_owner)`. Per planet, O(2^k)
   for k ≤ 6 candidates.
4. **Byte-exact full forward sim** — `lib/fast_sim.py` is ~20× faster
   than `env.clone()+step()`; rollout K turns of full game with any
   policy is affordable per turn (~5–10 ms per K=20).

We also have the orchestration pipeline (`lib/pipeline/`) and the
joint LP (`lib/joint_solver/lp_outcome.py`), and a queued-but-unrun
Level 1 topology A/B (reachability / mutual defense / recapture risk
in `_value_for_outcome`, gated `LP_TOPOLOGY_FEATURES=1`).

## 2. The joint LP, restated

Variables (per `lib/joint_solver/lp_outcome.py`):

- $x_c \in \{0,1\}$ — fire column $c$. $c$ = (source $s(c)$, target $t(c)$,
  ships $n_c$, wait $w_c$, ETA $\eta_c$, angle).
- $y_{p,S} \in \{0,1\}$ — subset $S$ of arrivals at target $p$ (|S| ≤ 6).

Constraints:

- (C1) one subset per target: $\sum_S y_{p,S} = 1 \;\forall p$. *Local to p.*
- (C2) linkage: $x_c = \sum_{S \ni c} y_{p(c), S} \;\forall c$. *Local to c.*
- (C3) per-source ship budget over time: $\sum_{c:\,s(c)=s,\,w_c \le u} n_c\,x_c \le R_s + P_s u$.
  *The only block coupling different sources — i.e., the only coupling
  block at all.*

Objective: $\max \sum_{p,S} V_p(S)\, y_{p,S} - \kappa \sum_c n_c\, x_c$.

The leaf value $V_p(S)$ today is `prod_stream_me − α·prod_stream_opp`
plus a binary endgame bonus $\lambda \cdot \mathbb{1}[\text{tipping subset}]$
plus the in-tree Level 1 topology terms.

## 3. What "joint optimization" really wants — beyond what's there

Joint factors along four dimensions:

| Dimension | Current state | What's missing |
|---|---|---|
| Planets | Joint via $y_{p,S}$ subset choice + (C2) linkage | None — covered. |
| Time (multi-turn waits) | Joint via wait_N grid + (C3) budget over u ∈ [0, max_wait_N] | max_wait_N=5; deeper plans not in. |
| Opponent | Static one-shot `opp_projection` baked into ledger | Reactive opp (Stackelberg outer) — pipeline has a partial leader; not converged. |
| Game outcome | Step-function endgame bonus | **Smooth** $\Delta W$ from `is_winning_state` math. |

So the missing pieces are (a) a smooth, backwards-induction value layer
on top of $V_p(S)$, (b) a reactive opp loop, and (c) deeper time horizon —
all of which need wallclock headroom we don't have because the current
MILP eats ~300 ms and Stackelberg is unaffordable on top.

The Lagrangian inner is **the mechanism that buys the wallclock**.

## 4. Lagrangian decomposition — math sketch

Dualize the only coupling block, (C3), with multipliers $\lambda_{s,u} \ge 0$.
Define per-column rent $\bar\lambda_c := \sum_{u \ge w_c} \lambda_{s(c), u}$.
The Lagrangian decomposes target-by-target:

$$L = \sum_p \Big[ \max_{x,y \text{ obeying (C1,C2)}_p} \big(\sum_S V_p(S) y_{p,S} - \sum_{c: t(c)=p} (\kappa + \bar\lambda_c)\,n_c\,x_c\big) \Big] + \text{const}(\lambda).$$

Each per-target subproblem: enumerate 2^k subsets, pick $S^*_p = \arg\max_S [V_p(S) - \sum_{c \in S}(\kappa+\bar\lambda_c)n_c]$.

**Shadow price interpretation** (the PI's "invisible hand"):
- $\lambda_s$ = marginal LP gain from one more ship at source $s$
  (closed-form: the per-ship-value of the last unit consumed in
  source $s$'s waterfilled column ladder).
- $\pi_p(c) := V_p(S^*_{p, \text{with }c}) - V_p(S^*_{p, \text{without }c})$ —
  marginal value of column $c$ at its target.

A column fires iff $\pi_p(c) > (\kappa+\bar\lambda_{s(c)}) \cdot n_c$.
**Each source's decision is independent given prices.** Strong LP duality
makes this exact at the LP relaxation; the integer optimum has a small
gap (the subset $y_{p,S}$ block isn't totally unimodular, but $V_p(S)$
is approximately a step function in "winning subset," so rounding is cheap).

**Inner loop:**

```
λ ← warm-start from previous turn  # state changes slowly
for iter in 1..3:
  for each source s:                 # O(K log K)
    rank columns from s by (π_t - κ·n_c - λ_s·n_c) / n_c
    waterfill against budget R_s + P_s·u
  for each target p:                 # O(2^k), k ≤ 6
    S*_p = argmax_S [V_p(S) - Σ_{c∈S}(κ+λ_{s(c)})·n_c]
  recompute λ_s from the last-unit marginal in s's water-fill
```

Wallclock estimate: 30 sources × O(N_cols × log) + 30 targets × 64 subsets
× small arithmetic + 3 iters ≈ **10–20 ms.** (Current MILP: ~300 ms with
HiGHS.)

## 5. Backwards-induction value — smooth endgame

Today's endgame term:

$$V_p(S) \mathrel{+}= \lambda_{\text{endgame}} \cdot \mathbb{1}[\text{owner}_T(S) = me \;\wedge\; S \text{ tips us into winning state}].$$

Replace with the smooth ΔW:

$$V_p(S) \mathrel{+}= \lambda_W \cdot \big[ W(s_T(S)) - W(s_T(\emptyset)) \big]$$

where $W(s) = \text{prod\_advantage}(s) \cdot \text{remaining\_turns}(s) - \text{opp\_pool}(s)$
is the LHS of `is_winning_state`. Closed-form, O(1) per subset given
`outcome_table`'s `prod_stream`. **Defense and tipping captures both
appear in $\Delta W$ smoothly** — losing a planet drops $W$ proportional
to its production × remaining time, gaining one raises $W$ symmetrically.

This is what "backwards optimize" means concretely: the smooth game-end
value flows back through the per-subset enumeration into the LP's
per-column shadow prices.

## 6. Stackelberg outer with mirror Lagrangian

With Lagrangian inner at ~20 ms, alternating solves are affordable:

```
solution_us ← Lagrangian inner (assuming opp's static projection)
for outer_iter in 1..3:
  opp_arrivals ← opp's mirror Lagrangian inner (assuming our solution_us)
  solution_us ← Lagrangian inner (with new opp_arrivals merged into ledger)
  if |Δ objective| < ε: break
```

The pipeline already has `opp_mirror_analytical.py` (per HANDOVER bug
#2 fixed at commit `52fa7b8`) — it's the right primitive. The current
`decision_stackelberg_leader.py` does one half-step; this loop closes
to an approximate equilibrium.

Total budget: 3 outer × 2 inner solves × 20 ms = 120 ms. Still leaves
budget for leaf verification.

## 7. Forward-sim leaf verification

The top-K (e.g., K=4) Lagrangian solutions get forward-simulated via
`fast_sim.rollout` for K_step=20 turns with `lite_greedy_policy` on
both sides. Pick the candidate with the highest $W$ at the end of the
rollout. This is **simulation verifying analytical math** — using
"speed" to check the model.

Cost: 4 × 20 × ~0.5 ms = ~40 ms. Total per-turn budget breakdown:

| Stage | ms (target) |
|---|---:|
| perception + candidates + opp_model + prerank | 70 |
| value function (closed-form V_p(S) with topology + ΔW) | 150 |
| Lagrangian inner × Stackelberg outer (3×3) | 180 |
| top-K leaf verification with fast_sim | 60 |
| commit + bookkeeping | 10 |
| **total p50** | **470** |

Headroom under 1000 ms actTimeout: ~2×.

## 8. Revised phase plan

Each phase = one axis. Each phase gated by an n=4 → n=8 A/B (Rule 45),
serial, with bundle parity check (Rule 46). No two axes change
simultaneously (Rule 21).

**Phase α — Smooth endgame value (the "backwards optimize" wiring).**
Replace `_endgame_bonus` step function in `lib/joint_solver/lp_outcome.py`
with $\lambda_W \cdot \Delta W$ using closed-form `prod_advantage`,
`remaining_turns`, `opp_pool` deltas. Single knob $\lambda_W$. Gate:
n=8 vs `_phase4_step1_FND` (currently rolling pair) on seeds
[42,1,7,13,31,100,17,23], serial, 4P-included subset.
*Estimated 4 hours.*

**Phase β — Level 1 topology features A/B (queued).**
Bundle `_phase4_topology_NEW.py` per HANDOVER step 1; run screenshot
calibration on episodes 77321232, 77320686, 77323008; n=4 → n=8 A/B vs
FND. *Estimated 3 hours.* **Independent of α.**

**Phase γ — Lagrangian inner as parity-checked drop-in.**
Implement closed-form per-source water-fill + per-target subset
reconciliation + 3-iter fixed-point in a new module
`lib/joint_solver/dual_decomp.py`. Same `V_p(S)` as `solve_outcome_aware`.
Parity gate: on 8 representative seeds, run both solvers, measure
relative objective gap (target ≤ 2%) and move-set agreement (target ≥
90% column-set overlap). If clean: hot-swap behind
`LP_SOLVER=dual` env var. *Estimated 1 day.*

**Phase δ — Stackelberg outer (mirror Lagrangian).**
With γ in place, alternate us/opp/us/opp inner solves for ≤3 outer
iterations. Reuse `opp_mirror_analytical`. Gate: n=8 vs FND, but the
real measure is the audit `d76bbaa` "half-tempo" gap closing in
replay-stats screenshots. *Estimated 1 day on top of γ.*

**Phase ε — Top-K leaf verification with fast_sim.**
Take top-4 of γ's solutions; rollout K=20 each; pick the one with best
$W$ at horizon. Already partially in `leaf_outcome_table.py`. Gate:
n=8 vs FND. *Estimated 0.5 day.*

**Phase ζ — Bundle, 4P A/B, push.**
Per Rules 42 / 43 / 45 / 46. *Estimated 0.5 day.*

**Total: ~5 days for the full integration.** Each phase has a kill-gate;
if α or β fails, γ–ε don't ship.

## 9. What changes the order

The PI's framing "we really need joint optimization" doesn't quite
match what's missing — we already have a joint LP. What we're missing
is (a) a smooth backwards value, (b) reactive opp, (c) the wallclock
headroom to afford both. The Lagrangian is the mechanism for (c).
α and β answer "is the value-function direction right" before we spend
a day on γ. γ buys budget for δ + ε.

**Recommended start:**

1. (this session) Phase β — run the queued topology A/B. Already coded,
   3 hours wallclock. Decisive on whether the value-function direction
   is alive.
2. (next session, parallel-OK with β) Phase α — wire $\Delta W$ into
   `_value_for_outcome`. 4 hours. Strictly orthogonal to β (different
   term in the same objective).
3. (after both clear) Phase γ — Lagrangian inner. The investment that
   unlocks δ + ε.
4. δ + ε + ζ.

If β AND α both null, we have a stronger reason to pivot away from the
LP family entirely — toward a learned shot validator (Konbu17 +19pp
empirical, the only ML approach with precedent) — rather than build
more LP polish.

## 10. Risks called out

1. **Compound-axis confound (Rule 21).** Don't change value function
   and solver in one A/B. α and β are different terms in the same V;
   γ is a different solver under the same V. The plan respects this.
2. **Integrality gap (Lagrangian).** Claimed "small in practice"; must
   be measured in γ's parity gate. If > 2%, fall back to MILP and
   accept γ as research code.
3. **Bundle parity (Rule 46 candidate).** Three sessions burned in
   HANDOVER on broken bundles (constant collisions, missing `agent`,
   `lib.mirror` not inlined). Every phase has `tests/test_bundle_*`
   parity tests.
4. **"Joint optimization at speed" is not μ.** Speed buys options:
   Stackelberg, leaf-sim, deeper waits. Each of those needs its own
   A/B to confirm it produces μ. The Lagrangian itself is parity-by-
   construction with current MILP; the μ comes from what we layer on
   top.
5. **Top-10 plays heuristic, not LP.** Bowwowforeach μ=1683 "concentrated
   artillery," Konbu17 panel-best at 85% via MLP shot validator. The
   LP family may have a ceiling. The plan should be ready to pivot to
   ML if γ–ε don't crack μ=1150.

## 11. Concrete next action

Run Phase β. The bundle, calibration screenshots, and A/B harness are
already in tree; this is ~3 hours of execution, not new design. Outcome
decides whether the integrated plan above moves forward as written or
gets revisited from the top.
