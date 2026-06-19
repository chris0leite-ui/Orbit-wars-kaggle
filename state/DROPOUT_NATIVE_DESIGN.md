# state/DROPOUT_NATIVE_DESIGN.md — a dropout-native agent (design / build plan)

> Written 2026-06-18. Read `state/DROPOUT_PLAN.md` first for why we're here:
> dropout grafted onto producer_plus is a good CHEAP REPLACEMENT for the
> opponent mirror (~54% vs Producer V2 at ~half cost) but SATURATED as a
> strength engine — every refinement (incentive, winprob, deeper) is ≤ base
> 15/28 on the clean wide-map A/B. The binding constraint is the producer's
> ONE-PLY STATIC flow-delta value function, not dropout's drop measure. This
> doc designs the agent where dropout is the *forward model*, not a graft.

## Thesis
Value an action by a **statistic (mean, then CVaR) over a distribution of
adversarial futures**, where the forward model itself is stochastic ownership
evolution under a **per-step flip-hazard process**. The producer does ONE
deterministic exact projection and scores once; this does a distribution of
projections and scores by an expectation/risk measure. This is "dropout" in the
true sense (average over stochastic perturbations), not a 2-point blend.

If a distribution-aware forward model scoring the SAME candidates does not beat
the static one-ply scorer, the thesis is wrong — Phase A is the kill-gate.

## Core architecture

### 1. Forward model = ownership jump process (the heart)
Replace "exact combat recurrence + one bolted reflip" with ownership that
evolves stochastically each step. For each planet p and step k:
- **Flip hazard** `λ_p(k)` from local force balance: attacker reachable mass
  (`_reactive_reinforcement_margin`, already exists) vs our projected garrison.
  `λ = calibrated_sigmoid((atk − def)/(atk + def))` — a steep-near-parity curve;
  Phase 1b calibration (observed flip rates) feeds THIS function.
- Deterministic dynamics unchanged: production accrues, our launches land,
  in-flight fleets resolve, combat at arrivals.

Two flavors (build the first; add the second only if risk-shaping pays):
- **(v1) Mean-field / moment propagation — DETERMINISTIC, no RNG.** Track
  `P_mine(p,k)`, `P_opp(p,k)` (ownership probabilities) and expected garrison;
  propagate them through the step recurrence using `λ` as transition rates. ONE
  pass, no sampling, CPU/GPU bit-identical (the codebase's hard requirement).
  Strongly preferred for v1 — avoids all RNG-determinism plumbing.
- **(v2) Sampled ensemble — for CVaR/risk.** N rollouts, each sampling Bernoulli
  flips ~ λ per (p,k); value = mean or CVaR over N. Needs a deterministic seed
  derived from (step, board hash). Only build if v2's risk shaping beats v1.

### 2. Value functional
Competitive score = expected production-weighted ownership margin over the
horizon: `Σ_k discount(k) · Σ_p prod_p · (P_mine(p,k) − Σ_r P_opp_r(p,k))`.
Mean-field gives this directly; sampled ensemble gives mean or CVaR_α (optimize
the worst α-fraction → distributionally robust). This subsumes the current
terminal-prod / hold-value hacks.

### 3. Candidate generation & selection
- v1: reuse producer's shortlist (`build_target_shortlist`, planner_core) as the
  candidate SOURCE — proven, cheap. Score each candidate by running the
  stochastic forward model with that candidate's launches applied; pick greedily
  as today. (Isolates "does the forward model help?" from "does generation
  help?".)
- v2+: ensemble-driven / robust-action search (keep actions that help the
  expected/worst-case rollout), and **self-consistency**: recompute λ given the
  policy just chosen and re-score 1-2 iterations (fictitious play in outcome
  space) — partially recovers opponent sequencing.

## What to reuse (do NOT rebuild)
- `agents/producer/orbit_lite/adapter.py::single_obs_to_tensor` — obs→tensors.
- The batched exact recurrence `_run_exact_recurrence` (garrison_launch) — it
  already carries the leading [N] batch axis; extend it to propagate ownership
  probabilities (mean-field) or run N sampled worlds.
- `agents/producer_plus/main.py::_reactive_reinforcement_margin` / `_margin_reach`
  — the hazard inputs (attacker reachable mass per target/tick).
- `build_target_shortlist` (planner_core) — candidate source.
- The seat-independent turn counter in `tensor_action` (already fixed).
- `scripts/bundle_producer_plus.py` — bundling pattern for a new entry module.

## Build status (2026-06-18)
- **Phase A forward model BUILT** — `orbit_lite/native_forward.py`:
  `build_candidate_trajectories` (dense per-candidate trajectory via the trusted
  `_run_exact_recurrence`), `reachable_enemy_mass` (enemy physically-routable
  mass by step k from `cross_dist` + `fleet_speed`), `flip_prob` (steep-near-
  parity contest sigmoid), `hazard_ownership_value` (the value functional).
  Mean-field, no RNG. Unit tests (`tests/test_native_forward_model.py`): trajectory
  construction reduces EXACTLY to the engine recurrence with no launches; the
  core thesis (holdable capture out-values a thin one) holds; monotone in enemy
  mass; deterministic.
- **Wired** into producer_plus behind `PRODUCER_PLUS_NATIVE_HAZARD` (default OFF,
  OFF path byte-identical — verified). Bundles cleanly (cold-load full game vs V2
  OK). Knobs: `_STEEPNESS` (default 5), `_DISCOUNT` (default 1).
- **Kill-gate A/B RESULT (n=40 paired, continuous margin vs V2):**
  | variant | wins | mean margin | paired Δ vs base |
  |---|---|---|---|
  | base (bolt-on) | 21/40 | +0.051 | — |
  | native (steep 5) | 19/40 | −0.050 | −0.101 [−0.45,+0.25] up/dn 6/9 |
  | native_s8 | 19/40 | −0.050 | identical to steep 5 on ALL 40 maps |

  **Verdict: Phase A does NOT pass the kill-gate** — native is below base
  (Δmargin −0.10, point estimate negative; CI brackets 0 so not a significant
  *regression*, but it does not BEAT base as the gate requires). HOWEVER, unlike
  the saturated bolt-on (whose refinements were inert, ≤4 maps changed), native
  is a genuine alternative policy: it changes **15/40 maps**, **winning 5 maps
  base loses** (5002/5003/5018/5026/5029) while losing 7 base wins, and it is
  **faster** (max 223 ms vs 252 ms — no opponent mirror). So the forward model
  has real expressive traction; its v1 instantiation just nets slightly worse.
  **Steepness bracket — the flip-HAZARD was INERT:** steepness 0.5, 5, 8, 20 are
  byte-identical across ALL 40 maps. So Phase-A-v1 never tested the thesis — it
  tested plain ownership-margin vs the tuned `competitive_score` (which loses 19
  vs 21); the distribution machinery contributed nothing. CAUSE: the hazard was
  applied as an INSTANTANEOUS per-step haircut `(1-leak)`, a second-order
  perturbation dominated by the large deterministic ownership term — the very
  "thin layer over a coarse value function" failure the bolt-on had.
  FIX (committed): CUMULATIVE survival `surv = Π_j (1-leak_j)` applied only while
  I hold the planet, so sustained threat compounds over the horizon and the
  hazard becomes the primary signal (verified load-bearing: value now responds
  to steepness ON SYNTHETIC BOARDS).

- **Kill-gate RE-RUN with the CUMULATIVE hazard (n=40 paired vs V2):** native
  19/40, Δmargin −0.10 — **BYTE-IDENTICAL to the instantaneous version**, and
  native ≡ native_s20 on all 40 maps again. So even the load-bearing cumulative
  hazard is INERT FOR CANDIDATE RANKING in real games: it changes the value
  MAGNITUDE but never the argmax over candidates.
  ROOT CAUSE: `atk_reach` is candidate-INDEPENDENT (current enemy ships, same for
  every candidate), so the hazard discount is a near-constant across candidates
  that cancels in the argmax; where it IS candidate-dependent (the touched
  source/target garrisons) it is dominated by the large deterministic ownership
  term. A distributional layer over a per-candidate DETERMINISTIC trajectory +
  a FIXED shortlist is dominated by the deterministic core — the SAME failure
  mode as the bolt-on, one level deeper.

## PROGRESS LOG — making the native agent actually play (2026-06-19)
Observation-driven iteration after the code-review fix (PI watched games):
| value functional | wins/40 vs V2 | margin | launches | failure observed |
|---|---|---|---|---|
| ownership (post bug-fix) | 0/40 | −1.00 | ~30 | idle (passivity) |
| + opp-expansion + marginal | 5/40 | −0.75 | churn | leads ownership, bleeds ships |
| + max-threat | 5/40 | −0.75 | churn | (no aggregate move) |
| **expected SHIP-MARGIN** | **13/40** | **−0.35** | **218** | (churn gone; gap to base closing) |
| base bolt-on (reference) | 21/40 | +0.05 | 214 | — |

**Ship-margin reformulation (the breakthrough):** the value optimized
production-weighted OWNERSHIP, a proxy that diverges from the engine's win
condition (total ships) in the churn regime — the agent out-produced V2 yet bled
~12,760 ships/game in thin captures that reflip. Reformulated to EXPECTED
SHIP-MARGIN (`orbit_lite/native_forward.py`, `PRODUCER_PLUS_NATIVE_VALUE=ships`):
weight `(P_mine − P_opp)` by per-planet ship count + post-horizon production
credit (`prod·terminal`), add in-flight ship mass, instantaneous leak (cumulative
over-penalized dominant garrisons once ship-weighted), discounted mean (ship
units, matches the 1.5-ship roi floor). Result: churn eliminated (launches
30→218 = base level), win-rate 5→13/40, margin −0.75→−0.35. Seed 5000 (the close
loss the PI diagnosed) flipped to a dominant win (30 planets / 996 ships vs 1/2).
Still below base 21/40 — remaining gap is the next lever.

**λ/steepness sweep (n=40 each) — tuning is DONE:** native (λ=12, steep=5) 13/40
is the sweep optimum; λ=6 12/40, λ=24 **8/40** (higher λ HURT — over-credits
production, over-expands), steep=3/8 10/9. So the gap to base is NOT a tuning
issue. **PI observation (seed 5006):** we fail to capture a high-value corner
neutral that V2 takes. Refuted as candidate-generation (wide neutral shortlist =
no change) AND as production-weighting (higher λ worse). Diagnosis: the high-value
corner neutrals have HIGH garrisons (75/31); cracking one needs CONCENTRATED force
(multiple coordinated waves), but the chooser commits one wave per target and
takes cheap low-garrison filler instead — at turn 45 our 4 planets make prod 6 vs
V2's 5 planets / prod 11. **Next lever = force concentration / coalitions**
(PRODUCER_PLUS_FORCE_CONCENTRATION / SYNC / COALITIONS) to take defended
high-value neutrals, NOT tuning.

**Anticipatory threat growth (grow enemy reservoir by opp production over the
horizon) — REFUTED on aggregate.** Diagnosis (traced 5 losses): consistent
mid-game frontier collapse (t60-80) — P0 reaches mid-game even/ahead then loses
held planets to V2's production-grown army (reactive threat model is blind to it).
Fix gated `PRODUCER_PLUS_NATIVE_THREAT_GROWTH` (alpha). A/B n=40, paired vs native:
alpha=0.25 Δ=+0.000 [-0.30,+0.30] up/dn 5/5 p=1.00 (EXACT parity); alpha=0.5
Δ=-0.10 (worse); alpha=1.0 Δ=-0.40 p=0.04 (significantly worse, launches 218->180
= over-suppression). It fixed the target map (seed 5032 loss->win) but globally
over-defends maps whose frontier wasn't actually doomed, canceling the gains.

### FOUR closing-levers all failed to beat native 13/40 (gap to base 21/40 open):
1. wide neutral shortlist — no change (not a generation gap).
2. lambda (production credit) sweep — native lambda=12 optimal; higher HURT.
3. force concentration — worse (6/31 partial; over-concentrates).
4. anticipatory threat growth — parity at best, worse as alpha rises.
**The ship-margin value reformulation (0/40 -> 13/40) was the real, durable win.**
native one-ply hazard plateaus ~8 wins below the mature bolt-on; the remaining gap
likely needs structure the producer's tuned scorer already has (multi-ply
sequencing, calibrated coalition/sync), not another single knob. Recommend banking
native at 13/40 + the ship-margin finding and stopping the knob hunt.

## ⚠️ ALL RESULTS ABOVE THIS LINE ARE VOID (code-review 2026-06-19)
A code review found the native scorer threw a shape error on EVERY turn
(`garrison_status.arrivals_by_owner` is `[P, H+1, A]`; the code derived `H` from
it as `[P, H, A]` → reshape RuntimeError), silently swallowed by
`except Exception: pass`. Instrumentation: ENTERED 440 / SUCCEEDED 0 — native
NEVER ran. So every "native 19/40 / hazard inert / steepness-identical /
cumulative≡instantaneous" result above measured the STATIC scorer with the
dropout bolt-on OFF (which differs from base only by the bolt-on — explaining the
15-map gap, and why steepness/cumulative did nothing). FIXED (strip the k=0
frame; ENTERED 499 / SUCCEEDED 499; play changes). Plus correctness fixes
(clamp source debit; zero hazard where unreachable; guard native+dropout blend).
Kill-gate RE-RUN with the native scorer actually executing — REAL verdict below.

[old VOID verdict, retained for the record:]
~~The mean-field flip-hazard forward model ... does NOT beat the static one-ply
scorer (19/40 vs 21/40), and the hazard distribution is provably inert ...~~
(VOID — the model never executed.)

## REAL VERDICT — Phase A kill-gate FAILED decisively (model now executes)
With the native scorer actually running (n=40 paired vs V2):
| variant | wins | mean margin | Δ vs base | median focal launches |
|---|---|---|---|---|
| base | 21/40 | +0.051 | — | 214 |
| native | **0/40** | **−0.996** | −1.05 p=0.00 ✱ | **30** |
| native_s20 | 0/40 | −0.996 | identical to native | — |
| native_sc (self-consist) | 0/40 | −0.996 | identical to native | 32 |

native loses ALL 40 maps, near-eliminated every game. steepness (5≡20) and
self-consistency (concentrated adversary) change NOTHING — both 0/40 identical.
ROOT CAUSE (mechanistic): the agent is **pathologically passive** — ~30 launches
vs base's ~214. The forward model applies the flip hazard only to MY planets (it
models the opponent ATTACKING me) but never models the opponent EXPANDING onto
neutrals. So "do nothing" is costless in the value functional (neutrals stay
neutral, P_opp=0), and any launch only debits a source to chase a
hazard-discounted target → idleness dominates the ranking. The value functional
is HALF an opponent model.

This is the airtight refutation the PI asked for (self-consistency tried, 0/40).
The Phase-A thesis — a mean-field flip-hazard overlay on a per-candidate
deterministic trajectory, scoring the producer's shortlist — does not produce
competitive play and cannot beat base. A working dropout-native agent would need
the forward model to ALSO model opponent expansion (neutral capture) — i.e. a
genuine two-sided distributional rollout (the v2 ensemble), not a one-sided
hazard overlay. Given base is only parity-with-V2 (below the live champion), that
rebuild is not justified. RECOMMENDATION stands: bank the eval harness + this
(now real) negative result; stop the dropout line.

What would be required to make the distribution load-bearing for RANKING (NOT
funded without a fresh PI decision — these are big builds the doc gated behind a
Phase A pass that did not happen):
- **candidate-dependent threat / self-consistency (Phase D):** recompute λ given
  the chosen policy so a defensive candidate actually lowers its own planets' λ
  — only then does the hazard reorder candidates;
- **v2 sampled ensemble + ensemble-driven GENERATION (Phase C/D):** score
  genuinely different sampled futures AND generate the robust candidates the
  shortlist doesn't contain.
Given base itself is only PARITY with V2 (below the live champion), the evidence
does not justify funding that rebuild. RECOMMENDATION: bank the
continuous-margin eval harness (the durable win) + this negative result; do not
pursue the dropout line further unless the PI explicitly wants the speculative v2.

## Build phases (each default-OFF where grafted; each A/B-gated vs V2)
- **Phase A (KILL-GATE): mean-field forward model + value functional**, scoring
  producer's existing shortlist. Beat base 15/28 on the clean wide-map A/B? If
  NO → thesis refuted, stop (keep the cheap bolt-on replacement). If YES → go on.
  *(Now measured on the continuous paired margin, a far sharper gate than 15/28.)*
- **Phase B: calibrate λ** to observed flip rates. Real ladder replays are
  gitignored/absent → GENERATE data from local games: instrument every
  (planet, step) flip with its local force balance and the flip/no-flip
  outcome, fit a logistic, bake the curve.
- **Phase C: risk** — sampled ensemble + CVaR_α, and win-probability-aligned α
  (insure ahead / gamble behind) — now meaningful because the value function is
  a real distribution.
- **Phase D: ensemble-driven generation + self-consistency** iteration.

## Risks / guards
- **Determinism:** prefer mean-field (no RNG). If sampling, seed from
  (step, board hash); verify CPU/GPU parity.
- **Timing:** N rollouts × C candidates × H must stay < 1000 ms/turn. Mean-field
  is one pass (cheap); the freed opponent-mirror budget + the batch axis cover
  it. Bench every phase.
- **It may not beat producer.** Producer is a strong one-ply scorer. Phase A is
  a hard kill-gate — do not proceed on faith.
- **Eval discipline (mandatory):** one bundle per FRESH SUBPROCESS (bundles leak
  knobs via `os.environ.setdefault` in-process — this bit us); many DIVERSE
  map-seeds, one game/seed; NEVER condition win rate on seat (confounds with
  map); never `fast.py eval` (correlated map-pairs). Template:
  `/tmp/run_one_bundle.py` + subprocess driver, or per-variant `indep_one.py`.

## Decision
This is a NEW agent, not producer_plus with knobs. The fork (in
`state/DROPOUT_PLAN.md`): ship the cheap bolt-on replacement, OR commit to this
rebuild. Phase A settles whether the rebuild is worth finishing.
