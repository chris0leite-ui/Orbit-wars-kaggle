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
  to steepness). Kill-gate RE-RUN with the cumulative hazard recorded below.

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
