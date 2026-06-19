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

## Build phases (each default-OFF where grafted; each A/B-gated vs V2)
- **Phase A (KILL-GATE): mean-field forward model + value functional**, scoring
  producer's existing shortlist. Beat base 15/28 on the clean wide-map A/B? If
  NO → thesis refuted, stop (keep the cheap bolt-on replacement). If YES → go on.
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

---

## UPDATE 2026-06-19 — the path runs THROUGH least_resistance's search
Evidence reordered the plan: search DEPTH converts compute→strength (shipped
LR depth-3: 14/28→17/28 vs V2, margin monotone −379→−56), but depth is capped by
the 1000 ms wall because each node re-runs the PRODUCER MIRROR
(`_producer_move_obs`, ~10–50 ms, the per-node bottleneck). dropout already
replaced the mirror at parity (ladder μ 1085.6). So the dropout-native rollout
is reached INCREMENTALLY by swapping the mirror inside LR's deep search:

- **Phase 1 (cheap opponent, kill-gate):** knob `LR_DEEP_OPP` makes the deep-search
  opponent swappable; plug in the existing `lite_greedy_policy` (lib/opp_model.py,
  5–50× cheaper, models expansion). Cheaper opponent → afford depth 5–6 under the
  wall. Test: compute-scaling curve (`scripts/eval_panel.py`) `LR_DEEP_OPP=1` ×
  depth {2..6} vs V2, each timing-checked. Cheaper-deeper ≥ mirror-depth3 (17/28)
  → confirmed.
- **Phase 2 (the forward model):** `LR_DEEP_OPP=2` = the neutral-CONTAGION model —
  flip neutrals/my-planets toward the strongest rival by routable mass, grow the
  flipped garrison (free via the recurrence), and let the rival footprint SPREAD
  (newly-rival planets become new sources → snowball). Deterministic mean-field
  (no RNG). This is dropout generalized to opponent EXPANSION + compounding front
  — the cheap, model-free opponent the mirror is, minus the cost. Applied at the
  `step()` seam (main.py ~:502 in `_deep_pick`).

Seam + integration details: see the approved plan / commit. Phase 1 is the
kill-gate; Phase 2 only if cheaper-deeper wins. All default-OFF
(`LR_DEEP_OPP=0` = current producer-mirror behaviour, byte-identical).

### UPDATE — Phase 1 (cheap opponent) REFUTED; redirect to CACHING the mirror
lite_greedy as the deep-search opponent (LR_DEEP_OPP=1) scored 7-9/28 at depths
3/4/6/8 (margin -1000..-1800) vs the producer mirror at depth3 (17/28, -56).
Searching deep against a weak/wrong opponent is actively harmful — depth pays
ONLY with an ACCURATE opponent model. So do NOT replace the mirror; CACHE it.

## Iterative deepening (the redirect — afford more MIRROR-depth under the wall)
The deep search is a fixed-depth ROLLOUT per candidate (opponent = mirror each
ply, no opponent branching). ID here = anytime incremental deepening, keeping
the accurate mirror.

- **Phase A (anytime ID + incremental extension), default-OFF `LR_ITERDEEPEN`:**
  deepen d=1..cap within the timebox; **cache each candidate's Snapshot after d
  plies and extend ONE ply to reach d+1** (no re-roll → O(D) not O(D^2)); check
  the clock between levels; adopt only the deepest COMPLETED level's best;
  always hold a legal move (fallback `candidate_plans[0]`). Fixes the depth-4
  wall breach (anytime never exceeds the wall) AND lets us set a high cap and
  let time decide the depth per turn. Gate: ID-mirror >= fixed-depth-3 mirror
  (17/28) on the scaling harness, reaches effective depth >3 on a real fraction
  of turns, max-turn < wall.
- **Phase A-memo:** memoize `_producer_move_obs(board,seat)` + `_project_value`
  by a deterministic board signature (owners + int-rounded ships + step) — the
  mirror is the cost; same boards recur across candidates/levels. Reset per game.
  Only if A's effective-depth gains justify it (measure cache hit-rate first).
- **Phase B (cross-turn reuse):** persist the mirror/leaf memo (and approx. the
  chosen line's states) across turns. Bigger; gated on A.

Cross-cutting: deterministic board-hash keys, cleared at game start (seat/step +
env-leak lessons); cache-correctness assert (cached==fresh); keep the mirror as
opponent (accuracy is the lesson); OFF parity + anytime max-turn < 950ms +
same-seed determinism; eval via eval_panel.py (margin-first, fresh process, 28
maps then n>=32 + 4P + panel). Risk: memo hit-rate (rollouts share only early
plies); A (anytime) is low-risk, A-memo/B are the bets.
