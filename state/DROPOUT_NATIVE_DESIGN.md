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

### STATUS 2026-06-19 — Phase 1 knob IMPLEMENTED; kill-gate PENDING
The `LR_DEEP_OPP` knob is landed in `agents/least_resistance/main.py`
(`_deep_opp()` + `_deep_opp_move()` dispatch, wired into both opponent call sites
of `_deep_pick`, mode read once per turn). Default `0` = producer mirror,
byte-identical; `1` = `lite_greedy_policy`. Unit-tested
(`tests/test_deep_opp_dispatch.py`) + timing-smoked (cheap opponent keeps the
deep rollout under the 1 s wall — this is also the fix for the
depth-3-mirror **validation timeout** that ERRORed sub 53836276).

**NOT yet done (next session):** the win-rate **kill-gate** —
`LR_DEEP_OPP=1` × depth {3,4,5,6} vs Producer V2 on `scripts/eval_panel.py`,
triage n=16 → confirm best depth at n≥32 (Rule 45). Pass = cheaper-deeper
≥ mirror-depth-3 (17/28). Only on a PASS do we build (header bakes
`LR_DEEP_OPP=1` + the winning depth) + Rule 42 claim + PI-signed submit. On a
FAIL the knob stays default-OFF (byte-identical) and Phase 2 (`LR_DEEP_OPP=2`,
neutral-contagion) is the next idea.

**Phase-1 triage result (2026-06-19, n=8 vs Producer V2, P0, same seeds):**
mirror_d3 4/8 margin −538 (but maxms **10067** = the validation timeout);
lite_d3 2/8 margin **−3168**. lite_greedy is fast but a WEAKER opponent model
(too attack-biased, doesn't model rival expansion) → motivates Phase 2.

### STATUS 2026-06-19 — Phase 2 (`LR_DEEP_OPP=2`) IMPLEMENTED as an OPPONENT model
The branch `claude/dropout-plan-review-rb5817` refuted the mean-field flip-hazard
model **as a leaf SCORER** (0/40 passive → 5/40 w/ opp-expansion → 19/40 parity;
`native_forward.py`). Phase 2 reuses its *principles* (model neutral expansion,
max-aggregate threat, cumulative, deterministic) in a NEW role: a discrete
**contagion opponent** inside `_deep_pick`'s rollout (NOT a leaf scorer). Landed:
`_apply_contagion(snap, me)` + the `LR_DEEP_OPP=2` branch (opponents launch
nothing; each rollout step flips neutrals + my under-defended planets to the
strongest single reachable rival, bounded one-flip-per-source, snowballing). Strong
torch leaf `_project_value` unchanged. Unit-tested
(`tests/test_contagion_opponent.py`). Kill-gate triage result recorded in the
2026-06-19 thoughts note; n≥32 + PI sign-off gate any submit.

### REFUTED 2026-06-19 (late) — the whole LR deep-search line loses to V2
Well-powered panel A/B (n=27 paired, stratified `SEED_PANEL_128[::4]`, 1v1 vs V2):
**DEFAULT least_resistance (2-ply take-and-hold) = 18/27 (67%, +275) BEATS V2**;
contagion-d6 = 9/27 (33%, −1068); wide+calib / wide far worse. So `LR_DEEP_OPP=2`
contagion, deep-search depth, wide candidates, and the over-extension calibration are
all a ~34-pp REGRESSION over the default — refuted as a path to beating V2, same shape
as the native-scorer refutation above. Everything stays default-OFF / gated as a
recorded negative result. The shipped agent is the default (unaffected). See
`knowledge-base/thoughts/2026-06-19-contagion-deepsearch-refuted-vs-v2.md`. Next
leverage = the DEFAULT agent's losses, not deep search.
