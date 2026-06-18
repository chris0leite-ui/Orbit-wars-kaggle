# state/DROPOUT_PLAN.md — smart-dropout roadmap (executable from a fresh session)

> Written 2026-06-18. The active research line on `agents/producer_plus`.
> Read this + `HANDOVER.md` + the thoughts entry
> `knowledge-base/thoughts/2026-06-18-dropout-and-seat-eval-confound.md` to
> resume. All mechanisms are default-OFF env knobs; the OFF path is
> byte-identical to the shipped agent.

## What smart dropout is
Instead of MODELLING the opponent's launches, perturb our own forward rollout:
flip some of our planets to the opponent (captured-planet reflips + held-planet
drops), sized by the enemy's physically-routable mass, and score robustly. It
estimates the game-theoretic value of an action in *outcome space* ("which of
my planets change hands, when, with what probability") instead of *action
space*. It works well exactly when that drop distribution is a **calibrated,
incentive-weighted, self-consistent** image of optimal opponent play, blended
with a **win-probability-aligned risk functional**.

Code: `agents/producer_plus/main.py` — `_dropout_reflip_legs`,
`_held_dropout_plan`, `_dropout_adjusted_status`, the dropout block in
`plan_lite_waves` (gated by `dropout_ok` so it runs only on our real pass, not
inside opponent mirrors). Bundler variants in `scripts/bundle_producer_plus.py`:
`dropout` (standalone), `dropout_live` (add-on to the live stack), `dropout_repl`
(replaces the opponent model, keeps reactive_floor + FFA).

## Established this session (vs Producer V2, local)
- **Dropout replaces the opponent model at parity, ~half the per-turn cost.**
  On independent maps repl_base and the live opp-model stack win the SAME maps.
- **Compute is NOT the lever.** Deeper horizon and "more scenarios" did not help
  (deeper hurt); naive candidate-gen hurt (over-defends). The **drop measure**
  is where the headroom is (adding the held-planet half moved an early probe
  0/8→5/8 — *which* planets drop dominates).
- Timing: every variant benches < 1000 ms/turn (the removed opponent mirror
  frees budget).

## CLEAN wide-map A/B results (28 diverse maps 5000-5027, one game/seed vs V2)
Fresh process per variant (see eval warning below):

| variant | wins | vs base |
|---|---|---|
| **base (dropout_repl)** | **15/28** | — |
| more-sims (DROPOUT_SCENARIOS=4) | 14/28 | parity (neutral) |
| incentive (Phase 1a) | 13/28 | worse |
| winprob γ=0.5 (Phase 2) | 12/28 | worse |
| winprob γ=1.0 (Phase 2) | 11/28 | worse |
| deeper horizon (HORIZON_2P=30) | 6/28 | catastrophic |

**Verdict: nothing beats base.** Every measure/risk refinement (incentive,
winprob) regresses; deeper horizon is catastrophic; more-sims is the only
neutral one. The bolt-on is SATURATED — the binding constraint is the
producer's one-ply static flow-delta value function, not dropout's drop
measure. Refining dropout within this architecture has no traction.

## ARCHITECTURE verdict & the fork
Dropout grafted onto producer_plus is well-suited for exactly one job —
**cheaply replacing the opponent mirror** (validated: ~54% vs V2 at ~half
cost). It is NOT a path to surpass the producer, because it collapses "a
distribution over adversarial futures" into a 2-point blend bolted onto a
static one-ply scorer; the refinements perturb a thin layer over a coarse value
function. The plateau (above) is the evidence.

A **dropout-NATIVE** design would be a different agent:
1. Ensemble of N stochastic rollouts (sampled drop masks), value = mean/CVaR —
   `_run_exact_recurrence` already has the batch axis to run N at once.
2. Forward model = per-step flip HAZARD (Markov ownership), not exact-combat +
   a bolted single reflip — so calibration (1b) and risk (2) get real traction.
3. Candidate selection driven by the ensemble (robust-action search), not a
   fixed greedy shortlist; self-consistent drops.

THE FORK for next session: (a) lock in / ship the cheap opp-model replacement
(what we have), or (b) commit to the ensemble-rollout rebuild if dropout is to
be a strength engine. Do NOT keep refining the bolt-on — the data says it's done.

## EVALUATION — how to measure (read before running any A/B)
- **HARNESS BUG to avoid (learned the hard way):** do NOT exec multiple bundles
  in ONE python process. Bundles set knobs via `os.environ.setdefault`, so
  env vars LEAK between variants (the first variant's knobs persist). This
  silently contaminated a run (H30 leaked into more-sims → false 6/28;
  winprob γ=0.5 leaked into γ=1.0 → identical results). Run **one bundle per
  fresh subprocess** (template: `/tmp/run_one_bundle.py` + a subprocess driver,
  or the original per-variant `indep_one.py`).
- **Outcome is MAP-determined and seat-invariant.** Verified: 7/8 seeds give the
  identical result at both seats; focal win rate P0 3/8 vs P1 2/8. There is NO
  first-mover effect and NO seat bias.
- Therefore: **use many DIVERSE map-seeds, one game per seed.** Seat balancing is
  unnecessary. Do NOT condition win rate on seat (it confounds with map).
- Do NOT use `fast.py eval` for the dropout A/B: it plays each seed at BOTH seats
  (common-random-numbers), so its "n" is correlated map-pairs, not independent
  games. Use a fresh-load-per-game harness over distinct seeds (template:
  `/tmp/wide_ab.py` pattern — fresh exec per game, one game per seed vs
  `audit/external/agents/slawekbiel_the-producer-v2/main.py`, `PYTHONPATH`
  includes `agents/producer` so V2 can import orbit_lite).
- Run variants **sequentially** (5 parallel torch procs OOM-kill the heavy ones).
- Producer V2 is the discriminating peer; pull it fresh per container:
  `kaggle kernels pull slawekbiel/the-producer-v2 -p <dir>` then extract the
  `%%writefile main.py` cell to
  `audit/external/agents/slawekbiel_the-producer-v2/main.py`.

## Phases (ranked by leverage)

### Phase 0 — seat/`step` fix — DONE (committed)
The orbit_wars interpreter omits `step` for non-P0 seats; adapter defaulted it
to 0. Fixed with a seat-independent turn counter in `tensor_action`
(byte-identical when `step` is present). NOTE: the PI believes the platform
actually supplies `step` to all seats, so this is a harmless safety fix, not the
cause of anything. Kept because it's byte-identical when step is present.

### Phase 1 — faithful drop MEASURE (make-or-break) — IN PROGRESS
- **1a. Incentive-weighting** (`PRODUCER_PLUS_DROPOUT_INCENTIVE`, committed
  default-OFF): rank which held planets drop by the opponent's incentive = our
  loss value (`prod·(H−flip_tick) + garrison`) instead of raw enemy mass.
  **REFUTED (leave OFF):** A/B at n=28 diverse maps vs V2 — base 15/28 vs
  incentive 13/28; paired it flipped 2 maps W→L and gained 0. The heuristic
  "our-loss-value" ranking is a worse drop proxy than raw reachable mass.
  The principled replacement is Phase 1b (calibrate to OBSERVED flip rates),
  not this proxy. (`orbit_lite/strategic_value.py::denial_bonus` remains a
  reusable opponent-value signal if revisited.)
  NB baseline finding: dropout_repl (dropout REPLACING the opp model) scores
  ~54% vs V2 on 28 diverse maps — competitive/slightly ahead, at ~half cost.
- **1b. Calibrate the flip probability** (not built): mine replays
  (`scripts/replay_mine.py`, `classify_losses.py` over `audit/live-episodes/`)
  for empirical flip frequency vs force-ratio/distance/production/#threateners;
  fit a logistic; replace the raw `threat/(threat+defense)` contest ratio.

### Phase 2 — risk functional (not built)
- **2a.** Adversarial selection + CVaR over the worst plausible drops at a
  tunable α (replace the plain mean).
- **2b.** Win-probability-aligned risk attitude: heavier drop weight when ahead
  (protect the lead), lighter when behind (a trailer must gamble).

### Phase 3 — the RIGHT "deeper" (not built)
- **3a.** Per-step flip HAZARD / Markov-ownership value (track `P(I own p at t)`
  integrated over the horizon) instead of a single discrete flip.
- **3b.** Self-consistency: iterate exposure↔policy 1-2 rounds (fictitious play
  in outcome space) — partially recovers the move-sequencing dropout can't see.

### Phase 4 — generation + estimator hygiene (not built)
- **4a.** Re-test `PRODUCER_PLUS_DROPOUT_GEN` AFTER 1a/1b (naive gen hurt; on a
  calibrated/incentive measure the generated defensive candidates should pay).
- **4b.** Variance reduction (antithetic scenarios, clean-score control variate,
  importance-sample the discriminating scenarios) for sharp rankings from few
  sims.

### Known ceiling
Outcome-space dropout cannot represent opponent SEQUENCING (tempo, feints,
combinations). 3b recovers some; the escape hatch is a hybrid — dropout for
breadth + a shallow real-opponent lookahead on the single most critical line.

## Per-phase gate
Default-OFF knob; `pytest tests/test_dropout.py` green; bundle builds;
`fast.py bench` max < 1000 ms; wide-map A/B vs V2 beats the parity baseline
before stacking the next phase. Never ship without the Rule 42 submit gate.
