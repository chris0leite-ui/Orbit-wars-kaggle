# 2026-06-18 — smart dropout, and the seat/map evaluation confound

## What we did
Built "smart dropout" on `agents/producer_plus`: a model-free robustness
mechanism that REPLACES opponent modelling (opp-projection + response-veto) by
perturbing our own rollout — flip some of our planets to the opponent (captured
planets we just took, and exposed planets we hold), sized by the enemy's
routable mass, and blend the clean and dropped scores by a contest ratio. All
default-OFF; OFF path byte-identical. Variants: `dropout`, `dropout_live`
(add-on), `dropout_repl` (replacement). See `state/DROPOUT_PLAN.md` for the full
roadmap and the math framing.

## Key results (vs Producer V2, local, independent maps)
- Dropout REPLACES the opponent model at PARITY and ~half the per-turn cost
  (repl and the live opp-model stack win the same maps). The replacement
  hypothesis holds on independent maps (an earlier correlated 4-seed probe had
  mislead us with "live 6/8 vs repl 5/8").
- Compute axes do NOT help: deeper horizon hurt, "more scenarios" neutral,
  naive candidate-gen hurt. The DROP MEASURE is the lever — adding the
  held-planet half moved an early probe 0/8→5/8.

## The evaluation lesson (the important one)
We saw an apparent "second-seat (P1) collapse" — the focal lost ~all P1 games.
Chased it as a first-mover effect, then as a missing-`step` seat bug
(orbit_wars.py omits `step` for non-P0 seats; fixed defensively). Both wrong.

Root cause was an **analysis confound we introduced**: the A/B set `seat = i%2`,
tying seat to seed parity, so "P1 games" were just the odd seeds — which
happened to be focal-losing MAPS. Proven by playing each seed at BOTH seats:
**7/8 maps give the identical result at both seats** (focal P0 3/8, P1 2/8).

Conclusions, now load-bearing for all future eval:
- **No first-mover effect, no seat bias.** Outcome is MAP-determined and
  deterministic; seat is essentially irrelevant in Orbit Wars (geometry/opening
  decides).
- The harness itself is sound (self-play unbiased, in-process == file-path,
  agent-reuse == fresh-load — all verified).
- **Evaluate on many DIVERSE map-seeds, one game per seed.** Do NOT condition
  win rate on seat (confounds with map). Do NOT use `fast.py eval` for this
  (it plays each seed at both seats = correlated map-pairs, not independent n).
- Run variants SEQUENTIALLY (5 parallel torch procs OOM-kill the heavy ones —
  `live`/`gen` died silently the first time).

## Phase 1a result (incentive-weighting) — REFUTED
Wide A/B, 28 diverse map-seeds (5000-5027), one game each vs V2:
- base (dropout_repl): **15/28 (54%)** — competitive/slightly ahead of V2.
- +incentive: 13/28 (46%); paired it flipped 2 maps W→L, gained 0.
So the heuristic "our-loss-value" ranking is a worse drop proxy than raw
reachable mass — leave `PRODUCER_PLUS_DROPOUT_INCENTIVE` OFF. The principled
replacement is Phase 1b (calibrate to OBSERVED flip rates), not this proxy.
Also note: on a proper wide-map sample dropout-as-replacement is ~54% vs V2,
much better than the earlier confounded 8-seed read (3/8) suggested.

## Phase 2 + clean re-test of all axes (28 maps vs V2) — base wins
Clean (one fresh process per variant): base 15/28, more-sims(M4) 14 (parity),
incentive 13, winprob γ0.5 12, γ1.0 11, deeper(H30) 6 (catastrophic). NOTHING
beats base. Phase 2 (win-prob risk) refuted; deeper-horizon catastrophic;
more-sims neutral.

## Second eval bug (harness env-leak)
Running multiple bundles in ONE python process leaks knobs via
`os.environ.setdefault` (first variant's knobs persist). It falsely showed
more-sims 6/28 (H30 leaked) and made winprob γ0.5/γ1.0 identical. Fix: one
bundle per fresh subprocess. (The original per-variant `indep_one.py` was
already correct; my single-process `wide_ab` shortcut introduced the leak.)

## Architecture verdict
The bolt-on is SATURATED: dropout grafted on producer's static one-ply scorer
is a good CHEAP REPLACEMENT for the opp-mirror (~54% vs V2, half cost) but not a
strength engine — every refinement ≤ base. A dropout-NATIVE design (ensemble of
N stochastic hazard-rollouts on the existing batch axis, value=mean/CVaR,
ensemble-driven candidate selection) is the only way the theory (Phases 1b/2/3)
gets traction — but that's a new agent. Fork for next session: ship the cheap
replacement vs commit to the ensemble-rollout rebuild. Don't keep refining the
bolt-on. Full detail: state/DROPOUT_PLAN.md.

## SEARCH DEPTH converts compute->strength (the session's first real lift)
Built `scripts/eval_panel.py` (panel/margin/fresh-process harness). Compute-
scaling curve, least_resistance vs V2, 28 maps (one game/seed, P0):
- depth0 (2-ply): 14/28, margin -379, max 641ms
- depth2:         13/28, margin -217, max 658ms
- depth3:         17/28, margin  -56, max 818ms   <- best, beats producer/dropout 15
- depth2+wide48:  13/28, margin -203 (IDENTICAL W/L to depth2 -> breadth is a NO-OP)

MARGIN improves MONOTONICALLY with depth (-379 -> -217 -> -56). Win-rate noisy
(14/13/17) but depth3 clearly best. So `LR_ROLLOUT_DEPTH=3` is the first config
all session to beat the ~15/28 wall, using more of the 1000ms budget. Breadth
(LR_MAX_CANDIDATES) does nothing (<28 sensible candidates typically).
Caveat: n=28, depth3 wlo=0.42 -> triage, not yet a confident lift (Rule 45 wants
n>=32 wlo>=0.5). And LR searches a PRODUCER opponent model while playing V2, so
this lift is despite model-mismatch (encouraging; the accurate-model curve vs
producer/self-play should be even cleaner).

## depth-4 (#2): margin curve crosses zero; depth is wall-limited at ~3-4
Timing gate: depth4 UNCAPPED breaches the wall (max 1189ms 2P / 1035ms 4P).
Capped (LR_WALLCLOCK_MS=650, no overage bank) it FITS: max 925ms (2P) / 871 (4P).
Capped depth4 vs V2, 28 maps: 16/28 (wlo 0.39), margin +20, max 925ms.
Full margin curve (monotone in depth): d0 -379, d2 -217, d3 -56, d4 +20.
- MARGIN improves monotonically with depth and crosses zero at depth4 (LR now
  avg AHEAD of V2 in final position) — clean compute->strength.
- WIN-RATE plateaus ~16-17/28 (d3=17, d4=16, tied within noise).
- depth4 is AT the timing wall (925ms capped). depth5+ infeasible without a
  CHEAPER per-node leaf. So depth ~3-4 is the operating ceiling, set by the
  1000ms wall, not by the idea. Next lever = faster/ensemble leaf eval (afford
  more depth), or confirm depth3 at n>=32 and ship.

## Cheap-opponent kill-gate: REFUTED (depth needs an ACCURATE opponent model)
Shipped depth-3 mirror LR (sub 53836276). Then Phase 1 = swap the per-node
producer mirror for cheap lite_greedy (LR_DEEP_OPP knob; ~6x cheaper: depth4
p50 453->71ms, so depth 6-8 fits the wall). Kill-gate vs V2, 28 maps:
- lite_greedy d3 7/28 (margin -1778), d4 9 (-964), d6 8 (-1577), d8 9 (-1060)
- mirror d3 17/28 (-56)  [reference]
lite_greedy is FAR worse at every depth, and depth doesn't recover it. Lesson:
**searching deep against a weak/wrong opponent model is actively harmful** (you
optimize against a phantom; V2 is a producer variant so the mirror generalizes,
lite_greedy doesn't). Depth pays ONLY with an accurate opponent model.
=> Phase 1 REFUTED; keep LR_DEEP_OPP=0 (mirror) default (shipped depth3 unaffected).
=> Phase 2 (contagion) now in DOUBT (also a cheap approximate opponent) -- it's a
   better approximation than lite_greedy but must rival the mirror's accuracy.
=> REDIRECT: don't REPLACE the accurate mirror, CACHE it -- iterative deepening +
   cross-turn tree reuse keep accuracy while cutting recompute. That's the right
   way to afford depth 5+. (The knob/seam from Phase 1 stays as default-OFF infra.)

## Open questions / next
- Confirm shipped depth3-mirror at n>=32 + panel + 4P (the validated config).
- Caching the mirror (iterative deepening / cross-turn tree reuse) to afford
  more depth WITHOUT losing accuracy -- the redirect.
- Phase 2 contagion only if it can be shown accurate enough (uncertain).

## Flags
- The seat/`step` fix is default-ON but byte-identical when `step` is present;
  PI believes the platform supplies `step` to all seats, so it is a harmless
  safety net, not a behaviour change on the ladder.
- Producer V2 (`slawekbiel/the-producer-v2`) and other external opponents live
  in `audit/external/` (gitignored) — re-pull per fresh container.
