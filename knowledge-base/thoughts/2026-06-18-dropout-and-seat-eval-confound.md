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

## Open questions / next
- Phase 1b calibration (mine real flip rates from replays) is the biggest
  remaining lever for the drop measure.

## Flags
- The seat/`step` fix is default-ON but byte-identical when `step` is present;
  PI believes the platform supplies `step` to all seats, so it is a harmless
  safety net, not a behaviour change on the ladder.
- Producer V2 (`slawekbiel/the-producer-v2`) and other external opponents live
  in `audit/external/` (gitignored) — re-pull per fresh container.
