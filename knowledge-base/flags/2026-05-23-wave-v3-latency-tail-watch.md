# Flag — wave V3 latency tail watch

## The risk
V3 submit (sub 52966655, PENDING) max-turn latency in n=16 local
eval was 1199ms, over the 1000ms environment cap. On Kaggle
hardware (typically slower than local), the tail could be 1.2-1.5x
larger, pushing some turns into hard-timeout territory.

Precedent: sibling sub 52963659 (orbitfix_kt_p23 v4) submitted with
local max 1867ms → ERROR'd in production with "60s overage budget
exhausted by ~turn 203."

## What to watch
- V3's settled status in `kaggle competitions submissions orbit-wars`.
  ERROR (not COMPLETE) means the latency tail bit.
- If V3 completes but lands μ << predicted (say < 950), latency
  may have caused per-turn forfeits without an outright ERROR.

## If ERROR
- Pivot the next session immediately to latency hardening.
  Targets: identify which turn-types produce >1000ms (likely
  early-game with full planet/comet enumeration in
  `predict_fleet_fate`), apply targeted cache (analogous to
  planet_positions but for comet positions / aim solutions).
- Do NOT push another wave variant before max < 900ms locally.

## If COMPLETE with μ ≥ 990
- Latency tail did not cause forfeits in the validation games.
- Still worth a one-session pass on max-turn reduction to make
  V4-style pushes safer.
