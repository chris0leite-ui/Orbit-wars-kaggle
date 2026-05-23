# Question — what's the empirical local 75% A/B → LB μ mapping?

## The gap
We submit on local A/B winrates against a fixed strong sibling. The
calibration ladder (`state/calibration-ladder.md`) lists predicted
vs actual μ per submit, but I don't have a clean "for n=16 local
12/16 = 75% vs <opp X>, expected μ shift = ?" table.

The sibling branch's data point (local 12/16=75% vs their internal
opp → μ ~963-985, vs a prior internal of μ ~973) suggests **75%
local with Wlo just over 0.50 maps to roughly parity or slight
regression**, not the +X lift the point-estimate suggests.

Our V3 submit (sub 52966655) will give us our own data point.

## Why it matters
If the mapping is "12/16 = noisy parity, need 14/16+ for +20μ
expected lift," our submit policy is too aggressive at n=16.
Rule 45's minimum (0.50) becomes a soft floor that submits parity-
likely candidates and burns the rolling-pair slot.

## What I want to know post-V3
1. V3's settled μ.
2. Three more local-75%-or-higher pushes from any branch in the
   next week, plotted: local Wlo on x, μ shift from evicted on y.
3. Whether the slope is positive, zero, or negative.

## If the slope is ≤ 0 for n=16 75% submits
- Raise Rule 45's submit gate from 0.50 → 0.55 (panel parity).
- Require n=32 with Wlo ≥ 0.55 (matching Rule 43 panel target).
- Effectively close the n=16-as-submit-gate loophole.

## If the slope is positive but small (+10-30μ per +5pp local lift)
- Keep Rule 45 as-is but add a "predicted μ shift" column to the
  push-claim board that uses the mapping explicitly. Force the
  Rule 42 evicted-vs-predicted comparison to use calibrated numbers,
  not eyeballed ranges.
