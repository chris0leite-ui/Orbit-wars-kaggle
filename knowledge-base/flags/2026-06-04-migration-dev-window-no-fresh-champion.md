# 2026-06-04 — FLAG: migration dev window, no fresh champion submissions until Step 8

## Flag

During the Producer-host migration (Steps 2-7 of `state/MIGRATION_PLAN.md`),
we will NOT submit anything new to the Kaggle ladder until the hybrid
`producer_plus` clears its A/B gate vs our live champion at n=32 with
Wilson-lo ≥ 0.55.

## Why this is a flag and not just a plan note

The rolling-pair eviction logic (Rule 42) means a careless submit during
the dev window could evict our backstop. The two slots Kaggle keeps for
final scoring are:
1. `champ_computeByShips_on.py` (sub 53332500) — live μ ≈ 1185
2. `champ_adaptiveK_on.py` (sub 53324164) — μ = 1185.2

Both are needed as backstop while `producer_plus` matures. A premature
Producer-derived submit (even for "calibration") would evict `champ_adaptiveK_on`,
costing us the diversity in the rolling pair.

## Mitigation

- No `kaggle competitions submit` until Step 8 of MIGRATION_PLAN explicitly
  starts.
- Step 8 begins ONLY after `producer_plus vs champ_computeByShips_on` clears
  Wilson-lo ≥ 0.55 at n=32.
- Rule 42 push-claim board check still mandatory at that point.

## Ladder erosion risk

If Steps 2-5 take weeks, our live μ may drift downward as opponents
improve. Acceptable cost — better than burning a slot on a premature
submission. Re-evaluate if μ drops below ~1150.

## Status

OPEN — closes at Step 8 launch.
