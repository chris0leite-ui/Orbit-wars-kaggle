# Q: Is the sync coalition a genuine ladder gain, or only a panel-beater?

**Opened:** 2026-05-31 (`champion-strategy-rules-00JzI`)
**Status:** OPEN — answered empirically by sub `53223160`'s settled μ.

## The question

The synchronized two-source coalition beats our calibration panel decisively
(v7_0 90.6% / v4_planner 93.8% / v3.5.1 87.5%). But head-to-head vs the
champion (`baseline_launch_rules_universal`) it's only ~44–56% — essentially a
tie. So is sync actually *better* than what's already on the ladder, or does
the champion also crush the same panel (making sync a lateral move)?

## How to answer

1. **Primary:** read sub `53223160`'s settled μ. ≥~1140 (near/above the evicted
   1183 champion) → real gain. ≤~1100 → panel-beater only.
2. **The missing control we never ran (~30 min):** run the *champion* against
   the SAME three panel opponents. If it also scores ~90%, sync is not an
   upgrade over the live agent; if clearly lower, sync is the upgrade.

## Why it matters

It's the gate for the whole sync line: if sync is a real gain, make it the
production base and decide on locking it in the rolling pair; if it's lateral,
stop polishing sync and pivot to a new mechanism (H44 fleet-survival defense or
2-hop redeploy). Don't iterate further on sync until this is answered.
