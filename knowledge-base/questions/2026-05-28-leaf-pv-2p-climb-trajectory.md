# OPEN — Where does leaf_pv_2p (sub 53117942) land?

**Raised:** 2026-05-28 PM2.

Sub 53117942 read μ=921.3 ~30 min after submit. That is a starting
point on the μ-climb (μ₀=600 entry, climbs as it plays). Lifetime
question: where does it stabilize?

## Branches the answer cuts

- **μ stabilizes ≥ 1100**: 2P leaf production-PV term is at worst
  neutral; the silent-turns thesis was directionally right and the
  2026-05-18 calibration debt was overstated. Investigate whether
  PV_ETA + leaf_pv_2p combined gives further lift.
- **μ stabilizes 950-1100**: term is mildly harmful on the live
  pool. The silent-turns thesis was right on the mechanism but the
  fix introduced enough other regression to net-cost. Revert path
  is "remove the BASELINE_LEAF_PV_2P alias" — code stays, env-var
  stays default-OFF.
- **μ stabilizes < 950**: 2026-05-18 calibration debt was the
  whole story — re-enabling without fresh calibration regressed.
  Strong evidence to NOT touch the term until we can re-tune
  PRODUCTION_PV_GAMMA + PRODUCTION_PV_HORIZON via dedicated A/B.

## How to read it

Re-check `kaggle competitions submissions orbit-wars --csv` at
session start. Compare against PV_ETA (μ=1163.5) and PEAK RESTORE
(μ=1114.5) which have stabler reads. Note opponent-pool drift in
the meantime — see comp-context.md SCORES DO NOT SETTLE block.
