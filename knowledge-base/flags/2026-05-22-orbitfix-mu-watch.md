# 2026-05-22 — orbitfix μ watch

Sub **52912707** baseline_joint_aggr_consolidated_orbitfix posted at
2026-05-22 04:56 UTC. TrueSkill needs ≥6h to settle (Rule 12 caveat
`early-trueskill-mu-unreliable`).

**Watch for:**
- Settled μ at ≥6h. Predicted range 1110–1130 (between consolidated's
  1124 and phase4_step1_FND's 1117.9).
- If μ < 1078 (baseline_full's value): regression. Was the fix wrong,
  or did Kaggle's seed mix favour the broken-position scoring?
- If μ < 1117.9: orbitfix becomes the rolling-pair floor. Next submit
  should NOT evict phase4_step1_FND (μ=1117.9) without a clear lift
  signal.

**Decision rules for the next submit:**
- Settled ≥ 1115 → orbitfix safe; next submit can evict either.
- Settled 1080–1115 → no submit until cause-driven candidate ready;
  don't trade floor for noise.
- Settled < 1080 → revert thinking applies; figure out what regressed.
