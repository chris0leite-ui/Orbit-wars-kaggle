# Flag: synthetic-baseline A/Bs misleading vs live-agent A/Bs

**Status**: ACTIVE, structural

The 2026-05-22/23 session ran 7 A/Bs of α+β stacked vs derivatives
of `analytical_phase_c` (alpha_beta_off, maximin_off, smooth_dw_off,
etc.) at point estimates in [0.50, 0.625]. The two A/Bs vs the
actual live submissions (`_phase4_step1_FND`, `orbitfix`) gave
3/8 = 37.5% — directionally NEGATIVE.

Synthetic baselines isolate "did my code do anything" but don't
predict ladder behaviour. Any future session that runs only
synthetic A/Bs is likely to ship a variant that nulls (or
regresses) on the actual ladder. Rule 43 already says re-pull μ
at session start; an analogous rule should say "compare against
the live agent bundle, not a derived no-features variant, for any
A/B that intends to gate a submission."

**Candidate Rule**: A/Bs that intend to gate a submission MUST
include the current live rolling-pair leader as one opponent.
Synthetic-baseline A/Bs are diagnostic, not gating.

Origin commit ref: this session's pushed audit at
`audit/2026-05-23/items-1-3-4-5-execution.md`.
