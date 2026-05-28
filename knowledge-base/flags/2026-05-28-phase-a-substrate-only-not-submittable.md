# 2026-05-28 — flag: Phase A artifact is substrate-only, NOT submittable

**Surface this on every Phase A → Phase B handover** so the next agent
does not accidentally promote `baseline_learned` (Phase A bundle) to a
Kaggle submission slot.

## What the flag is

`submissions/baseline.py` and `agents/baseline/value_learned.py` now
embed the Phase A distilled weights. The bundle works; the agent
plays clean games; latency is in budget. It is technically pushable.

**Do not push it.** It is calibrated to match `favor_hybrid`
(μ=1149 EVICTED) at parity. A learned head that faithfully mimics a
hand-coded head adds inference cost and zero ladder upside. It is a
**substrate diagnostic**, not a competitive agent.

## When it WOULD be submittable

After Phase B:
1. Re-trained on advantage + CRN + multi-horizon + strong opponent
   pool (per HANDOVER.md Phase B section).
2. Cleared Rule 43 multi-opponent panel at Wilson-lo ≥ 0.55 per opp.
3. Cleared Rule 45 n ≥ 32 vs current rolling champion at Wilson-lo
   ≥ 0.50.
4. Cleared Rule 46 bundle + parity smoke.
5. Cleared Rule 42 cross-branch coordination claim board.
6. Explicit PI sign-off (Rule 1).

Until ALL of those clear, the answer is no.

## Cost evidence if this gets violated

The 2026-05-20 five-step rolling-pair eviction chain cost ~320 μ of
ladder floor in 24 h (state/MULTI_BRANCH.md). Pushing a substrate-
only Phase A bundle would replay the same failure mode — a known-
parity artifact evicting whatever's currently in the rolling pair,
unrecoverable for ~24 h.

## Clearance

This flag clears the moment a Phase B candidate ships under PI sign-
off. Until then, mention it in any session-start handover read.
