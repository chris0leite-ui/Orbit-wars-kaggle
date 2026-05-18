# FLAG: bug #15 v2 (PV term) shipped as production default; regressing 39.6% vs bundle

Date: 2026-05-18

## The hazard

Commit `b285882` (Bug #15 v2: PV-term-only) landed with the
production-PV term enabled by default in
`lib/value_heads.py:composite_capture_value`. The kill-switch
`_COMPOSITE_PV_ENABLED` is wired (env var `COMPOSITE_PRODUCTION_PV=0`)
but the default is `1`. A/B at n=96 vs the pre-fix bundle landed at
**39.6% winrate** (Wlo=0.304, Whi=0.496, verdict FAIL).

This means production code (`agents/baseline/main.py` via
`composite_capture_value`) is currently regressed vs the pre-#15-fix
baseline. The 1141.0 trajectory champion submission (52754310) is
unaffected because it's already on the ladder, but any new submission
built off this branch would inherit the regression.

## Why we shipped it anyway

The sanity oracle (`test_oracle_sanity_trivial_capture`) PASSES with
the PV term and FAILS without it. The team wanted that property
preserved during the bug #14 fix investigation, which I'd
hypothesised was the upstream cause of the regression. That
hypothesis is now fully falsified (option 5 v2 A/B also failed at
exactly the same 39.6%) — so the rationale for keeping PV default-on
is gone.

## Next session must

Flip `_COMPOSITE_PV_ENABLED` default to `False` in `lib/value_heads.py`
OR set `os.environ.setdefault("COMPOSITE_PRODUCTION_PV", "0")` in
`agents/baseline/main.py`. Either move restores the chooser's
pre-#15 calibration. Verification: bench (should match toggle-off
numbers) + small A/B (Wlo ≥ 0.45 is acceptable since pre-fix was 50%
by definition).

If the PV term is going to be re-introduced later, it needs paired
chooser-gate recalibration — see
`knowledge-base/thoughts/2026-05-18-PV-term-recalibration-debt.md`.

## Pointers

- `lib/value_heads.py:160-200` — PV-term implementation.
- `lib/value_heads.py:177` — toggle.
- `audit/2026-05-18-postmortem-bug-15-v2-and-bug-14-option-5.md`
  — full session postmortem.
