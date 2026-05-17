# audit/hypotheses/ — pre-submit hypothesis register

> Every submission to Kaggle is pre-registered here. Required by the
> living plan at `/root/.claude/plans/start-to-carefully-layout-witty-gem.md`
> (section "Pre-submit hypothesis registration").

## Why

Local panel + h2h passes are necessary but not sufficient. We've burned
the `local-overpredict-2x` friction tag twice (v3.5.1, geo v3.1 both
over-predicted live by 20-30 pp) and the v17/v18 case where panel passed
but live h2h vs v15 failed. The fix is **pre-registering a falsifiable
live-replay hypothesis before every submit**, so every push becomes a
calibration data point rather than just a μ reading.

## Workflow

**Before** `kaggle competitions submit`:
1. Copy `template.md` → `<sub_id>-<short-name>.md` (sub_id is left
   blank until after submit; back-fill it).
2. Fill in: hypothesis (mechanism, not just μ), metric, expected delta
   vs current baseline, decision rule.
3. PI sign-off.
4. Submit.

**After** the submit lands (≥80 episodes, usually 24-48 h):
5. `python -m scripts.measure_hypothesis audit/hypotheses/<sub_id>-<name>.md`
   pulls replays via `live_episode_summary.py --pull`, runs the
   declared metric from `lib/metrics.py`, appends the row to
   `results.md`.
6. If refuted AND live μ regressed → trigger postmortem (Rule 14).
7. After every ~10 submits, recompute local-vs-live regression. That
   IS the calibration table that closes `local-overpredict-2x` with
   data.

## Files in this directory

- `template.md` — fillable scaffold. Start here.
- `results.md` — append-only log of pre-registrations + post-submit
  measurements. Becomes the empirical calibration table.
- `<sub_id>-<name>.md` — one per submit.

## Reference precedent

`audit/2026-05-17-pre-submit-hypotheses-composite-a2-hybrid.md` is the
format precedent (composite + A2 hybrid, sub 52744234 / re-bundle
52744856). The template below is a stripped-down generic version.
