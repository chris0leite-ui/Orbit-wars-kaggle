# audit/hypotheses/results.md — pre-registration calibration log

> Append-only. One row per submit. Fill the "PRE-SUBMIT" column at
> registration time; back-fill "POST-SUBMIT" once ≥ 50 episodes settle
> (≥ 6 h per `early-trueskill-mu-unreliable`).
>
> Becomes the empirical calibration table that closes the
> `local-overpredict-2x` friction with data rather than intuition.
> After every ~10 rows, recompute the regression of `live μ delta` on
> `predicted μ delta` per hypothesis class.

## Schema

Each row records:
- `sub_id` — Kaggle submission ID (back-filled).
- `date` — UTC date of submit.
- `agent` — short bundle name.
- `parent_step` — which step of the plan this submit tests
  (e.g. `Step 2 shot validator`, `Step 3 value head`, …).
- `pre_register_doc` — path to the per-submit pre-registration markdown.
- `local_predicted_μ` — the H2 / H3 prediction.
- `actual_settled_μ` — settled mean after ≥ 6 h.
- `μ_delta` — `actual - predicted` (negative = over-predicted).
- `H_results` — pass/fail count for the registered hypotheses.
- `mechanism_metric_delta` — the H4 (non-μ) behavioural metric delta.
- `verdict` — `confirmed` / `wrong_axis` / `refuted` / `mixed`.
- `postmortem_triggered` — y/n (Rule 14).

## Rows

> Add new rows at the TOP (newest first). Use `kaggle competitions
> submissions orbit-wars` to look up sub_id and settled μ.

| sub_id | date | agent | parent_step | pre_register_doc | local_predicted_μ | actual_settled_μ | μ_delta | H_results | mechanism_metric_Δ | verdict | postmortem |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 52744856 | 2026-05-17 | baseline (composite + A2 hybrid, re-bundle) | (pre-plan: composite+A2 pivot) | audit/2026-05-17-pre-submit-hypotheses-composite-a2-hybrid.md | ≥1108 (H2) | TBD | TBD | TBD | TBD | TBD | TBD |
| 52744234 | 2026-05-17 | baseline (composite + A2 hybrid, first bundle) | (pre-plan) | audit/2026-05-17-pre-submit-hypotheses-composite-a2-hybrid.md | ≥1108 (H2) | ERROR (validation failed) | n/a | n/a | n/a | bundle_failure | bundler-modular-agent-namespace-access-breaks-bundle (closed) |

## Notes

- Sub `52744234` errored at validation due to the bundler missing
  agent submodule namespace; sub `52744856` is the re-bundle after
  the fix (commit e7bdefe). Friction tag closed.
- Pre-plan rows are imported for historical calibration; the workflow
  defined in `template.md` applies to all NEW submits going forward.
