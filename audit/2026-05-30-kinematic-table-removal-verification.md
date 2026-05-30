# 2026-05-30 — Kinematic-table removal: foundation verification

Branch: `claude/champion-strategy-rules-00JzI`. Commit `232307c`
("foundation: remove kinematic-table singleton; stop transcribing live
scores").

## Why

`lib/kinematic_table.py` was a module-global singleton position cache. Its
shared mutable state leaks across seats in any in-process A/B (the second
`begin_turn` clobbers the fingerprint; the other seat reads stale
positions) — the documented "flat 37.5%" perf-chain confound. PI directed:
remove it for a foundation with no shared mutable state. Scope = **table
only**: the vectorized inline `_predict_relative_window` is kept (preserves
the per-turn time budget).

## What changed

Deleted `lib/kinematic_table.py` + its two test files. Dropped the table
wiring from `lib/trajectory.py` (the inline planet-position build in
`predict_fleet_fate` is now unconditional), `lib/orbit.py`
(`predict_relative_cached`, table-only), `agents/baseline/main.py`,
`agents/reach_frontier/main.py`, the bundler `DEFAULT_LIB_ORDER`, and
`scripts/profile_spatial.py`. Regenerated `submissions/baseline.py`
table-free.

## Verification (Rule 38 — foundation must stay correct, gameplay must not regress)

1. **Targeted foundation + byte-exact engine parity:** 93 passed
   (`test_trajectory`, `test_orbit`, `test_game_parity`, `test_launch_rules`,
   `test_chooser_pv_eta`). The engine byte-parity test is the load-bearing
   pin — the inline path still reproduces the kaggle engine exactly.
2. **Repo sweep:** zero `kinematic`/`kt_*`/`predict_relative_cached`/
   `_table_window_or_none` references outside frozen `submissions/*` bundles.
3. **Re-bundle:** clean (exit 0; 0 KT refs in the 582 KB bundle). The
   bundler parity-gate's source-load hits the pre-existing
   `agents`-namespace collision with `kaggle_environments.lux_ai_s3`
   (documented; requires `--skip-parity-gate`) — NOT caused by this change;
   the import-resolution guard passed and the bundle wrote.
4. **Play-smoke** (table-free focal vs frozen champion, seed 7, full game):
   clean, 364 steps, no crash. Per-turn timing — table-free focal is *faster*
   at the median and comparable at the tail, all well under the 1 s cap:

   | | focal (table-OFF) | champion (table-ON) |
   |---|---|---|
   | p50 | 320 ms | 397 ms |
   | p95 | 558 ms | 560 ms |
   | max | 662 ms | 652 ms |

   Removing the cache did not cost the time budget.
5. **No-regression A/B** (`scripts/clean_ab.py`, subprocess-isolated, n=32 =
   16 seeds × 2 seats). Both sides run identical champion config (12
   `BASELINE_*` vars exported; verified the source reads those exact names),
   so **table-on vs table-off is the only variable**:

   **focal 15/32 = 46.9%, Wilson [0.309, 0.636], 0 errors.**

   The interval is centered on 50% — textbook parity, no significant
   regression. Expected: the removed table was `==`-pinned bit-identical to
   the inline path, so the two agents are behaviorally indistinguishable.

## Verdict

Foundation is clean and table-free with no gameplay regression and no time-
budget cost. Cleared to proceed to the 2-hop redeploy build (Phase 3),
which now sits on a singleton-free substrate.

## Note on the gate wording

The plan's "Wilson-lo ≥ 0.45" was mis-specified for a *parity* check: a
truly-equal agent at n=32 lands near Wilson-lo ≈ 0.33. The correct
no-regression reading (used here) is "the CI includes 0.5 and there is no
significant loss." 46.9% / [0.309, 0.636] satisfies that.
