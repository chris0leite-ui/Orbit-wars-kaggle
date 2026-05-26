# 2026-05-26 PM — flags (observations from sa_online warm-start cycle)

Standing items that may persist across sessions. Observations only.

- **Rolling pair currently fragile.** `{53062327, 53063161}` at
  session close. Both PENDING. Both untested on live ladder.
  Next push will evict 53062327. If 53063161 also ERRORs the
  pair will be {ERROR, low-μ-sub}.

- **Plan file at `/root/.claude/plans/wiggly-singing-elephant.md`
  rewritten this session.** The prior plan content (cascade-aware
  admissibility + ALNS) had already been implemented and merged;
  the file now describes the warm-start design. Other sessions
  reading the plan file may expect the prior content.

- **Test failure `test_admissible_set_only_physics_valid` pre-
  existed this session.** Surfaced because I ran the full test
  suite; the test iterates 4 seeds in one process and the
  module-level `_FATE_CACHE` (introduced in commit `4165227`
  earlier today) was not reset between seeds. Added an autouse
  fixture in `tests/test_sa_core.py` that calls
  `reset_fate_cache()` before each test. Production behavior is
  unaffected (1 episode = 1 process).

- **Numba is available on Kaggle but not in local dev env.**
  Confirmed by failed `import numba` locally. Means any numba-
  based optimization can be tested only via Kaggle submission or
  by installing numba locally first.

- **`_co_evolve` divergence local vs Kaggle.** Locally consumes
  ~35s at module load and pre-fills `_PLAN_BY_TURN`. On Kaggle
  fails silently (recursive `make()` blocked). Smokes that don't
  set `SA_COEVOLVE_CYCLES=0` test a different code path than
  what ships.

- **Sa_online's per-turn behavior depends on `SA_REFINE_OPP_POLICY`
  default.** Changed from `agents/simple/nearest.py` to `noop` in
  commit `147d6aa`. Older audits / postmortems referencing
  "nearest" as the opp model no longer apply to current behavior.

- **CLAUDE.md Rule 12 caveat is active.** Kaggle keeps the
  rolling last 2 submits for final evaluation, not PI-selected.
  4 of today's 5 submissions consumed (this session). Tomorrow's
  5/day budget is fresh, but the rolling-pair eviction risk from
  any push remains.
