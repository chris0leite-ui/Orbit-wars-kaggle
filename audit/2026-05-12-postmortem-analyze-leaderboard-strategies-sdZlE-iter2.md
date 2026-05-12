# Postmortem (iter-2) — 2026-05-12 analyze-leaderboard-strategies-sdZlE

> Companion to `audit/2026-05-12-postmortem-analyze-leaderboard-strategies-sdZlE.md`
> (iter-1 postmortem, written after the v3.5 stack regression).
> This iter-2 postmortem covers the work after PI prompted
> "think hard how to do it better and iterate again."

## What went wrong

- **Iter-1 wasted ~30 min on a stack-first build.** I composed all
  four iter-1 Mission classes (opening, drain, gang_up, recapture)
  into one v3.5 agent and ran the 32-seed full-stack A/B before any
  individual ablation. Stack failed 39.1% (Wilson lo 28.1%). The
  per-mission ablation, run AFTER the full-stack regression, showed
  each class failed individually. The priors were available in main's
  audit corpus — v3.3 blanket-eta-fix regressed (42.2%) and v3.4
  NEUTRAL_BONUS=1.5 regressed (28.1%) — but I built the ambitious
  stack first anyway because the PI directive was "be ambitious."
  Decision quality given priors: poor. The PI directive was about
  ambition of empirical reach, not commit-pattern; iter-2 was
  equally ambitious in scope (4 surgical variants + sweep + FFA)
  but produced a submittable agent.

- **PI override that saved the session.** PI typed "think hard how
  to do it better and iterate again" after my iter-1 wrap-up. Without
  that override I would have left the session with no live
  submission and a regression-only narrative. Decision quality data:
  I was too quick to declare the v3.5 stack a session-ending failure.
  Calibration count this session: 1 PI override (this one). Net
  outcome: v3.5.1 PASSES 32-seed gate (68.8%, Wilson lo 56.6%) and
  is now submitted as #52565976. The override was load-bearing.

- **Module-mutation parameter-sweep design had a worker-reuse race.**
  First cut of `agents/v35_iter2/aggressive_sizing_0{6,8,9}/main.py`
  set `base.SHIP_FRACTION = X` at module-import time, relying on
  Python module caching for the value to persist. In multiprocessing's
  worker-reuse, a worker that imported variant_06 (fraction=0.6) then
  reused for variant_08 (fraction=0.8) would observe the LATER
  mutation when later asked to load variant_06 again. Caught at
  design time before launching the sweep; fixed by moving the
  assignment inside `agent(obs)`. Not a regression — a near-miss
  worth logging.

- **No rule-bypass failures.** Rule 1 (single-shot submit) honored.
  Rule 26 (32-seed gate before submit) honored. Rule 18 (claim leaf
  before compute) — the iter-2 work was research-driven from the
  approved plan + iter-1 lessons, not a claimed ISSUES.md leaf.

## Frictions logged this session (iter-2 additions)

Appended under `## 2026-05-12 (analyze-leaderboard-strategies-sdZlE)`
in `audit/friction.md`:

- `tag: stack-first-ablate-later-is-the-wrong-order` — iter-1's
  build pattern (stack 4 classes, A/B the stack, only then ablate)
  cost 5-10 min of compute that per-class-ablation-first would have
  saved.
- `tag: module-mutation-patching-has-worker-reuse-race` — caught at
  design time; fixed in-session.
- `tag: data-main-py-not-fetched-by-bootstrap-recurrence` —
  second recurrence in 2 days. `bootstrap.sh` data-download path
  remains broken; downloaded manually mid-session.

## Promotion candidates (PI ratified: NO promotions this session)

PI was asked to ratify two candidates:

- **`stack-first-ablate-later-is-the-wrong-order` →
  improvements.md** (would have added a rule on when to ablate
  before stacking). **PI verdict: NO** — "Iter-1 already taught me
  this; the same friction was logged earlier today; another
  promotion attempt is noise." Kept as in-comp friction only.

- **`data-main-py-not-fetched-by-bootstrap-recurrence` →
  improvements.md** (cross-comp pattern for sim comps with shipped
  baselines). **PI verdict: NO** — "Just fix bootstrap.sh now in
  this comp; not worth a cross-comp rule." Kept in-comp.

## PI additions (from step 4)

PI: *"Nothing to add."*

## Framework version at session-end

- Commit SHA at iter-2 wrap: `ba0e956bbf2443abea997362b23081c89777930b`
- Branch: `claude/analyze-leaderboard-strategies-sdZlE`
- Active rules: CLAUDE.md `Operating rules` 1-36 (no changes this
  session)
- Loaded skills this session:
  - `kaggle-comp`
  - `postmortem` (invoked twice — once after iter-1 wrap, once now)

## Calibration snapshot (Rule 5b)

Iter-2 produced a LIVE submission:
- Predicted (local-A/B math, 32-seed Wilson lo 56.6%): expected live
  μ ~1090-1100 in 24h.
- Actual: **PENDING** (validation episode running, #52565976
  submitted 2026-05-12 05:20:09 UTC).

Calibration will close in the next session (Rule 5b: predicted vs
actual μ over the rolling-last-2 window).

## Session-end status

- Branch `claude/analyze-leaderboard-strategies-sdZlE`: ahead 7 /
  behind 2 vs origin/main.
- Rolling-last-2: [σ-equivariance #52565034 μ=976.3,
  v3.5.1 #52565976 PENDING].
- Submissions used today: 1 (v3.5.1, PI-approved).
- Submissions used total: 7.
