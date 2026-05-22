# Postmortem — 2026-05-22 review-skills-improvements-moKOR

Wrap-up postmortem per WRAPUP.md section A step 4b and the postmortem
skill. Scored on decision quality given pre-decision priors, not on
outcome (sub 52912707 μ still settling).

## What went wrong

**Bad decisions:**

1. **Parallel A/Bs at workers=4 each on a 4-CPU box.** Ran orbitfix-vs-
   baseline_full and orbitfix-vs-consolidated simultaneously, each
   spawning 4 worker processes — 8 workers on 4 cores. Games stretched
   from typical 150-250s to 300-500s, and the 600s `subprocess.run`
   timeout in `clean_ab.py` fired on a pathological seed-3 game; the
   exception bubbled via `fut.result()` and crashed both scripts. ~25
   min wallclock plus debugging. I knew nproc was 4 at decision time;
   should have run sequentially or scaled workers. (friction tag
   `cpu-oversubscription-kills-ab-throughput`)

2. **Didn't pre-search sibling branches for the bundle-no-agent
   friction.** `claude/strategy-axis-decision-3437` documented the
   exact failure 2 days earlier (c25a329) — root cause and fix were
   in tree. Hit it cold, burned ~30 min before pulling the fix.
   (friction tag `bundle-agent-doesnt-inline-from-baseline-main`)

**PI-overrides (calibration data):**

- "now A/B test against our latest two submissions" — I was ready to
  submit; PI required local A/Bs first. Over-eager to submit once
  unit tests passed.
- "look at individual games rather" — I had been running 32-game
  aggregates; PI redirected to per-game inspection.
- "why so long for 4 games?" — PI flagged sequential vs parallel. I
  had defaulted to sequential interpretation of "individual games."
- "cause over statistics!" — explicit reframing, ratified the small-n
  submit decision (Rule 40 in spirit).

**Rule applications:**

- Rule 1 (PI explicit per-submit) — followed; submit on PI "submit".
- Rule 42 (push-claim board) — filled before submit.
- Rule 43 (multi-opponent panel) — partially honored; PI directed
  single-opponent A/Bs.
- Rule 45 (n≥32 for lift claim) — overridden by PI ("cause over
  statistics"); documented in the orbitfix push-claim row.
- Rule 46 (bundle + parity smoke) — cleared (test_bundle.py 10/10,
  single-game + H2H smokes).
- Rule 38 (fix-verification reproduces failure state) — applied for
  the bundler safety cherry-pick (a) reproduced 0 `def agent` output
  (b) applied fix (c) confirmed refusal + unlink + exit 1.
- Rule 31 (≤2 CPU-heavy jobs) — followed in letter but not in
  spirit: two jobs × four workers ≠ "two jobs". Rule needs the
  workers-per-job constraint.

**Rule-gap:**
- No rule covers "workers-per-job × num-jobs ≤ nproc."
- No rule covers "grep sibling branches' friction.md before
  bundling / before subsystem edits in known-shared infra."

## Frictions logged this session

Cross-references to `audit/friction.md` 2026-05-22 block:

- `bundle-agent-doesnt-inline-from-baseline-main` — fixed in-source
  via cherry-pick of c25a329 (commit 708f197). Plus parametrised
  regression test `tests/test_submissions_loadable.py`.
- `clean-ab-crashes-on-single-game-timeout` — fixed in-source (commit
  38372f4).
- `cpu-oversubscription-kills-ab-throughput` — fix-forward: scale
  workers to nproc / num_concurrent_abs.
- `bundler-rebuild-requires-prepend-recipe` — fix-forward: teach
  bundler the wrapper pattern; today's c25a329 cherry-pick REFUSES
  broken bundles but doesn't generate working ones from the wrapper.

## Promotion candidates (PI not yet ratified — DO NOT promote)

**C1 — Rule 31 refinement.** Total worker processes (sum of `workers`
across all running jobs) ≤ nproc. Two jobs × 4 workers each on a 4-core
box = oversubscription. First articulation today; ate ~25 min wallclock.

**C2 — Bundler wrapper-shim auto-inline.** Third recurrence of the
"wrapper-shim cannot be bundled directly" pattern in 2 days. c25a329's
guardrail REFUSES broken bundles but the manual `prepend env vars to
submissions/baseline.py` recipe is still the workaround. Hits Rule 36's
3-recurrence threshold for promotion.

**C3 — Pre-bundle sibling-branch friction grep.** Before any new bundle
build or subsystem edit, `git log --all --grep="bundle\|agent symbol"`
across sibling branches' last 7 days. Today's bundler debug was pure
waste — sibling branch had the answer 2 days ago. Pattern shape matches
Rule 44 (state-of-truth read before subsystem edits) but for friction
logs not state files.

PI was asked to ratify; stop-hook triggered commit before reply.
Candidates remain in `audit/friction.md` and this postmortem; promote
in a follow-up session when PI signs off.

## PI additions

(None this session; stop-hook triggered commit before the verbatim
"anything to add?" question received a reply.)

## Framework version at session-end

- Commit SHA before this postmortem: `708f197` (bundler safety fix +
  test).
- Active rules: 1–47 (CLAUDE.md).
- Loaded skills this session: `postmortem` (just now), `kaggle-comp`
  (passive).
- Branch: `claude/review-skills-improvements-moKOR`, ahead 43 of main
  (after this postmortem commit).
- Live Kaggle rolling pair: `52912707 baseline_joint_aggr_consolidated_
  orbitfix` (μ TBD, posted 04:56 UTC) + `52894340 _phase4_step1_FND`
  (μ=1117.9).
