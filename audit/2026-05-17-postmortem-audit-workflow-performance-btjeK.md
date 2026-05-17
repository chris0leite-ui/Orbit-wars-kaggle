# Postmortem — 2026-05-17 audit-workflow-performance-btjeK

Session: claude/audit-workflow-performance-btjeK (PM, this afternoon).
Branch HEAD at write: `d0ac842` (state-update commit).

## What went wrong

Nothing egregious. Three "minor" entries by decision-quality framing:

- **Comet-collision sub-classifier was built before the PI hypothesis
  was cheaply validated.** I implemented `_comet_swept_hit_at` in
  `scripts/episode_postmortem.py`, ran replay-mine on v15's 92
  episodes, observed only 12/9507 fleets classified as comet
  collisions, THEN ran the diagnostic that revealed the real cause
  (orbital planet hits the static `best_d < 5.0` check missed). A
  10-line replay dump before the full implementation would have shown
  "obs_prev has 0 comet groups for fid 198" and pointed at the
  bug-in-classifier-not-fleet-strategy hypothesis sooner. Cost:
  ~30 min on the wrong hypothesis. Output was still net positive
  (the swept-pair classifier IS the right fix), but the order of
  operations was hypothesis-build-then-validate when it should have
  been hypothesis-validate-cheaply-first. Same priors test: would
  re-take this with one-shot diagnostic first.

- **Re-bundled the agent twice.** First bundle shipped with
  `BASELINE_VALUE_HEAD` unset → default `favor` head → missed the
  composite-in-2P lift the A/B validated. Caught at re-test, added
  `os.environ.setdefault` to `agents/baseline/main.py`, re-bundled.
  Cost: ~5 min and one bundle cycle. Avoidable: thinking through
  "what env will the Kaggle runner see" before the first bundle.

- **Initial cap-shrinks test asserted the wrong invariant after #2
  (adaptive WorldModel horizon).** After making composite ~free on
  empty-fleet boards, the test that asserted `cap_composite <=
  cap_favor` failed on a step-0 snap. Updated to construct an
  in-flight fleet first; passed thereafter. Cost: ~10 min iteration.
  Test was correct in intent but didn't carry the invariant through
  the optimization that landed an hour later. Same priors test: not
  preventable without lookahead.

PI-overrides taken cleanly: 3 (adaptive-tier phrasing, "go 4P"
directive, "check the sibling branch" instruction — the last opened
up the A2 merge that would otherwise have been missed).

Rule-bypass failures: 0.

Rule-gap failures: 2, both surface in promotion candidates below.

## Frictions logged this session

`audit/friction.md` appends under
`## 2026-05-17 (claude/audit-workflow-performance-btjeK)`:

- `kaggle-cli-401-in-followup-shells` — session-start hook installs
  `$HOME/.local/bin/kaggle` shim that re-derives `KAGGLE_API_TOKEN`
  on every CLI invocation.
- `vanished-in-space-was-classifier-bug-not-comets` — replaced the
  earlier `vanished-in-space-dominates-trajectory-waste` tag after
  the swept-pair classifier showed 0.1% comet rate, not 8.8%.
- `composite-head-2p-only-no-4p-opp-aggregation` — flag filed at
  `knowledge-base/flags/2026-05-17-composite-value-head-2p-only.md`.
  Closed by the A2 merge (`favor_hybrid` dispatcher; 4P → A2-favor).
- `composite-head-wallclock-over-1000ms-on-heavy-turns` — fix
  shipped (`affordable_validate_cap` now probes per-leaf cost; max
  turn-ms dropped 1292 → 1196 → 1580-under-contention; p95 stayed
  under 800ms).

## Promotion candidates (PI ratified: yes / yes)

PI: "promote as suggested."

1. **`fast.py --require-h2h` skip-by-name misses env-var dispatch.**
   Promoted to `.claude/skills/kaggle-comp/improvements.md` pending
   list. Tag `require-h2h-skip-by-name-misses-env-var-dispatch`.
   Fix: include the env-var snapshot in the focal-vs-opponent
   identity check, or accept `--force-h2h`. Recurrence risk applies
   to every modular-agent + env-var-dispatch combination — the new
   production default is this pattern.

2. **Bundler should inject submission env manifest.** Promoted.
   Tag `bundler-ships-with-wrong-default-env-var`. Fix: extend
   `scripts/bundle_agent.py` to read `agents/<name>/SUBMISSION_ENV`
   (simple KEY=value lines) and emit matching `os.environ.setdefault`
   lines at the top of the bundle. Eliminates the per-agent
   "did you remember setdefault?" gotcha.

Both entries land in the same commit as this postmortem.

## PI additions

> "promote as suggested" — interpreted as ratification of both
> candidates with no additions. No new frictions surfaced by PI.

## Framework version at session-end

- Commit SHA: `d0ac842` (will be superseded by the postmortem
  commit itself).
- Active rules: 1..40 (CLAUDE.md). Rule 38 (fix-verification
  reproduces failure state) and Rule 40 (modeling-correctness over
  restriction-tuning) load-bearing this session.
- Loaded skills this session: `kaggle-comp` (via session-start
  hook), `postmortem` (this artifact).

## Strategic implication (not strictly required, kept brief)

Composite head + A2 is the wholesale architectural change the
2026-05-17 fleet-efficiency negative-result session said was
needed. First lift past the v9_scavenge ceiling (93.8% vs the
team peak). Submission is BUNDLE-READY (286 KB, parity OK over
712 turns, hybrid default baked in) but **NOT submitted** —
Rule 1 holds; PI sign-off required.

Outstanding before any live push:
- Max turn-ms still 1196-1580 (over the 1000ms env cap on heavy
  turns). Engine drops over-budget actions; doesn't kill agent.
  Acceptable risk if PI signs off; or do the deeper WorldModel-
  reuse refactor.
- 4P FFA panel still running at session-end (started ~mid-session,
  ~50% complete on focal=baseline-with-A2). Result will land in
  `audit/tournaments/ffa-panel-<utc>.json` independent of this
  session.
