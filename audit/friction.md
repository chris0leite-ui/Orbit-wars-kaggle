# audit/friction.md — current friction summary

> One entry per distinct friction event. Format:
>
> ```
> - `tag: <kebab-slug>` — <session/day context>: <what happened>.
>   <Root cause>. **Fix:** <concrete action>.
> ```
>
> Reuse existing tags where possible. **One line per event ideal;
> three lines max.** When this file exceeds 150 lines, rotate per
> self-improvement.md::Weekly distillation — archive to
> `audit/friction-archive-YYYY-MM-DD.md` and reset.
>
> Last rotation: 2026-05-14 (claude/audit-workflow-friction-XD56a).
> Full prior history at `audit/friction-archive-2026-05-14.md`.

## Just-landed fixes (claude/audit-workflow-friction-XD56a — 2026-05-14)

These patterns recurred multiple times; the fix is now in source. If
the next session sees them re-fire, the regression test is missing —
fix forward AND add a test.

- `[FIXED] data-main-py-missing-on-fresh-clone` — `bootstrap.sh` now
  guards on `[[ -f data/main.py ]]`, not on "any non-gitkeep file".
  Stops `data/shot_validator/` from masking the download skip.
  Recurred 3x (5/10, 5/12, 5/13).
- `[FIXED] kaggle-cli-401-wrong-auth-env-var` — `bootstrap.sh` now
  detects `KGAT_`-prefix tokens and routes them via
  `KAGGLE_API_TOKEN` instead of `kaggle.json`'s legacy 32-hex path.
  Also maps harness names (`KaggleUserName` / `KaggleAPIToke`) to
  canonical. Plus a `kaggle competitions list -s orbit` cred smoke
  surfaces 401s in minute one. Recurred 3x (5/10, 5/12, 5/13).
- `[FIXED] pip-blinker-system-conflict` — `bootstrap.sh` does a
  `pip install --ignore-installed blinker` preflight before
  `pip install -r requirements.txt`. Stops Debian's record-less
  `python3-blinker` from aborting the whole install.
- `[FIXED] bundler-missing-block-e-modules` (and 5 related tags) —
  `scripts/bundle_agent.py` now raises `RuntimeError` if any
  `from lib.X import ...` strips to an X not in the bundle order
  list. No more silent-NameError bundles. Bundle parity-gate alone
  was insufficient — the historical parity test used `v1_orbitfix`,
  which doesn't exercise the snipe-stack mission framework.
- `[FIXED] bundler-overwrites-tracked-submission` — bundler refuses
  to overwrite a git-tracked file in `submissions/` without
  `--force`. Caught one silent deletion of `v7_0_drop_one.py` on 5/13.
- `[CODIFIED] consecutive-falsification-cap` — Rule 37 in CLAUDE.md.
  3+ consecutive variants in the same design axis ⇒ pivot or
  escalate. Cost evidence: v7_X chooser axis sweep on 5/13-14 burned
  6 h on diminishing-EV variants past v7_3.
- `[CODIFIED] kaggle-kernel-mandatory-two-tier-smoke` — folded into
  Rule 2. Local CPU single-state smoke + small-scale GPU smoke
  before any production T4 push. Cost evidence: 90 min T4 quota on
  a JIT compile a 5-min CPU run would have flagged.
- `[FIXED] local-vs-v7_0-only-misses-ladder-distribution` —
  `fast.py eval --vs-panel` runs a 3-opponent calibration panel
  (`v7_0`, `v4_planner`, `v3.5.1`) by opt-in; PASS verdict requires
  every opponent's Wilson lower bound ≥ gate, not pooled. v3.5.1
  (5/12, -150μ vs prediction) and geo v3.1 (5/14, -80μ floor) both
  passed single-opponent A/Bs and regressed on the ladder.
  **Open follow-up:** make `--vs-panel` mandatory before submission
  (workflow rule, not yet hard-gated in source).

## 2026-05-15

- `tag: pi-hypothesis-unaudited-pre-architecture` — PI proposed
  "iter starts too late, garrison too high" as a root cause for ladder
  losses. I started planning the geo allocator candidate around it.
  Live-game audit (9× iter_v2 2P games) **refuted all three claims**:
  median first launch at step 3.3, 12.9 ships vs opponents' 11.5, 1.7
  ships left at home in steps 0–5 (7.5× drain ratio). **Root cause:**
  no rule requires verifying a PI behavior-hypothesis against recent
  live games before building architecture for it. Rule 26 says "PI is
  read+strategy" but doesn't mandate hypothesis-data verification.
  **Fix:** AskUserQuestion surfaced the contradiction mid-session;
  refine to a pre-architecture audit step. Promotion candidate.

- `tag: build-large-architecture-without-probe-of-value-gradient` —
  MSP shipped ~250 LOC across 4 templates + analytical scorer +
  orchestrator BEFORE testing whether any plan would emit a step-0
  action that beats incumbent's analytical score. Trace at completion:
  0/40 turns produced a winning plan. **Root cause:** built breadth
  (4 templates) before depth (one template's value gradient). Could
  have surfaced the "first-turn action collapses to incumbent" failure
  mode in a 30-min probe. Same pattern as mission-persistence v1
  (−42pp shipped before probe). **Fix:** probe-before-build rule for
  >50 LOC architectural additions. Promotion candidate.

- `tag: value-head-magnitude-uncalibrated` — wired `cluster_value` at
  weight=1.0 in iter without comparing its increment over base to
  `composite_capture_value`'s increment. At +30 turns: cluster
  increment = 164, composite increment = 53.6; cluster dominates
  composite 3:1 in the layered head. First A/B was confounded by
  scale. **Root cause:** TERRITORY_WEIGHT=0.01 is the existing
  scaling precedent for high-magnitude heads, but no rule requires
  checking magnitude before A/B. **Fix:** magnitude-comparison gate
  before any new value-head A/B. Promotion candidate.

- `tag: a-b-compute-budget-not-pretimed` — 32-seed A/B was scoped in
  the plan as "~12 min" but each game with TWO_PHASE=True takes 139s
  (256 turns × 544ms/turn-pair). 600s wallclock budget for 32 games ÷
  4 workers was insufficient; first A/B timed out producing zero
  data. **Root cause:** estimated A/B time without timing one game
  first. **Fix:** before scoping multi-seed A/B compute, time a
  single game with the variant config. Single-event; not yet a
  promotion candidate but flag.

- `tag: chooser-leaf-noise-resists-additive-candidates` — 3 strict-
  additive candidate generators (mission persistence, geo allocator,
  MSP) all failed to lift despite chooser argmax-gating. Decomposed
  geo allocator's −12.5pp at 8 seeds: TWO_PHASE alone is byte-
  identical to baseline; the regression is from the chooser picking
  geo's over-extension candidate in seed=5. **Root cause:** the K=10
  leaf scorer's accuracy is the bottleneck; additive candidates
  amplify ranking error rather than fix it. **Fix:** Rule 37 cap
  fired on this axis; next swing must NOT be another additive
  candidate. Promotion candidate.

## Newly-fired patterns (this session)

- `tag: fix-not-validated-against-real-failing-state` — 2026-05-14
  audit-pass: I patched `bootstrap.sh` for `data-main-py-missing-on-
  fresh-clone`, ran the unit guards (syntax check, AST tests, --help
  output), saw 16 pytest failures with the exact `data/main.py` error
  string the patch was meant to neutralise, and categorised them as
  "pre-existing, not regression" instead of running the patched
  bootstrap. PI caught it: "you have not been able to bootstrap
  properly even though you noticed the friction." Same pattern as
  `agent-introspection-skipped-bootstrap` (2026-05-13). **Root cause:**
  fix-verification protocol was "unit-test the new code path" rather
  than "reproduce the failure state and confirm fix neutralises it."
  The rule was written in friction.md but never bound because friction
  notes don't gate behaviour. **Fix this session:** promoted to
  CLAUDE.md Rule 38; bumped SessionStart bootstrap hook to top
  pending in `improvements.md`.

## Still-open patterns (next-session priority)

- `tag: handover-stale-at-session-start-no-git-log-check` — Rule 32
  already requires session-start `git fetch + git log HEAD..origin/main`.
  Enforcement is aspirational. **Promotion candidate:** SessionStart
  hook (the `session-start-hook` skill exists in this environment).
  Cost evidence: 5/13 LATE wrote a full plan-mode design for work
  already completed on the same branch (`cb02fd9`, `4ba55f4`).
- `tag: jax-vmap-already-wired` — claimed integration was missing
  while it was actually live. **Pattern lesson:** before claiming a
  capability is unbuilt, grep `agents/*_v*_*/main.py` and
  `lib/*.py` for existing wrappers. Specifically applies to
  `score_candidate_jax_pure_jit` (6 ms after JIT).
- `tag: geo-v2-three-failed-wallclock-fixes` — three orthogonal
  attempts to bound the K=10 lookahead's max-wallclock all regressed
  strategy more than they saved time. **Promotion candidate:** when
  a single-knob change costs more than it saves in three orthogonal
  directions, the config IS the local optimum — stop tuning;
  submit if positive, find structurally different lever otherwise.
- `tag: env-clone-cost-grows-with-history` — `env.clone()` cost rises
  4× across an episode (5.6 ms cold → 22 ms warm) because
  `Environment.clone()` walks `self.steps`. Mid/end-game rollout
  cost is hosed; `lib/fast_sim.py` is the working bypass.
- `tag: trueskill-noise-vs-signal` — TrueSkill σ is large for the
  first ~24 h after submit (initial σ≈300, shrinks ∝ 1/√N). Wait
  ≥24 h before reading rank delta into strategy decisions.

## How to add an entry

```
- `tag: <kebab-slug>` — <session context>: <what happened>.
  <Root cause>. **Fix:** <concrete action>.
```

Reuse tags. New tags get one cycle of grace before promotion. If a
tag fires 3+ times, it goes to
`.claude/skills/kaggle-comp/improvements.md` and then into the
relevant skill file or source code, not back into friction.md.

## Anti-spam — what does NOT belong here

- Successful experiments → `audit/YYYY-MM-DD-*.md`.
- LB / rank results → `state/calibration-ladder.md`.
- Hypothesis churn → `state/hypothesis-board.md`.
- Multi-paragraph reasoning → audit postmortem or
  `knowledge-base/thoughts/`.

If something is worth a paragraph, it's not friction. It's a real
postmortem.
