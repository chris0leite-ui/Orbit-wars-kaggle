# Postmortem — 2026-06-01 competition-objective-alignment-hqNVM

## What went wrong

- **Bad decision: decorator chain on v7_add_one output.** The approved
  plan and Phase-1 exploration both explicitly recommended SKIP all
  trajectory-side decorators on the v7_add_one branch's output. I
  deviated to "match the ROI branch's decorator chain" on the local
  reasoning that `used_srcs` checks made it safe-by-construction.
  Result: 0/16 vs jsr at n=16. Removing the decorators alone did not
  recover (the other bugs were hiding) but the plan-deviation cost
  was real — burned one full n=16 panel (~6 min compute + diagnostic
  time) on a known-bad design choice.

- **Reasonable-then-falsified decisions (not flagged):**
  - Counting in-flight ledger entries toward `my_ships` in the first
    resolver. Mirrored an existing idiom in chooser_trajectory.py at
    line 1037-1046 (SURPLUS_AGGRESSION). Reasonable copy-paste; only
    falsified by examining seed=0 turn-by-turn ratio trajectory.
  - Trying closed-form ROI as the aggression handoff first. PI's
    diagnosis ("jsr fails to convert") plus existing wired ROI
    chooser made it the cheapest first attempt. ROI's structural
    over-commit was provable only by running it (seed=0 collapsed
    from ratio 1.54 to 0.53 in 9 turns).

- **Rule-bypass:**
  - Rule 38 (fix-verification reproduces failure state) was followed —
    every fix was verified by re-running the failure scenario. The
    decorator-skip fix's RE-verification (still 0/16) actually
    surfaced the bundle name-collision bug. So Rule 38 worked.

- **Rule-gap (PI declined to promote, see "Promotion candidates"):**
  - No rule about bundle name-collision verification at build time.
  - No rule about stderr being swallowed by kaggle_environments
    in eval mode.

## Frictions logged this session

Six new entries appended to `audit/friction.md` under
`## 2026-06-01 (claude/competition-objective-alignment-hqNVM — jsr
aggression-mode-handoff axis exhausted)`:

- `tag: bundle-name-collision-overwrites-imported-function`
- `tag: bundle-collision-internal-late-binding`
- `tag: kaggle-environments-swallows-agent-stderr`
- `tag: roi-resolver-counted-outgoing-inflight-as-reserve`
- `tag: closed-form-roi-myopic-about-cumulative-drain`
- `tag: plan-deviation-decorators-on-v7-output-was-0-of-16`
- `tag: jsr-line-cannot-beat-champion-axis`

## Promotion candidates (PI ratified: NEITHER)

Two candidates were drafted, both DECLINED by PI:

1. **Bundle name-collision audit at build time** —
   `scripts/bundle_agent.py` AST-scan for duplicate top-level
   `def NAME` and warn/raise on rebind risks. Declined.

2. **File-side-channel debug pattern** for per-turn agent state.
   `kaggle_environments/agent.py:185-218` swallows stderr in
   eval contexts; pattern is `if (_dbg := os.environ.get(...)):
   open(_dbg, "a").write(...)` not `print(..., file=sys.stderr)`.
   Declined.

Both remain in `audit/friction.md` for cross-session pattern
detection.

## PI additions

PI added no additional frictions or decisions during the postmortem
step. PI's strategic direction ("wrap up + commit fixes, no submit")
was issued before postmortem began.

## Net session result

- **Submissions used today: 0** (vs daily budget 5). Both rolling-pair
  entries preserved.
- **Live μ change: 0** (no submission).
- **Architecture: addone-v5 stack secured.** v7_search alias + rename
  + dispatch branch + `_addone` wrapper all committed to
  `claude/competition-objective-alignment-hqNVM`. Re-usable for any
  future aggression handoff that wants K=10 rollout add_one scoring.
- **Strategic finding:** jsr-line has an architectural ceiling below
  champion. 5+ attempts today, all 0/16 vs champion locally. See
  `knowledge-base/flags/2026-06-01-jsr-line-architectural-wall.md`
  and `knowledge-base/questions/2026-06-01-what-load-bearing-component-makes-champion-stronger.md`
  for the next-session-priority question.

## Framework version at session-end

- Commit SHA: 5bda9a2119c5532ec318fadb2af82541ea42261d
- Active rules: 1..47 (per CLAUDE.md top-level rules)
- Loaded skills this session: postmortem
- Bundle: `submissions/baseline_pv_eta_vh_dist_jsr_addone.py`
  (1,145,xxx bytes; Rule 46 parity 10/10 GREEN; bundle alias + rename
  fixes verified by 144/144 successful v7_choose invocations in
  seed=0 smoke).
