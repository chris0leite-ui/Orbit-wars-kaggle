# Postmortem — 2026-05-12 fix-early-game-strategy-YSClQ (session 2)

Late-PM session after the first YSClQ wrap-up (commit 517bf3b). Started
when PI surfaced the v4_planner submission this branch had missed.

## What went wrong

1. **Stale state at session start — missed the actual ladder leader.**
   At session start, this branch's `state/current.md` claimed
   v7_minimax (μ=1063.0) was TEAM PEAK. PI surfaced that v4_planner
   (#52579863, **μ=1118.8**) had been submitted earlier today at 14:25
   UTC from the parallel branch `claude/research-lookahead-strategy-
   kfRsy`. **+83 μ I didn't know about.** This is exactly the failure
   mode the `live-mu-pull-at-session-start` rule I promoted to
   `improvements.md` at 15:27 was designed to catch — but the rule is
   filed as `[ ]` pending and no session-start hook implements it.
   Rule-bypass failure: rule existed, was applicable, not applied.

2. **v4.5_robust design hypothesis falsified at the gate.** Designed
   v4.5 as "best of both worlds" = v4_planner framework + v7's maximin
   robustness. Sold PI on Option C (synthesize maximin-over-opp-models
   into v4's scoring loop, K reduced to 4-7 to fit budget) on grounds
   that v4's value head + candidate diversity should let v7's
   adversarial robustness express better. 16-seed A/B vs v4_planner:
   **50.0% pooled WR, Wilson 95% [33.6%, 66.4%] — FAIL gate**. Strong
   seat asymmetry (P0 = 44 %, P1 = 56 %) and loss-magnitude greater
   than win-magnitude (mean loss ~-3500, mean win ~+2900) point to
   K-cap depth-loss outpacing maximin gain. **Decision-quality at
   priors:** the design was defensible — v7 was on the ladder, v4 was
   on the ladder, the synthesis was structurally sensible. The lesson
   is that doubling per-portfolio scoring cost requires K cuts that
   should be ablated separately, not stacked.

3. **Bundle DEFAULT_LIB_ORDER still not auto-discovered (third time
   this comp).** v4.5_robust's first bundle silently produced a 95.7 KB
   artifact missing lookahead/lookahead_planner/candidate_portfolios
   because they're not in the hand-maintained list. The bundle imported
   cleanly and would have crashed at first call of
   `score_joint_action_symmetric`. Caught by counting module markers
   versus v4_planner's bundle. **Rule-gap failure:** the AST-discovery
   improvement candidate has been in `improvements.md` since 2026-05-11
   and isn't implemented yet.

4. **Stale bundle processes ate CPU during the A/B.** Two earlier
   bundle invocations sat at 99 % CPU each for >5 min past artifact
   write, on a post-bundle parity-check loop. Dropped my 4 A/B workers
   from ~95 % to ~64 % CPU each — extended the A/B wallclock by ~30 %.
   Killable, but I didn't notice until I checked `ps aux`. **Fix:**
   bundle script's parity check should have a built-in timeout.

5. **Seat-symmetric scoring didn't fully cancel seat bias in v4.5.**
   `score_joint_action_symmetric` is supposed to average over both seat
   assignments, but v4.5's win-rate split 44 % P0 / 56 % P1.
   Hypothesis: σ-equiv patches cover the planner tie-break but the
   value head and rollout-policy invocations introduce residual
   asymmetry. **Forward implication:** σ-equiv must be audited
   pathway-by-pathway, not relied on as a single wrapper-level
   guarantee.

## Frictions logged this session

All appended under `## 2026-05-12 (PM late — fix-early-game-strategy-
YSClQ session 2)` in `audit/friction.md`:

- `stale-state-missed-parallel-branch-submission`
- `bundle-DEFAULT_LIB_ORDER-still-not-auto-discovered`
- `stale-bundle-process-eats-AB-cpu`
- `v4.5-robust-design-hypothesis-falsified`
- `seat-symmetric-scoring-doesnt-fully-cancel-bias`

## Promotion candidates (PI ratified: pending)

### [ ] [CODE-COMP-DISCOVERED] WRAPUP step 1: enforce live-μ pull (not pending TODO)

**Tag:** `live-mu-pull-must-be-blocking-step` (Orbit Wars 2026-05-12).

**Where to insert:** `improvements.md` — upgrade the existing
`live-mu-pull-at-session-start` candidate from `[ ]` to `[~]`
(implementation underway) and rewrite acceptance:

**What to add:**

Acceptance must be: `scripts/check_live_mu.py` exists, `.claude/
settings.json` calls it from a `SessionStart` hook, and the output
auto-diffs against `state/current.md::our_best_rank` printing a
`MU-DRIFT` warning if any rolling-last-2 entry's μ differs from the
state file by > 10. The hook must run before the agent answers the
first user prompt.

**Why:** the original `[ ]` candidate's existence was insufficient.
This session lost ~30 min on opener variants and 90 min on v4.5_robust
construction against a baseline I thought was v7_minimax — actually
v4_planner was the live ladder leader the whole time.

### [ ] [CROSS-CUTTING] Ablate dual changes separately before stacking

**Tag:** `dont-stack-design-changes-without-component-ablation` (Orbit
Wars 2026-05-12).

**Where to insert:** `improvements.md` under `## Pending`.

**What to add:**

When a design change has TWO simultaneous components (e.g. v4.5 =
"maximin-over-opp-models" + "K reduced 6-10 → 4-7"), require that each
component be ablated separately (one variant with maximin alone, one
variant with K-cut alone) before combining. If the joint variant FAILS
a gate, you cannot diagnose which component is responsible. v4.5's
50 % pooled result is consistent with EITHER (a) maximin is neutral and
K-cut hurts, (b) K-cut is neutral and maximin hurts, or (c) both hurt
slightly — and the postmortem can't distinguish.

Acceptance: CLAUDE.md Rule 21 (family falsification ≥3 variants)
extended with sub-clause "if a candidate combines ≥2 hypotheses,
require ≥1 ablation per hypothesis isolated from the others." Cost is
2× the variant-build budget but recovers diagnosability.

**Why:** the v4.5 fail leaves us without a clear next step. We don't
know whether maximin scoring is intrinsically wrong in this game or
whether the K cut was the killer. Cannot pivot intelligently.

### [ ] [CODE-COMP-DISCOVERED] Bundle parity check needs built-in timeout

**Tag:** `bundle-parity-check-runaway` (Orbit Wars 2026-05-12).

**Where to insert:** `improvements.md` under `## Pending`.

**What to add:**

`scripts/bundle_agent.py`'s post-bundle parity-validation loop should
have a hard timeout (e.g. 60 s). Today two bundle invocations sat at
99 % CPU for >5 min each past artifact write, eating cycles from the
in-progress A/B. The bundle artifact itself is already written when
the parity check starts; if the check times out, the bundle is still
usable (we already have a separate manual parity test in
`scripts/run_*_ab.py` flows).

**Why:** stalled bundle processes are silently the most common form
of "why is my A/B running so slowly?" friction this comp.

## PI additions (from step 4)

(pending — will be filled in after PI replies)

## Framework version at session-end

- Commit SHA: `1b43577d7e52db6b7245381aa907024217094d02`
- Active rules: 1..36 (CLAUDE.md `## Operating rules — concise`),
  R1/R2/R5/R7 from prior-comp postmortem + R8 end-of-comp logging.
- Loaded skills this session: `postmortem`, `kaggle-comp` (referenced
  for the rule discipline but not explicitly invoked).
- Branch: `claude/fix-early-game-strategy-YSClQ`.
- Submissions used today: 4 (σ-equiv-v1, v3.5.1, v7_minimax,
  v4_planner). Rolling-last-2 unchanged: `[v7_minimax 1035.5,
  v4_planner 1118.8]`. No new submissions this session.
