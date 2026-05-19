# Postmortem — 2026-05-19 PM trajectory_roi saturation + v4 pivot

## What went wrong

Shipped 5 versions of trajectory_roi over one session, all 0-1/32
A/B vs baseline. Each version added architecture (joint 2-opt,
multi-source bundles, defense candidates, mirror-opp, forward-
projection, longer horizon) on top of analytical primitives that
had never been independently verified. The architecture itself
hit a structural ceiling: forward-projection with `lite_greedy`
as the opp model under-margins captures against the stronger real
opp (baseline), and the benchmark proved that using a stronger
analytical opp inside the projection (mirror-v2) is computationally
infeasible (130-216s/turn against the 1000ms env cap).

The 5-version iteration was the wrong response to "v1 doesn't
work." The right response was to verify the analytics first and
then reconsider the architecture if no bugs were found.

## Frictions logged this session (PM)

All appended to `audit/friction.md § 2026-05-19 PM`:

- `kaggle-loader-picks-last-callable` — `get_last_callable` returns
  the LAST callable in a module; v3's helper after `agent` caused
  the agent to silently emit `[]` for whole games. ~4 inspection
  cycles to localise (env.run debug log P0 duration = 1e-05 was
  the smoking gun).
- `predicate-too-strict-vs-real-behaviour` — DI1 required a single
  ≥50-ship strike; v3's analytically-minimal multi-launch
  deployment of 110 total ships across 5 launches was PASS in
  spirit but FAIL by predicate. Broadened the predicate.
- `env-cap-vs-budget-confusion` — JOINT_SOLVE_BUDGET_MS=700 wasn't
  agent-wide; enumeration + overhead pushed late-game turns to
  1119ms (over env's 1000ms cap → action drops). Lesson: budget
  the WHOLE turn, not a slice.
- `5-version-iteration-without-verification` — the meta-friction.
  Five versions on unverified primitives. Promotes to Rule 41
  candidate.
- `lite_greedy-projection-underestimates-real-opp` — the ARCHITECTURAL
  finding that drove the v4 pivot.

## PI overrides this session

- "We have so much headroom, do we optimize the joint actions?" —
  redirect from v2's 2-opt heuristic to v3's forward-projection.
  Mid-session.
- "Are we really solving analytically? Is there a bug behind it?" —
  end-of-session redirect that produced the v4 plan: verify first,
  then goal-directed planner.
- "Stay in trajectory world" (earlier) — saved us from patching
  baseline; kept us in the analytical lane.
- "Maximum blast radius is the goal" — gave the centrality bonus
  in v3 a strategic anchor (compounded in v4 via portfolio).

## Promotion candidates (PI ratification pending next session)

- **Rule 41 candidate**: "verify before stack — no v_(n+1) on a
  v_(n) that didn't beat random+10%." Rationale: this session
  shipped 5 versions, none of which beat baseline, several of
  which had unverified analytical bugs. Verifying the primitives
  (5 closed-form checks) takes ~150 LOC and ~30 min; not
  verifying cost us 5 iterations × ~1h each. Cost-evidence
  ratio: ~10:1 in favour of the rule.
- **Lint candidate**: agent module last-callable check. Add a
  test that grep's `^def ` in each `agents/*/main.py` and asserts
  the FINAL def is `agent`. Would have caught the v3 loader bug
  at commit time.
- **Rule candidate**: "budget the agent turn, not slices." For
  any agent that calls forward-projection or any expensive
  per-turn work, the time-budget must cover the WHOLE
  `agent(obs, configuration)` execution, not just the inner loop.

## Rule-gap

- The agent function placement (last callable in module) is an
  UNDOCUMENTED requirement of `kaggle_environments`. No existing
  rule covers it. A repo-local lint catches this once but
  agents/future-contributors may regress.

## What was load-bearing this session (positive)

- The analytical-depth benchmark (`audit/2026-05-19-analytical-
  depth-benchmark.md`) was the right move BEFORE designing v3.
  Empirical data on K=50+lite_greedy = 12ms/plan was the
  foundation for v3's design. Mirror-v2-as-opp at 1300-2200ms/plan
  was the falsification of "mirror your full agent as opp."
- The kaggle loader-bug fix is permanent value — found a real
  silent-failure mode in the env contract.
- The Phase 1b scenario substrate (DI1 + G1) is a useful
  regression gate for v4 — independent of whether v4's
  architecture matches v3's.
- PI's questioning ("are we doing analytics right?") forced the
  pivot. Without that question we'd have shipped v3.2, v3.3, ...

## Framework version at session-end

- Commit SHA: `5d88f4b` (v3.1) is HEAD. Wrap-up commit pending.
- Active CLAUDE.md rules: 0-40 (no new rule added this session;
  Rule 41 promotion candidate drafted in `improvements.md` for
  next-cycle ratification).
- Loaded skills this session: `postmortem` (this skill), the
  plan-mode workflow.

## Calibration snapshot

| Predicted (closed-form ROI) | Actual (n=16 A/B vs baseline) | Gap |
|---|---|---|
| v1 expected: marginal positive | 0/32 | unbounded |
| v2 expected: ≥ 4/16 | 1/32 | -3 |
| v3 expected: 8-12/16 (per plan) | 0/32 (latency cap) | -8 to -12 |
| v3.1 expected: 4-8/16 | 0/32 | -4 to -8 |

Pattern: every version predicted positive A/B; all delivered
~zero. Calibration is **systematically over-optimistic** about
analytical-projection-based agents vs a rollout-based agent.
This is the empirical evidence for the architecture-ceiling
finding — and the motivation for v4.

## Next-session entry point

See `/root/.claude/plans/read-the-handover-do-abundant-quokka.md`
and `HANDOVER.md § Day-19 PM`. Copy-paste prompt sits at the
bottom of the plan file.
