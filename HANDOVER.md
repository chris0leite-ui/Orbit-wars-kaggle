# HANDOVER.md — next-session brief

> Last written: 2026-05-19 (end-of-session ROI-pivot wrap) by
> `claude/audit-workflow-performance-btjeK`.
> **Production agent unchanged on the ladder.** Default chooser
> remains `"trajectory"` (`agents/baseline/main.py:38`); the ROI
> pivot is on the dev branch only.
>
> This session attempted to invert the architecture — closed-form
> ROI prior + thin opp-modifier posterior, replacing the trajectory
> rollout. Phases 1-5, Tier 1, Tier 2 all landed; the chooser passes
> 13/14 synthetic oracles. **G3 panel A/B failed catastrophically
> (0/32 vs v7_0, v4_planner, v3.5.1; 8/32 vs the trajectory bundle).**
> Three iteration rounds confirmed closed-form ROI has a structural
> ceiling: closed-form vuln/gross math can't track 2P planet-control
> dynamics, and Tier 2's surrogate opp (`lite_greedy_policy`)
> doesn't match real ladder opponents.

## Live state (snapshot 2026-05-19 AM; drifts)

| Submission | μ | Role |
|---|---:|---|
| **52784853** | **1121.2** | Rolling pair (most recent) — PV off + clean math fixes |
| **52766596** | **1118.8** | Rolling pair (older) — joint v3 |
| 52754310 | 1143.7 | **EVICTED** (trajectory champion) |
| 52744856 | 1149.2 | Evicted (older) |

Floor for push decisions: **1118.8**. `state/current.md` was corrected
this session — earlier sessions had claimed the trajectory champion was
still in the pair; it isn't.

Daily submission budget: 5/19 used **0**. 5 unused.

## Where to start

**Read `audit/2026-05-19-next-session-plan.md` first** — the structured
plan for picking up cleanly. Summary:

- **Phase A (5 min)**: verify default chooser is still trajectory,
  refresh ladder snapshot.
- **Phase B (60-90 min, PRIMARY work)**: validate the hold-feasibility
  filter as a solo lift. It's been default-on since 2026-05-18 PM but
  never the sole change in a submission. A/B `with_holdfeas` vs
  `without_holdfeas` at n=64. If it lifts, push.
- **Phase C (stretch, 30-60 min)**: run the 14 oracle suite under
  trajectory chooser; map which fail and which patterns to backport.
- **Phase D (defer)**: decide ROI's fate (recommend: park as opt-in).

## What's preserved on the dev branch

Branch `claude/audit-workflow-performance-btjeK`, ahead 100+ commits:

- **`agents/baseline/chooser_roi.py`** (~750 LOC) — closed-form ROI
  chooser, opt-in via `BASELINE_CHOOSER=roi`. Includes:
  - Solo ROI with margin × production × PV-held + endgame bonus
  - N-way coalition enumeration (`_best_coalition_for_target`)
  - Source-vulnerability loss (closed-form opp counter model)
  - Defensive coalition post-pass (B reinforces exposed A)
  - Tier 2 forward-sim posterior (rollout top-K via fast_sim)
- **`tests/test_planner_oracles.py`** — 14 oracle scenarios; 13 pass
  under ROI. Tier 2's rollout broke `solo_capture_but_loses_source`
  (joint-rollout support is a bigger refactor).
- **Phase 1-5 + Tier 1 + Tier 2 commits** (`1967743` is the tip).
  All implementation is honest and principled in isolation — the
  failure mode is architectural, not bug-driven.

## Verified findings (will hold next session)

1. **`fast.py bench` "PASS" is wallclock only, not focal-win.** Use
   `fast.py eval` for actual A/B.
2. **Bundler can't handle multi-line `from lib.X import (...)`.**
   Single-line imports mandatory; documented at `main.py:71-76`.
3. **`lib/fast_sim.py::delta_us_minus_them` misses eliminations.**
   At game-end, ship_totals is `{me: N}` and reads as just N — same
   as idle-but-growing baseline. Tier 2 needed `_terminal_value` that
   returns ±1e6 on `snap.fake_env.done`. Pattern is in
   `chooser_roi.py::_terminal_value` for future reuse.
4. **`lite_greedy_policy` as surrogate opp doesn't generalise.**
   v7_0, v4_planner, v3.5.1 each play differently. Any rollout-based
   opp model needs the actual opponent's policy class, not a generic
   greedy.
5. **Rolling pair is rolling LAST 2** (literal). Don't trust state
   files claiming otherwise. Verify via `kaggle competitions
   submissions orbit-wars` every session start.

## What NOT to do next session

- **Don't push the ROI bundle to the ladder.** It loses 0/N vs v7_0.
- **Don't flip the chooser default to `"roi"`.** Production stability.
- **Don't delete `chooser_roi.py` or the new oracle additions.** The
  defensive-coalition / coalition-ROI / oracle-scenario work is useful
  research output worth keeping for now.
- **Don't backport ROI's defensive coalition to trajectory** without
  a session-budget decision — it touches the joint mechanism and is a
  bigger change than a "small safe ship."

## This session's commits (`8cbc9d1` → `1967743`)

```
1967743 chooser_roi: Tier 2 rollout posterior — implemented but doesn't fix G3
dc6a264 chooser_roi: reinforce scoring + contested neutrals + transient vuln
f355143 chooser_roi: partial fixes from G3 diagnosis (not yet competitive)
5594371 chooser_roi: split multi-line lib.scoring import (bundler-safe)
9589954 chooser_roi: Tier 1B — wallclock budget on coalition enumeration
20ca470 chooser_roi: Tier 1A — defensive coalition (B reinforces exposed A)
2187fb2 housekeeping: correct rolling-pair + drop dead import
a926e7c chooser_roi: Phase 5 — ship-count enumeration + xfail conversions
5cdb08d chooser_roi: Phase 4 — source vulnerability + endgame bonus
19e97a6 chooser_roi: Phase 3 — N-way coalition (no rollout)
a8c40f1 chooser_roi: Phase 2 — solo_roi + greedy emit
7ce6de7 chooser_roi: Phase 1 — env-var dispatch + stub
```

11 commits, all on `claude/audit-workflow-performance-btjeK`. Pushed.

## Rule reminders

- **Rule 1**: submissions PI-approved, single-shot, no retry loops.
- **Rule 12**: rolling pair is literal-last-2; verify via kaggle CLI.
- **Rule 32**: session-start git fetch is REQUIRED. Origin/main has
  diverged further (archetype-strategies line); this branch stays
  on the trajectory + ROI track per PI directive.
- **Rule 36**: session-end second-brain update + postmortem. Plan file
  + this HANDOVER cover the doc side; a postmortem note at
  `audit/2026-05-19-postmortem-roi-pivot-failure.md` covers the
  retrospective side.
- **Rule 40**: prefer modeling-correctness over restriction-tuning.
  The hold-feasibility filter (Phase B's primary target) IS a
  restriction. The session's confirmation is that ladder validation is
  the only real test of whether a closed-form restriction generalises.

## Pointers (new this session)

- `audit/2026-05-19-next-session-plan.md` — **THE plan to read first.**
- `audit/2026-05-19-postmortem-roi-pivot-failure.md` — postmortem.
- `agents/baseline/chooser_roi.py` — research code, opt-in via env var.
- `tests/test_planner_oracles.py` — 4 new oracles (no_launch_past_horizon,
  roi_picks_higher_production, n_way_coalition_three_sources,
  hold_feasibility variants).
- `state/current.md` — rolling-pair record corrected; μ snapshots
  refreshed.
- `/root/.claude/plans/okay-we-can-do-elegant-lampson.md` — the original
  ROI pivot plan from this session's planning phase. Useful context for
  any future revival of the ROI direction.
