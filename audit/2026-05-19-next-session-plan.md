# Plan — next-session direction (synthesised from 2026-05-19 ROI session)

> Last session (2026-05-19): pivoted to ROI-prior + opp-modifier
> architecture. Phases 1-5, Tier 1 (defensive coalition + wallclock
> budget), Tier 2 (forward-sim posterior on top-K) all implemented.
> All committed and pushed on `claude/audit-workflow-performance-btjeK`.
> Result: G3 panel A/B failed catastrophically (0/32 vs v7_0,
> v4_planner, v3.5.1; 8/32 vs the trajectory bundle).
>
> Root finding: closed-form ROI cannot match the dynamic balance
> that fast_sim achieves in trajectory chooser. Even with Tier 2's
> rollout posterior, the surrogate opp (`lib.opp_model.lite_greedy_policy`)
> doesn't match real opponents' play, so the rollout's "measured
> opp response" is also wrong-shaped.
>
> Production agent unchanged on the ladder. ROI work is on the dev
> branch only.

## Context — where we are

- **Current ladder floor**: 1118.8 (submission 52766596 in the rolling
  pair with 52784853). 52754310 (trajectory champion at 1143.7) has
  been evicted. Rolling-pair record at `state/current.md` was corrected
  this session.
- **`BASELINE_CHOOSER` default**: `"trajectory"` (unchanged at
  `agents/baseline/main.py:38`). The ROI chooser is opt-in via env var.
- **`submissions/baseline.py`** (May 18 17:29 mtime): the pre-Phase-1
  trajectory bundle. Represents current ladder behaviour for h2h
  comparisons.
- **Branch**: `claude/audit-workflow-performance-btjeK`, ahead 100+
  commits of origin/main (which is on the simpler v15/archetype-strategies
  line per PI directive — we stay on the trajectory + ROI track).

## The session's verified learnings

These are real findings that will hold regardless of next-session direction:

### About the codebase
- **`fast.py bench`'s "PASS" is wallclock only, not focal-win.** Line 725-726:
  `verdict = "PASS" if p95 < 800 and over == 0 else "WATCH"`. Game outcomes
  (`outcome=p1_win` etc.) are reported but don't gate the verdict. I misread
  bench as A/B during G2 and reported a false-positive. **Use `fast.py eval`
  for A/B; `fast.py bench` is for wallclock only.**
- **Bundler can't handle multi-line `from lib.X import (...)`.** Friction
  `bundler-modular-agent-namespace-access-breaks-bundle` documented at
  `agents/baseline/main.py:71-76`. Single-line imports are mandatory in
  any agent module that gets bundled.
- **`lib/fast_sim.py::delta_us_minus_them` doesn't capture eliminations.**
  After we capture opp's last planet, `snap.fake_env.done = True` and the
  rollout stops; ship_totals at that point is `{me: N, opp: 0}`. The delta
  reads as just N, which an idle-but-growing baseline can match. Tier 2
  needed a `_terminal_value` wrapper that returns ±1e6 on game-end
  (`agents/baseline/chooser_roi.py::_terminal_value`).
- **Original `solo_capture_but_loses_source` test scenario is structurally
  undecidable** for closed-form ROI: A=110, B=80, equal productions, single
  opp → opp can only counter one source, so reinforcing A trades A's loss
  for B's loss. Net is zero before ship cost. Phase 5 conversion required
  bumping A's production to 2 (asymmetric value).

### About the architecture
- **Closed-form ROI has a structural ceiling.** Three rounds of iteration
  (Phases 1-5, then Tier 1, then Tier 2 + vuln calibration + reinforce
  scoring + transient vuln) all came back 0/N vs v7_0. The pattern:
  every modeling fix is correct in isolation but doesn't close the gap
  because the closed-form vuln/gross math can't track the actual game's
  dynamics.
- **Tier 2 rollout posterior didn't fix it** because the surrogate opp
  (`lite_greedy_policy`) plays differently from any actual ladder opponent.
  The rollout measures "what does lite_greedy do?" but the actual game is
  against v7_0 (with its own fast_sim + drop-one chooser), v4_planner
  (receding-horizon mission portfolio), v3.5.1 (aggressive snipe), etc.
- **The trajectory chooser at μ≈1120 is doing real work** with its
  fast_sim rollout + composite leaf. Throwing that away in favour of
  closed-form ROI was a bigger commitment than the session budget.

### About what survives
- **14 synthetic oracle scenarios** in `tests/test_planner_oracles.py`
  encode real game properties (hold feasibility, drain frontier, defense
  against multi-fleet, coordinated capture, last-opp finish, etc.). These
  are useful regardless of which chooser is active. 13 of 14 still pass
  under ROI (Tier 2 broke `solo_capture_but_loses_source`).
- **`_target_holdable_after_capture`** (proposer.py:407) — the Tier 2
  filter from 2026-05-18 PM. Active by default since then, but NEVER
  ladder-tested as the sole change. This is the cheapest possible lift
  to validate next session.
- **`_source_survives_launch`** (proposer.py:355) — drain-frontier filter.
  Same story: default-on, never solo-validated.
- **`chooser_roi.py`** (~700 LOC of research code) — parked on the branch.
  The closed-form opp model, N-way coalition enumeration, defensive
  coalition, endgame elimination bonus, and ship-count variant
  enumeration are all useful concepts that might re-emerge later.

## Next session — the plan

Three phases, ordered by safety + ROI.

### Phase A — Reset to known-good (5 min)

1. Verify `agents/baseline/main.py:38` is still `setdefault("BASELINE_CHOOSER", "trajectory")`.
2. Verify `submissions/baseline.py` is the May 18 trajectory bundle (sha256
   prefix `a434b56...` or whatever the mtime confirms).
3. Run `python -m pytest tests/test_planner_oracles.py -q` under default
   chooser. Note which pass/fail — sets the trajectory-baseline truth
   table for Phase C.
4. Read `kaggle competitions submissions orbit-wars` to refresh μ snapshots
   (per Rule 32 session-start fetch).

### Phase B — Ship the hold-feasibility lift (PRIMARY work, 60-90 min)

The most-likely-to-help SMALL change. `_target_holdable_after_capture`
(proposer.py:407) has been default-on since 2026-05-18 PM but has never
been the sole change in a submission — every submit since then has
bundled it with OTHER changes (PV terms, joint v3, etc.). We don't
know if it alone lifts μ.

**Step B.1 — Build the control bundle.**

```
# Trajectory + everything EXCEPT hold-feasibility.
# Strategy: bundle current source, then sed-rewrite the bundle to
# remove (or env-disable) the hold-feasibility check at proposer.py:627.
git checkout 82df5b8 -- agents/baseline/proposer.py  # pre-Tier-2-filter version
python scripts/bundle_agent.py agents/baseline --out-dir /tmp --force --skip-parity-gate
mv /tmp/baseline.py /tmp/baseline_no_holdfeas.py
git checkout HEAD -- agents/baseline/proposer.py  # restore current
```

Alternative if the above breaks: bundle current and rewrite the
`setdefault("PROPOSER_HOLD_FEASIBILITY", "on")` line to `"off"`. Even
simpler: set env-var override in the bundle's import header.

**Step B.2 — Build the treatment bundle.**

Current source IS the treatment (hold-feasibility default-on). Bundle
without rewriting:

```
python scripts/bundle_agent.py agents/baseline --out-dir /tmp --force --skip-parity-gate
mv /tmp/baseline.py /tmp/baseline_with_holdfeas.py
```

**Step B.3 — A/B them.**

```
python fast.py eval /tmp/baseline_with_holdfeas.py \
    --vs /tmp/baseline_no_holdfeas.py \
    --max-seeds 64 --workers 6
```

Note: NOT `--vs-panel` (that needs the panel infrastructure). Just
direct h2h. Gate at Wlo ≥ 0.55. If treatment wins:

**Step B.4 — Submit the treatment** (PI approval required per Rule 1):

```
python scripts/bundle_agent.py agents/baseline --out-dir submissions/ --force
kaggle competitions submit -c orbit-wars -f submissions/baseline.py \
    -m "hold-feasibility filter validated solo (Wlo=<X>)"
```

This is a slot-spend on a single mechanism. If it lifts to μ ≥ 1120,
we've established the filter is a real lift independent of other
changes from the late-PM 2026-05-18 session.

### Phase C — Harvest oracles for trajectory (stretch, 30-60 min)

Under default trajectory chooser, run the 14 oracles. Map:

| Oracle | Likely trajectory behaviour | Action if it fails |
|---|---|---|
| `test_oracle_cleanup_capture_last_opp_planet` | May pass (trajectory rollout sees endgame) | If fails: trajectory needs endgame finish bonus too |
| `test_oracle_coordinated_capture_two_sources` | Likely fails (was xfail under trajectory; ROI converted via downsize-residue-5) | Backport ship-count enumeration to chooser_trajectory |
| `test_oracle_solo_capture_but_loses_source` | Likely fails (multi-step planning required) | Park; needs joint defense in chooser_trajectory |
| `test_oracle_no_launch_past_horizon` | Should pass (proposer's MAX_HORIZON gate) | n/a |
| `test_oracle_roi_picks_higher_production` | Should pass (any chooser ranking by prod×PV) | n/a |
| `test_oracle_n_way_coalition_three_sources` | Likely fails (trajectory only does pair joints) | Future: generalise `score_candidate_v4_joint` |
| `test_oracle_opp_modifier_drops_exposed_launch` | n/a (ROI-specific) | Skip under trajectory |
| `test_oracle_hold_feasibility_*` (3) | Should pass (proposer filter) | n/a |
| `test_drain_frontier_*` (2) | Should pass (proposer filter) | n/a |
| `test_oracle_defense_*` (2) | Should pass (proposer reinforce) | n/a |
| `test_oracle_sanity_trivial_capture` | Should pass (any chooser) | n/a |

For any oracle that fails on trajectory AND maps to a real game pattern,
file as a follow-up issue. Don't backport this session — keep Phase B's
A/B + push as the primary work.

### Phase D — ROI fate decision (15 min, defer)

Three options for the chooser_roi.py code, parking, deletion, or
revival:

1. **Park as opt-in** (recommended). Keep the file. Future sessions can
   pick up the architecture if a different opp model becomes available
   (e.g., a learned policy or a top-tier-mirror that actually mirrors
   ladder opponents). The 14 oracle scenarios remain.
2. **Delete chooser_roi.py + the new oracle additions.** Removes ~700 LOC.
   The proposer filter wins from this session's exploration would still
   stand.
3. **Revive on top of trajectory.** Keep ROI's hard-constraint oracles +
   defensive coalition + N-way coalition enumeration. Use trajectory's
   score_candidate_v4 for the actual scoring. This is the "best of both
   worlds" that the consolidation analysis pointed at — but it's
   another 2-3 sessions.

Recommend option 1 (park). It's free.

## Submission cadence reminder

Per Rule 12 (Orbit Wars caveat), the rolling pair is the LAST 2
submissions. Current state:
- `52784853` (PV off + clean math) — μ snapshot 1121.2 today
- `52766596` (joint v3) — μ snapshot 1118.8 today

A 3rd push would evict 52766596 (the older). Protected floor would
then be min(1121.2, new_μ). Phase B's push, IF it wins the local
A/B, becomes the new rolling-pair entry; 52766596 evicted. If new μ
≤ 1121.2, we've lost the 1118.8 floor — but we already had a worse
floor than the prior 1143.7 anyway (trajectory champion is already
evicted). So the risk is bounded: any push that lifts above 1118.8
is a net gain.

## Critical files

- `agents/baseline/proposer.py:407` — `_target_holdable_after_capture`
  (the hold-feasibility filter; primary target of Phase B).
- `agents/baseline/proposer.py:627` — the filter's wiring (env-var gate).
- `agents/baseline/main.py:38` — `BASELINE_CHOOSER` default.
- `submissions/baseline.py` — current trajectory bundle on the ladder.
- `state/current.md` — rolling-pair record (corrected this session).
- `tests/test_planner_oracles.py` — 14 oracles, useful for Phase C.
- `agents/baseline/chooser_roi.py` — research code, park.
- `audit/2026-05-19-postmortem-roi-pivot-failure.md` — TODO write this
  postmortem at session-end per Rule 36.

## How to start next session

1. **Read this file first.** Then `state/current.md` and `HANDOVER.md`.
2. Refresh ladder snapshot: `kaggle competitions submissions orbit-wars`.
   Note current μ for 52784853 and 52766596 (drift expected per
   `kaggle-mu-does-not-settle` friction).
3. Verify default chooser is still trajectory (sanity check the panic
   read).
4. **Start with Phase B Step B.1**. Phase A is housekeeping (~5 min).
5. PI approval required before submitting. Don't push without confirming.

## Rule reminders for next session

- **Rule 1**: submissions are single-shot, PI-approved.
- **Rule 12** (caveat): rolling pair is rolling LAST 2; the literal
  rule overrides any state-file claim.
- **Rule 27 caveat for Orbit Wars**: pre-submit h2h ≥10 games vs
  current rolling agent. Phase B already does this (64 seeds).
- **Rule 32**: session-start git fetch + ladder query mandatory.
- **Rule 38**: fix-verification reproduces failure state. For Phase B,
  the failure-state is "without hold-feasibility, candidate captures
  unholdable targets and bleeds ships." Could write an integration
  test that records a 10-game series under both bundles and counts
  bled-fleet rate; promote to permanent if signal is positive.
- **Rule 40**: prefer modeling-correctness over restriction-tuning.
  The hold-feasibility filter IS a restriction (a candidate pre-cut).
  If Phase B's A/B confirms it's the right call, the modeling-correct
  fix is for trajectory's rollout to see opp's recapture within
  horizon — but trajectory's K=25 rollout limits this. The filter is
  a pragmatic band-aid; ladder validation is the real test.
- **Rule 39**: no Claude session URLs in commits / PR bodies.
- **Rule 36**: session-end second-brain update + postmortem at
  `audit/YYYY-MM-DD-postmortem-roi-pivot-failure.md`.
