# HANDOVER.md — next-session brief

_Refreshed 2026-06-02. Read this first (Rule 15). Also read `state/MULTI_BRANCH.md`
(live rolling pair / track registry) per Rule 44._

## This session's outcome (the unlock)

The **kinematics table** (per-turn planet-position cache → lets the time-adaptive
search score more candidates inside the 1 s/turn budget → better moves) was proven
to be a real ~20 μ lever and **re-enabled, de-singletonized** (`world._kt`,
per-turn/per-seat — no more shared-state A/B corruption). It had been removed
2026-05-30; the live peak μ=1183.7 was from the brief window it was active. It is
bit-identical to the inline path (`scripts/kt_position_parity_check.py`: 0/32
mismatches) and prevents real timeouts.

Also this session:
- **Submitted** `baseline_state_driven_k` (state-driven horizon-K **+** table). Live
  rolling pair now = `baseline_state_driven_k` + `baseline_launch_rules_universal`
  (table-ON champion). Both warming up. **Adaptive-K (μ≈1170) was evicted** — if
  state-K under-converges, re-submit adaptive-K (`champ_adaptiveK_on.py`); 5
  submits/day, deadline 2026-06-23.
- **Timeout fixed:** predictive per-step deadline bail (`1e3234c`). Bundle bench
  (separate-process = live): **max 944 ms, 0 over cap**. In-process `fast.py eval`
  can show false one-off highs from GC/cold-import — judge timing with `fast.py
  bench`, not eval.
- **Loss diagnosis (selection-bias-free, vs weaker opponents):** we lose by **losing
  the step-50→100 planet-expansion race** in certain maps (seed 2 loses from both
  seats) — NOT conversion/waste, NOT timeouts. Fleet-outcome mix is identical in
  wins and losses; only the planet-lead trajectory differs (wins +9 by step 100,
  losses −5). Tooling: `scripts/analyze_local_losses.py`.

## NEXT-SESSION PLAN — re-test 3 shelved features WITH THE TABLE ON

**Why:** these were judged "null/parity" under a handicapped agent — either tested
table-OFF (less search) or via the old singleton-corrupted in-process A/B. Both
confounds are gone. Two of them (#1, #2) directly target the expansion-race loss
mode we diagnosed. **All three are already in the code behind env flags —
flag-flip, not rebuild** (verified 2026-06-02).

| Feature (plain language) | Enable flag | Prior (confounded) result |
|---|---|---|
| **1. Team-up attacks** — 2+ fleets from different planets arrive at one target on the same turn, sized to take *and hold* it | `BASELINE_JOINT_SYNC=1` (+ size-to-hold, `chooser_trajectory.py:325`) | "size-to-hold NULL", sync probe μ≈1150 — table OFF |
| **2. Optimized opening** — solve the first ~50 turns as an optimization (which planets, what order) instead of hand-rules | `BASELINE_OPENING_MILP=1` (`lib/joint_solver/opening_planner.py`) | parity 4/8 (n=8) — table OFF |
| **3. Smart position score** — value a position by *where* things are on the map, not just counts | `BASELINE_VALUE_HEAD=composite` (`lib/value_heads.py`) | never measured (runs stalled) |

**Execution (one lever at a time — Rule 37):**
0. Baseline = current source, table ON + state-K (the table-ON champion). Re-confirm
   the live rolling pair before any submit (Rule 42).
1. For each feature, default-OFF → ON in isolation on top of the champion:
   (a) parity smoke (OFF byte-identical), (b) cost smoke `fast.py bench`
   separate-process — **max < 1000 ms WITH the table**, (c) `clean_ab.py` **n≥32**
   table-ON vs the table-ON champion, (d) compare to the prior table-OFF result —
   the delta is the table-confound size.
2. **Stack winners:** any lever clearing Wilson-lo ≥ 0.50 stays ON; re-baseline and
   add the next on top; re-A/B the stack (a solo win can regress when stacked).
3. **Order:** #1 team-up → #2 opening → #3 position score.
4. Each lever adds compute — re-check `fast.py bench` max < 1000 ms after each
   stack. If a strong lever blows budget, tune horizon, don't drop it (Rule 40).
5. Submit only on Rule 46 + 43a panel + 45 (n≥32) + 42 (eviction).

**Deferred — reassess WITH THE PI after the three above (do NOT start this session):**
- **2-hop redeploy** (shuffle forward so a follow-up can capture) — reverted
  (`5ec6a0d`); needs rebuild from spec `727e1bf`.
- **Reach-frontier chooser** (from-scratch value chooser) — separate agent
  (`agents/reach_frontier/`), had a hold=0 bug; it *replaces* the chooser, so
  evaluate whole-agent after the fix, don't stack.
- Cheap re-checks: H41 pv_horizon floor (`9ebd311`), PV_ETA tuning.

**Do NOT re-open** (dead for non-table reasons): H44 "fleets die in flight" (false —
sun/OOB only), 4p-cushion (4/32), b5 reward-axis (0/32), flat expansion-credit
(targeted the hoarding loss mode we refuted; real loss mode is the expansion race).

Full provenance (commit hashes per feature, the two confound mechanisms, deferred
detail) is in the session plan, mirrored into git history of this file.
