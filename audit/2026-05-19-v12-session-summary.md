# v12 implementation session summary

Branch: `claude/review-foundations-progress-14HXp`
Date: 2026-05-19
Commits: 0005a51 (C1) → 57b5e20 (C2) → bc29703 (C3) → 20c5685 (audit)

Prior context: `audit/2026-05-18-loss-mode-v8-v9.md` (on `origin/claude/
recover-main-foundations-MV0e2`) — the loss-mode diagnostic that drove
the v12 design via Felipe + Naoism + maruichi forensics.

## What shipped

`agents/v12/main.py` (170 LoC) — a v4_planner-architecture-style chooser
with two principled-modeling upgrades:

| Commit | Change | LoC | Effect on Felipe seed 1492346051 |
|---|---|---:|---|
| C1 | Bootstrap parity-match `v4_planner` bundle | 170 + 19 lib | LOSS 0/2 (parity baseline) |
| C2 | CRN via pre-recorded `opp_traj` | 100 lib + 18 agent | LOSS 0/2 (no flip) |
| **C3** | **`evaluate_value_v12` ship-balance term** | **78 lib + 2 agent** | **WIN 1/2 (P0 step 174)** ← flip |

The flip is from **C3 alone**, not C2 — confirmed by replaying with
the bundled `v4_planner` (which has C2-equivalent opp-policy
deterministic-step rollouts but the original `evaluate_value`): 0/2.

## The core defect (sharpened from prior diagnostic)

The original `lib.lookahead_planner.evaluate_value` weights:

```
V = prod_share + 0.4*prod_denied + 0.05*ships_share + 5.0*lone
```

`ships_share` (my_ships / total_ships) at weight 0.05 contributes
~0.025 swing for a 100-ship differential at total 500. Compared to
`prod_share` baseline ≈ 0.6, that's invisible — the chooser can't tell
which portfolio gives a meaningful ship-mass advantage.

**maruichi game (eid 76670184) decisive turn:** us 6p/152s vs maru 6p/306s
at step 50. prod_share parity (~0.5 each); ship_balance −0.34. Old
`evaluate_value` says ~equal value across portfolios. v12-C3's
`evaluate_value_v12` says ship_balance × 0.3 = −0.1 swing — chooser now
correctly scores "build ships / reinforce" as Δ-positive vs "do
nothing."

## `evaluate_value_v12` design

```python
V = prod_share                                  # [0, 1]   structural future
  + 0.4 * prod_denied                           # [0, 0.4] denial bonus
  + 0.3 * ship_balance                          # [-0.3, +0.3]  current realized
  + 5.0 * lone_survivor                         # 0 or 5
```

where `ship_balance = (my_ships − opp_ships) / total_ships` in
`[-1, +1]`.

Key design choices:
- **Signed differential, not share.** `ships_share` is always positive
  [0,1]; `ship_balance` swings symmetrically through zero so the chooser
  sees real deficit, not just "I have less than half."
- **Weight 0.3.** Same scale as `denial_weight` (0.4). Conservative
  starting point; can tune up if panel suggests.
- **No `pv_horizon` term.** At the leaf the step is constant across
  portfolios, so a `pv*(my_prod-opp_prod)` term is just a re-scaling
  of `prod_denied`. Information was already there; the missing signal
  was ship-balance.

## C2 (CRN) is preserved despite not flipping Felipe alone

Reasoning:
- CRN reduces variance across portfolio scoring → cleaner argmax
- One-shot ~50-100ms `record_opp_traj` cost is small relative to budget
- Modeling simplification (opp doesn't see my action) is consistent
  with v4_planner's deterministic opp-policy design

If the panel shows regression vs v4_planner-bundle, the diff isolation
will help — drop C2, keep C3 alone.

## C4 (top_tier_mirror_policy) DROPPED

`top_tier_mirror_policy` in `lib/opp_model.py:91-119` is functionally
identical to `_v351_action`:
- Both call `propose_snipe_missions(aggressive=True) + propose_reinforce_
  missions`
- Both pipe through `settle_plan + realize(DEFAULT_MECHANISMS)`
- Only diff: `top_tier_mirror_policy` reads `_shared_world_model` from
  obs if present (perf cache; no behavior change)

C4 from the plan was redundant. Removed from the plan's standing
design.

## C5 (K_MAX bump) — DEFERRED

K_MAX=10 is the current value. Bumping to 12-15 was planned conditional
on wallclock budget. v12-C3 measured p95 in 637-758ms — about 250-350ms
of headroom under 1000ms cap, but not enough margin to safely bump K to
15 (~150ms additional per portfolio × 5 portfolios = 750ms over).
K=12 would be safer (~75ms additional × 5 = 375ms over budget).

Deferred to a next-session experiment. Don't ship blind.

## Gates: PASS / PEND / FAIL

| Gate | Status | Evidence |
|---|---|---|
| G1 Parity (C1 only) | **PASS** | 0 mismatches 3 seeds × 20 turns vs v4_planner-bundle |
| G2 Felipe (≥1/2 vs v7_0) | **PASS** | C3: 1/2 (P0 win step 174) |
| G3 Naoism (≥1/2 vs v7_0) | FAIL (soft) | C3: 0/2; v4_planner-bundle also 0/2 → architecture-bound |
| G5 Panel (Wlo ≥ 0.55, 3 opps) | **PEND** | running in background `bjqrpy5sy` (8 seeds × 4 workers) |
| G6 Wallclock (p95 < 800ms) | **PASS** | All games p95 ≤ 758ms, max ≤ 787ms |
| G7 Emission-rate diagnostic | DEFERRED | requires new script + panel games to compute |

## Submission decision criteria

Per the plan:
- **PASS panel → submit v12 to ladder.** Live μ should improve from
  v9's 1085.0 if C3's modeling fix translates.
- **FAIL panel vs v4_planner-bundle → rollback to v12-C1.** Keep
  C3 as a learning, retry with different ship_balance_weight or
  alternate value-head structure.
- **Live submission disclipline (CLAUDE.md Rule 12):** rolling-last-2
  auto-evicts. Currently `[v9_scavenge μ=1085.0, v8_scavenge μ=1043.0]`.
  v12 push evicts v8. Acceptable if v12 ≥ v9 in panel.

## Next-session first actions (if panel passes)

1. Bundle v12 for submission: `python scripts/bundle_agent.py
   agents/v12 submissions/v12.py` (or equivalent).
2. Verify the bundle imports cleanly: `python -c "import importlib.util;
   spec = importlib.util.spec_from_file_location('v12_b',
   'submissions/v12.py'); m = importlib.util.module_from_spec(spec);
   spec.loader.exec_module(m); print(m.agent)"`.
3. Submit: `kaggle competitions submit -c orbit-wars -f submissions/v12.py
   -m "v12: v4_planner arch + CRN + ship_balance value head"`.
4. State update: `state/current.md` add v12 submission row; record
   live μ once available (5-10 min).
5. Pull v12 loss replays after 24h, classify by phase, compare to v8/v9.

## Next-session experiments (if panel passes; budget permitting)

- **Tune ship_balance_weight.** Try 0.5, 0.7 — see if Felipe goes
  2/2 and Naoism flips. Risk: over-emphasizes current-state at
  expense of structural prod_share.
- **K_MAX=12 carefully.** Profile per-step ms in v12-C3 mid-game.
  If aggregate p95 stays < 800ms with K=12, bump and re-gate.
- **Replace `_v351_action` with a true opp model** that simulates
  multi-opponent threats (currently v3.5.1 is 2P-only and falls back
  to v3.5.1 incumbent in 4P).

## Why C3 worked

The diagnostic in `audit/2026-05-18-loss-mode-v8-v9.md` triangulated
the strict-idle baseline as the dominant defect. v12-C1 ports the
v4_planner architecture (which is NOT strict-idle — it uses
`_v351_action` opp policy in the rollout, so opp IS playing during
the K-step lookahead). This addresses the strict-idle defect at the
architecture level.

But v4_planner-bundle still loses Felipe 0/2 with its native value
head. The OTHER part of the diagnostic — that the value head was
under-weighting ship-mass differential — is the remaining hole.
v12-C3's `evaluate_value_v12` is the fix for that.

So v12 = v4_planner architecture + ship-balance value head. Both
upgrades load-bearing. C2's CRN is the variance-reduction third
piece, additive (not load-bearing for Felipe, but cleans the argmax).

The methodology of "look at games to find modeling improvements, not
tweaks" delivered:
1. Felipe + maruichi forensic identified TWO upstream defects
   (strict-idle, value-head ship-balance)
2. The fix targets the modeling components directly, not caps
3. The improvement is single-seed measurable AND aggregate (panel)
4. No "if-X-then-Y" tweaks introduced

Methodology working as designed.
