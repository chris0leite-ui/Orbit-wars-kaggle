# 2026-06-01 — Opening lookahead (PI direction) + expansion/hoarding diagnosis

## PI voice — next steps (Rule 35, append-only)
> "Next steps will be to work on the opening. Possibly a dynamic lookahead
> will give us a better opening. In the beginning there is not as much going
> on as mid or end game so we might afford farther lookahead."

Interpretation: the opening has fewer entities (planets still neutral, few/no
fleets in flight, no combat stacks) → per-turn compute is cheap → we can
afford a **deeper/farther lookahead early** than we use mid/late game. A
**depth-adaptive (dynamic) lookahead** that spends the saved early-game
wallclock on deeper opening search — plausibly gated by entity count or
measured per-turn headroom.

## Why this is the right axis to pick up next
- The snowball is decided in the **opening**. In every watched loss the gap
  opened by ~step 49 (vs istinetz 2P: we sat on 195 ships / 3 planets while
  istinetz spread to 9). A deeper opening search can plan a multi-capture
  expansion *sequence* the myopic per-candidate chooser can't see.
- It is a **fresh axis**, distinct from the heavily-falsified chooser /
  value-function axis (closed tracks: analytical-chooser 10 slices 0 lift,
  reach-frontier prescriptions, value-head aggregators, chain-bonus). Rule 37
  does not bind here.
- Existing assets to build on / replace: `lib/lookahead.py`,
  `lib/lookahead_planner.py`, `lib/joint_solver/opening_planner.py`
  (`opening_plan`, `OPENING_MILP_ENABLED`, `OPENING_HORIZON` — main.py already
  routes `step < OPENING_HORIZON` through a MILP opening planner and falls
  through to the chooser; PI's dynamic-lookahead likely extends/supersedes
  this). Read `agents/baseline/main.py:896-911` first.

## Session synthesis (what today established)
1. **Dominant live loss mode = ship-hoarding / under-expansion.** Replay-mined
   48 live games of sub 53248277: 59% of our fleets just reinforce planets we
   already own, only 25% capture. All 3 watched losses (istinetz ×2, xdddd):
   we had ≥ as many ships but FEWER planets — opponents spread, snowballed,
   won. `audit/2026-06-01-live-replay-diagnosis.md`.
2. **Root cause (chooser).** `score_candidate_v4` scores a launch as
   rollout-delta vs a do-nothing baseline in which ME goes passive while the
   opponent attacks every tick → fresh captures net ≤0 (step-39 istinetz trace:
   a +916-production neutral scored −0.99; agent emitted 0 launches on 115 idle
   ships). NEUTRAL/FOLLOWON bonuses are gated on `delta>0` so structurally
   cannot fix it. v4 had DROPPED the v2 static scorer's held-production term.
3. **Fix built** (default-OFF, commit `9a19221`): `BASELINE_EXPAND_CREDIT`
   restores the held-production capture credit, un-gated. Flips hoarding→
   spending at the failure state (step 39: 0→94 ships). Latency +10ms.
4. **But it is a non-gain where measured**: 40% vs champion mirror (n=40);
   live μ on sub 53259633 settling ~1086 (below the 1138 backstop).
5. **Key methodological learning**: the vs-champion A/B is the WRONG
   instrument for an expansion fix — both agents are hoarders, so expansion
   cannot differentiate them. Expansion helps vs **aggressive expanders**
   (the live istinetz/xdddd field), not vs hoarder mirrors. A **flat** credit
   is the wrong shape; the rollout's restraint is *correct* vs strong
   symmetric play and *fatal* vs aggressive play → the real lever is
   **opponent-aware / state-aware expansion** (lean aggressive only when the
   opponent is out-expanding us, or early when neutrals are free).
6. **Kinematic table closed**: KT-OFF vs KT-ON = 50.0% (n=36) + 66/66 move
   parity → behaviorally neutral AND not corrupting in-process measurement
   (refutes the singleton-leak hypothesis at the outcome level). The KT-off
   rebuild is the clean champion base going forward.

## Open question for next session
Does a depth-adaptive opening lookahead (deeper early / shallower late, gated
by entity count or wallclock headroom) measurably improve the opening —
evaluated vs AGGRESSIVE opponents, NOT the champion mirror? And does
opening-quality alone close the snowball gap, or must it be paired with the
opponent-aware expansion credit?
