# 2026-06-03 — Joint-coordination closed, ME-defends negative, the throughput wall

Branch: `claude/champion-strategy-rules-00JzI`. Session arc: finished the
joint-coordination thrust, then took the value-leaf axis to a first negative.

## What happened, in order

1. **Joint-coordination planner — closed (Rule 37).** Two ways:
   - *Greedy-as-replacement* underperforms the champion (9/16 vs 16/16 vs v7_0).
     Its conditional marginal gains are passive-self-pessimistic, so it
     under-commits good independent launches. Raising the build horizon did
     nothing (9/16 → 9/16); flat capture-credit (EXPAND_CREDIT=1.0 mirrored
     into the joint scorer) made it WORSE (6/16) and blew the turn budget
     (max 1983ms). The coordination *waste* seam is small.
   - *Augment-not-replace refiner* (champion + exact-oracle teamwork-add,
     `chooser_refine.py`, default OFF, built + tested + bench-PASS) is
     COMPLETELY INERT: `generate_sync_coalitions` yields ZERO raw candidates
     in real games vs both v7_0 and v7_minimax. The 2-source "neither alone but
     both together" structure essentially never arises — sources accumulate
     enough ships to solo-take their targets (solo-skip gate, chooser_trajectory.py:1124).
   - **Conclusion:** the coordination seam (waste + teamwork) is empirically
     SMALL; the champion's independent solo-delta scoring + locks is near-optimal
     vs available opponents. Oracle + refiner + `out_chosen` kept default-OFF as
     latent capability.

2. **Value-leaf axis (PI pick) — ME-defends, first negative.** Target: the
   audited hoarding (out-ship but under-capture; "fix the valuation"). The leaf
   is read after a ~13-tick rollout with ME idle while opponents react, so
   defensible expansion looks fragile. ME-defends (future-me plays purely
   defensive reactions in the candidate rollout only; already built, default OFF)
   is the corrected ME-reacts. **Mechanism verified** (test: it raises the
   expansion candidate's score by the held-territory value). **Timing neutral**
   (p95 900 vs champion 878). **But A/B = 5/16 (31%) vs the champion mirror** —
   not a lift; joins ~8 falsified value-leaf mechanisms.

## The cross-cutting lesson (recurring this session and prior)

The champion is **strong and near-locally-optimal** against every opponent we
can run locally. v7_0 → 16/16 (saturated); v7_minimax → too slow to batch;
champion-vs-champion → the only "close" matchup, but heavy (~2 min/game). So
local A/Bs structurally CAN'T show a lift: either we're saturated (no room) or
the matchup is too heavy to get n. Both the refiner and ME-defends died on this
— not because the idea was obviously wrong, but because the testbed can't
distinguish "neutral" from "small lift" at feasible n. The real signal lives on
the **live ladder** (diverse ~1100–1200 field), which we can only access by
submitting (PI-gated, rolling-last-2 discipline).

Implication: before mining more single-mechanism value-leaf tweaks, the
higher-EV move is probably to fix the **A/B throughput** (decided-lead early-call
+ a lighter non-saturated triage opponent) OR accept submit-and-measure as the
validation path. Pure local A/B at the champion's level is a near-dead measuring
instrument.

## Durable artifacts (all default-OFF, champion byte-identical)
- `chooser_refine.py` + `choose_trajectory(out_chosen=...)` + exact oracle.
- `tests/test_chooser_me_defends.py` (Rule-38 repro, 3 green); `tests/test_chooser_refine.py` (3 green).
- `submissions/baseline_champ_defends.py` (A/B fork: champion + DEFENDS hardcoded).
