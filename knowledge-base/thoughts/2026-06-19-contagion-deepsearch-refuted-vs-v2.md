# 2026-06-19 — the LR contagion / deep-search line is REFUTED vs Producer V2

## Verdict (well-powered, n=27 paired, stratified panel, 1v1, focal P0)
| config (same 27 seeds, `SEED_PANEL_128[::4]`) | wins | margin |
|---|---|---|
| **DEFAULT least_resistance** (2-ply take-and-hold, `LR_DEEP_OPP=0`, no deep search) | **18/27 (67%)** | **+275** |
| contagion deep-search d6 (`LR_DEEP_OPP=2`, depth 6) | 9/27 (33%) | −1068 |
| wide candidates + gentle calib (n=13) | 3/13 | −1042 |
| wide candidates alone (n=12) | 2/12 | −1628 |

**The agent we started the session with beats V2 (~67%, positive margin). Every
deep-search variant built this session — the contagion opponent (`LR_DEEP_OPP=2`),
depth-6 rollout, wide candidate generation, over-extension calibration — makes it
dramatically WORSE (down to ~33%, deeply negative margin).** A ~34-percentage-point
win-rate regression. The whole contagion/deep-search direction is refuted as a path
to beating V2.

This matches the documented validation of the default ("+7 vs V2 in 2P" at n=32,
`state/STRATEGY.md`). The default was always the stronger agent.

## How we got misled (the methodological lesson)
- Early reads compared contagion variants **against each other** on the hard 1v1
  slice (seeds 5000–5007), where everything loses — so the *relative* signal
  ("depth helps", "contagion ~parity with the mirror", "wide hurts then helps")
  looked meaningful while the *absolute* truth (all of it loses to V2, and far worse
  than the default) was invisible.
- **Lesson (load-bearing): always anchor an A/B to the current-best / validated
  baseline.** I should have put `LR_DEEP_OPP=0` (the default) in the panel from game
  one. Relative comparisons among regressions are worthless.
- Single-seed replays were great for **bug-finding** (they surfaced the sun bug and
  the wide-candidate fragmentation) but useless for **ranking** — the margins swing
  hundreds of points seed to seed; only n≥~20 paired separated the configs.
- Small-n overconfidence: I called "wide is dead" at paired-3 and "calib5 is best"
  at paired-10; both were noise. Don't rank at n<~16.

## Why the deep-search line loses
The contagion opponent grows rivals unbounded and the strong torch leaf rewards raw
planet/production count, so the rollout systematically misvalues positions vs V2 (it
over-commits, then can't hold). Depth past ~6 floods; wide candidates fragment; the
over-extension calibration that fixes fragmentation also over-prunes good plans. It's
a thin distributional/forward overlay on a value function that's already weaker than
the take-and-hold heuristic vs this opponent — the same shape as the
dropout-native refutation (`2026-06-18-dropout-native-phaseA-kill-gate.md`).

## What we keep
- **The sun-clearance fix** (`agents/least_resistance/main.py`, committed `4f9cf4e`):
  a real correctness bug in the BASE agent — re-aiming a capture at its true size
  could route the fleet through the sun because the sun filter only checked the
  hint-size trajectory. Independent of the deep-search line; keep it.
- All deep-search code (`LR_DEEP_OPP`, contagion, wide candidates, calibration) stays
  **default-OFF / gated** as a recorded negative result — the shipped agent (default)
  was never affected.

## Next direction — improve the DEFAULT agent vs V2
The default already wins ~67%; to CRUSH V2 the leverage is its ~1/3 of LOSSES. On the
panel it lost seeds like 32 (−1922), 78 (−1410), 647 (−836). Pull those replays,
diagnose what V2 does to beat the take-and-hold agent there, and fix THAT
(observation-driven) — not a deep search that regresses everything. Candidate
mechanisms live in `state/_archive/` and the producer_plus line; but first, watch the
losses.
