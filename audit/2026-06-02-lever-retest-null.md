# Three-lever re-test on the table-ON / state-driven-K champion — NULL (2026-06-02)

Hypothesis (carried from the morning handover): team-up, opening-planner, and
position-score may have failed earlier only because they were judged table-OFF or
on the singleton-corrupted in-process harness. Re-test each on the current champion
(`submissions/baseline_state_driven_k.py`) via CRN-isolated `clean_ab`.

## Results (vs champion)
| Lever | Flag | n | result | Wilson-lo | verdict |
|---|---|---|---|---|---|
| Team-up | `BASELINE_JOINT_SYNC=1` | 16 | 9/16 (56.2%) | 0.33 | parity |
| Opening planner | `BASELINE_OPENING_MILP=1` | 16 | 7/16 (43.8%) | 0.23 | negative |
| Position-score | `BASELINE_VALUE_HEAD=composite` | 16 | 9/16 (56.2%) | 0.33 | parity |
| **Stack (team-up + poscore)** | both | 16 | **8/16 (50.0%)** | **0.280** | **NULL** |

Stack cost smoke PASS (p95=627ms, max=843ms, zero >1000ms).

## Key evidence — the stack is structurally identical to the champion
All 8 seed-pairs split exactly W/L between the two seats (CRN seat-swap): the seat /
geometry decided every game, not the agent. The two ~56% parity levers produced
*exactly* 50% combined — no interaction lift.

## Conclusion
The table-confound hypothesis is **falsified**. None of the three levers beats the
table-ON champion; they are genuinely at parity (or negative). Nothing clears the
submit gate (Rule 43b: Wilson-lo ≥ 0.50 vs rolling champion). Rolling pair untouched.

## Garrison axis (PI "might it work now" question) — see 2026-06-02-early-capture-gap-gate.md
Gate POSITIVE-but-non-discriminating: real ~7-10 turn opening delay on the first
affordable cheap-neutral launch (chooser prefers wait-band over fire-now), but the
delay rate is identical in wins (2.32/step) and losses (2.13/step) → not the loss
driver (Rule 41), and the fix lives in the closed value-head axis (Rule 44). Not a
reflexive build. PI rejected the aggressive "tech-and-kill" re-framing of it.
