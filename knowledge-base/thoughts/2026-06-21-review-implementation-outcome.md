# 2026-06-21 — implementing the three-lens review (points 1–4): what landed, what didn't

> AI session note. Implemented the review's points 1–4, verified by replay vs Producer
> V2 (2P, seat 0). Honest outcome: the two correctness/safety changes are good; the two
> modeling changes did NOT beat the validated config and were reverted / kept OFF.

## What shipped (kept)

**Point 1 — correctness (no intended behavior change):**
- C1 fixed: replaced the `os.environ.setdefault` bake with `_SHIP_DEFAULTS` + `_cfg`
  (no process-global leak). Guarded by a new test that imports main and asserts no LR_*
  keys appear in os.environ.
- S3: pinned torch to 1 thread (determinism + ladder timing).
- S4: the outer pick `except` now emits an `LR_FALLBACK` stderr marker instead of
  silently shipping the weak greedy move.
- S2: corrected the false `_concentrate` docstring.
- T1: added tests/test_neutral_margin.py (38 tests green overall).

**Point 4 — 4P de-risk (kept):** `LR_NEUTRAL_MARGIN` is now 2P-only; 4P reads
`LR_NEUTRAL_MARGIN_4P` (default 0). 4P (≈60% of the ladder) was never validated for the
margin, so it's off there until a 4P A/B.

## What was reverted / kept OFF (the modeling changes did not pan out)

**Point 2 — contest-based neutral sizing: REVERTED.** The math review was right that the
shipped `margin*(ships + prod*eta)` uses the slow probe eta and over-sizes. But replacing
it with the "correct" contest-based surplus REGRESSED: 6013 and 6019 (both wins under the
validated value*eta margin) became LOSSES. Lesson: the real benefit was the SPEED of
bigger fleets on *all* neutrals (faster expansion), not just holdability against a
contest — the over-sizing is effectively absorbed into the 0.25 tuning. Kept the
validated value*eta form.

**Point 3 — lead-gated win-equity: IMPLEMENTED, default-OFF, REGRESSES → needs rework.**
`LR_LEAD_GATE` (worst-case `max` threat when ahead; offense boost when behind), measured
once per turn on the production gap. Replay results (2P vs V2, seat 0):

| seed | default (lead OFF) | lead-gate ON |
|---|---|---|
| 6013 | WIN | **loss** |
| 1127764379 | WIN | **loss** |
| 6031 | WIN | WIN |

It flips validated wins to losses. Root cause (matches the game expert's warning): a HARD
ahead/behind threshold FLAPS turn-to-turn, switching the whole threat model and producing
incoherent play (the trace shows stagnation then collapse). The reviewers prescribed a
SMOOTH sigmoid blend on the gap + hysteresis, not a binary switch — that is the rework.
Kept the code default-OFF; do NOT enable until the smooth/hysteresis version is built and
clears an n≥32 A/B.

## Net effect of this pass on the shipped agent

2P behavior is UNCHANGED from the validated NM=0.25 config (default 6013 still WIN); the
only behavioral change is 4P (margin now off there). Plus the correctness/safety fixes.
No submission this pass.

## Next (for PI)

The big lever (lead-aware win-equity) is still the right direction per all three
reviewers — but it must be the SMOOTH, hysteretic version, and a candidate-level (not
just per-turn) lead read may be needed. Decide with the PI whether to build that rework
or pursue a different mechanism after the live-ladder replay.
