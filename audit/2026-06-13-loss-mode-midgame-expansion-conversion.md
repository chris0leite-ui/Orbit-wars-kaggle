# 2026-06-13 — Loss mode: mid-game expansion-conversion failure (step 50–100)

**Source:** real ladder episodes of sub 53595717 (our live agent;
`scripts/diag_live_loss_phase.py`). 2P: 30 W / 20 L. 4P: 30 W / 39 L.
Self-matches excluded. Corroborated by the un-blinded local 4P panel
(40.6% first-place vs a strong-peer pool, 71.9% vs weak lineage).

## The finding

We do **not** lose in the opening, and **not** in the endgame. We lose in
the **step 50–100 expansion window**. Focal planet count (absolute):

| step | 2P WIN | 2P LOSS | 4P WIN | 4P LOSS |
|---|---|---|---|---|
| 25 | 4.2 | 3.5 | 2.7 | **3.0** (we start ahead!) |
| 50 | 9.2 | 6.5 | 4.6 | 3.7 |
| 100 | 12.5 | **6.5 (stalled)** | 9.3 | **3.5 (stalled)** |
| 150 | 14.2 | 4.8 (declining) | 9.7 | 2.6 (declining) |

At step 25 the win/loss games are even (4P losses even start with *more*
planets). Then in losses our planet count **stalls** at ~6.5 (2P) / ~3.5
(4P) through step 50→100 while winners roughly double. By step 100 the
game is effectively decided.

## It is NOT hoarding and NOT under-launching — it is non-conversion

Garrison vs in-flight in the step 50→100 stall window of LOSS games:

| | 2P LOSS 50→100 | 4P LOSS 50→100 |
|---|---|---|
| planets | 6.5 → 6.5 (flat) | 3.7 → 3.5 (flat) |
| garrison ships | 175 → 156 (eroding) | 130 → 155 (flat) |
| **in-flight ships** | **206 → 197** | **123 → 112** |

We have ~120 (4P) / ~200 (2P) ships **in flight** the whole window — we
ARE launching — but planet count does not grow and garrison erodes. The
fleets are not converting into captured-and-held planets. Winners over the
same window launch MORE (in-flight 124→249 in 4P) AND convert (planets
4.6→9.3). So:

**Loss mode = our mid-game launches fail to convert to net planet gains
against a strong opponent, while the strong opponent's launches do.**

Plausible mechanisms (not yet isolated — next diagnostic):
1. We target contested/defended planets and lose the combats (no net gain).
2. Capture-and-immediate-loss churn (take a planet, lose it back within turns).
3. Launches diverted to reinforcement/defense rather than neutral expansion.
4. Reach: after nearby neutrals are taken (~step 50) the planner stops
   reaching for farther neutrals the winners still grab (echoes the
   historic "static K=10 hid 75% of the expansion map" finding).

Note: this is distinct from the shot-validator's "low-P attack" framing.
Removing doomed attacks (reject-only) was a null because the problem is
not that we launch bad attacks — it is that our launches don't CONVERT.
The fix must improve conversion (targeting / sizing / reach), not just
veto.

## Why this matters now

- It is the dominant loss mode on the real ladder, in BOTH formats, and it
  is exactly where we trail a strong ProducerLite peer (the un-blinded
  panel's 40.6% pool).
- It is now testable locally: the heterogeneous 4P panel reproduces it
  (`producer,v7_0,nearest`), so a candidate fix can be A/B'd before a live
  slot (the referee-blindness fix's payoff).

## Proposed next step (PI to steer — observation-driven loop)

Diagnose mechanism #1–4 above by tracing ONE step-50–100 loss game (what
are the ~120 in-flight ships targeting, and what happens to them?), then
propose the smallest existing-knob fix that lifts step-50–100 conversion.
Candidate knobs already in the stack: neutral-expansion reach/horizon,
capture-floor sizing vs contested neutrals, regroup/convoy thresholds.
No submission until a fix clears the un-blinded 4P panel.
