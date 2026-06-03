# 2026-06-03 — The teamwork (sync-coalition) seam reopens: a weak-opponent confound, corrected

## What happened

The 2026-06-02 session closed the joint-coordination axis (Rule 37) on the
finding that `generate_sync_coalitions` "generates nothing — the teamwork
structure doesn't arise." This session falsified that closure and validated the
refiner as the active submission candidate.

## The confound (Rule 41)

The "generates 0 every turn" measurement was only ever taken against **v7_0 /
v7_minimax** — opponents the champion beats 16/16. Against a weak opponent the
champion is **source-saturated**: every planet accumulates enough ships to
solo-take its targets, so the generator's precondition (a defended target that
*neither* nearby source can solo-capture but *both combined* can) never fires.
That is a property of a stomp, not of the game. The conclusion generalized a
weak-opponent artifact to "coordination is empirically small."

## The correction

Measured the generator **directly on the real boards** of champion-vs-strong-
opponent games (couldn't use the agent's own stderr — `kaggle_environments`
sandboxes per-agent output; walked `env.steps` post-hoc and called the
generator on each board). Result: vs a champion-strength opponent the generator
yields **~100+ two-source coalitions/game** (midgame 107 / 780 board-steps over
6 games, up to 8/turn). The driver is the **resource ratio** — coalitions
appear when my planets are contested / out-resourced (my sources ≈ or <
defended targets), i.e. exactly the close games that decide the ladder. Vs
v7_0 the count is still 0 (the old null reproduces).

## Exploiting it wins

Augment-not-replace refiner (`BASELINE_CHOOSER=refine`: run the champion
verbatim, then add only oracle-positive two-source coalition atoms that don't
conflict with the champion's locks) A/B'd on the **real champion config**
(adaptive-K + kinematic-table ON), both choosers on the same 16 seeds via
process-isolated `clean_ab.py`:

- Parity (trajectory vs champion): **16/32 = 50.0%** — clean anchor.
- Refine vs champion: **25/32 = 78.1%**, Wilson [0.612, 0.890].
- Paired: **+13 gained / −4 broke / net +9** on 32 matched seed-seats.

## Process lessons (two wrong reads I had to correct, both PI-caught or self-caught)

1. **First refine A/B ran on the wrong base.** I set a fixed horizon-10 config
   (adaptive-K OFF) on both sides — a non-champion setup. The headline number
   (68.8% vs a 62.5% noisy parity) was on stripped-down agents. PI caught it
   ("you considered the adaptive-K version?"). Re-basing on adaptive-K gave the
   clean 78%-vs-50% above. **Lesson: A/B on the LIVE champion config, not the
   repo default — confirm the parity anchor is ~50% before trusting a lift.**
2. **I claimed "not using the kinematic table" — wrong.** The table defaults ON
   (`main.py:896`, `get(..., "1")`); not setting the flag leaves it ON. Both
   sides had it the whole time, so the 78% is a fair table-ON result. **Lesson:
   "I didn't set the flag" ≠ "the flag is off" — check the default.**
3. **Mid-run win-rate trends are noisy.** I called the static run a "wash" at
   58% mid-run; it finished 68.8% (a 9-0 back-half streak). Withhold verdicts
   until n is complete (Rule 45).

## Open questions for next session

- **Regression tail:** refine BROKE 4 of 32 games the champion won (long
  contested seeds). The generator is fine (Step 2); the misfire is on the
  **scoring/selection** side — which coalition the oracle picks in long games.
  Diagnose via the saved replays (`/tmp/refine_adaptivek_replays`,
  `/tmp/refine_panel_replays`).
- **compute-by-ships × refine** — see
  `knowledge-base/concepts/refine-x-computebyships-compatibility.md`. PI wants
  to know if the two compose ("best of both worlds"). Short answer: they target
  different layers (per-source solo search breadth/depth vs teamwork addition)
  and likely compose, but (a) combined wallclock must be re-benched and (b)
  compute-by-ships raises solo reach which may partly cannibalize coalition
  opportunities — a measurable interaction.
