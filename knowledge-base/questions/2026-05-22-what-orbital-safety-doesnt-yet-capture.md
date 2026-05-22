# 2026-05-22 — what does orbital safety NOT yet capture?

The orbital safety fix landed at 2/4 vs the strong consolidated
baseline. One of the two losses was on seed 0 — game ran the full 500
steps before orbitfix lost. The other was seed 2, 292 steps.

**Open question:** what does the fix not yet model that the strong
baseline gets right on those seeds?

Hypotheses to test (in EV-per-investigation-hour order):

1. **Source-side rotation.** The fix predicts target+opp positions at
   our arrival but `_source_survives_launch` and `capture_size` (the
   reinforce path) still use current positions. An orbiting source
   that rotates INTO enemy reach during the wait/launch window would
   be silently scored as safe-to-drain. Easy reproducer: instrument the
   filter to log when an orbiting source's predicted position differs
   materially from current.

2. **`t_op` straight-line approximation in the proposer filters.** B1
   and B2 use predicted positions for the distance/eta to the nearest
   opp, then a plain `dist/v` for the actual recapture-time estimate.
   The B7-style fixed-point we added in `time_to_enemy_threat` was
   NOT extended to those filters. Easy port if it matters.

3. **Game-end (500-step) timeout semantics.** Kaggle resolves a
   500-step game by ship-total; we lost seed 0 at the cap. What does
   the strong baseline do in the late game that orbitfix doesn't?
   Possibly the recently-deleted/disabled snipe followon (B4 fixed
   the position but the sniper config is OFF by default).

4. **Inbound staggered waves.** B6 surfaced the "later wave hidden by
   earlier-pre-arrival" case, but the WAVE_LOOKAHEAD window in
   `capture_size` still operates on a flat `enemy_eta` returned by
   `incoming_enemy_eta` (not `_after`). If that surfaces a different
   shortfall calculation, it's a follow-up modeling fix.

Sequenced approach: trace seed 0 turn-by-turn diff (orbitfix decision
log vs consolidated decision log); identify the first diverging move
and back-chain to which subsystem made the wrong call. That tells us
which hypothesis above applies.
