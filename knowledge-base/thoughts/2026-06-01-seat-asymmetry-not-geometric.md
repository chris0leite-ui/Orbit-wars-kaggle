# 2026-06-01 — seat-asymmetry survives geometric rotation

Self-play with our deterministic bundle on both sides, seed=0: perfect
mirror symmetry for ~30 turns, then divergence to P0=30 / P1=1 at turn
500. Hypothesis was "geometric seat-asymmetry, rotate P1's obs 180°
through board center and the asymmetry disappears."

Built the rotation infra (commit `edd63b3`): `_rotate_obs` involution +
`_unrotate_actions` on all four return sites. Verified mathematically
(4 unit tests, bundle parity 15/15). Still: self-play across seeds
[0,1,2,3,4] → P0=5, P1=0. Rotation alone is insufficient.

The instrumented seed-0 trace shows P1 makes a DIFFERENT canonical-
frame decision than P0 — P1 waits one extra turn before first launch,
then fires 22 ships vs P0's 21. Same canonical input, different output.
The seat-asymmetric source is INSIDE the chooser, not in the obs
geometry. Candidates not yet investigated:

- Opp-model iteration: if `for pid in sorted(my_planets):` runs with
  absolute ids, P0's smallest-id home (base+0) and P1's home (base+3)
  see different iteration orders. Rotation doesn't change ids — it
  changes positions. The id-sort survives.
- Joint-chooser pair generation: pair-id tie-breaking by
  `(src_id, tgt_id)` lexicographic order. Again — rotation-invariant.
- Candidate dedup: if dedup is "keep first by emission order," P0 and
  P1 hit different orders even with identical scoring.

The next session's first diagnostic is "instrument `score_candidate_v4`
to dump per-candidate scores for both seats at the first divergence
turn." The earliest scoring delta isolates the subsystem.

Strategic implication: any further "fix the seat asymmetry" attempts
should NOT rebuild the obs / proposer surface. The bug is in the
scoring stack, and the scoring stack is where rotation-invariant
id-order survives.
