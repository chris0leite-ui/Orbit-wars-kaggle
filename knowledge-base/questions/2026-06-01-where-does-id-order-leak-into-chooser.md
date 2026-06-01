# Open question — 2026-06-01

**Where does absolute planet-id ordering leak into the chooser stack?**

After 180° canonical-frame rotation of P1's obs, P0 and P1 STILL emit
different decisions on identical canonical input. Rotation preserves
geometry but not ids — so any code path iterating by sorted id is
rotation-invariant in the wrong direction (it remains seat-asymmetric).

Three candidate sites:

1. `score_candidate_v4` — does the leaf-Δ comparison aggregate per-
   planet contributions in id-sorted order? If yes, FP roundoff differs
   between P0 (base+0, base+1, …) and P1 (base+3, base+4, …).
2. Opp model — `for pid in sorted(opp_planets)` over absolute ids.
3. Joint chooser — pair generation `(src_id, tgt_id)` tie-break.

The diagnostic: instrument the chooser to dump per-candidate scores
for both seats at the first divergence turn (canonical seed 0 turn
~31). The earliest scoring delta names the subsystem.

If id-order leak is the cause, the fix is to remap ids to a
canonical "my=0..N, opp=N+1..M" namespace at the chooser boundary
— ids become positions in a list, not absolute integers.

If something else is the cause (geometric FP ordering, set-iteration
hash order, …), the diagnostic surfaces it and we pivot.
