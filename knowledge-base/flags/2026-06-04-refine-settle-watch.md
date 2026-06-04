# FLAG — 2026-06-04: refine (sub 53336920) settle still unresolved; floor at risk

Carries forward `knowledge-base/flags/2026-06-03-rolling-pair-uncalibrated.md`.

As of 2026-06-03 18:13 the rolling last-2 pair was:
- `53336920` champ_refine_adaptivek — public **860**, but only ~20 min old (NOT settled)
- `53332500` champ_computeByShips_on — public **1177** (settled, a keeper)

The known-good `53324164` champ_adaptiveK_on (public **1188**) is OUT of the window
but recoverable by resubmit.

**Watch / action (first thing next session):** re-check the settled μ of `53336920`.
- If it plateaus low (≪ 1177) → resubmit `champ_adaptiveK_on` to evict the weak agent
  and restore the floor (Rule 42: 1188 ≫ 860, clean).
- If ≥ 1170 → keep; proceed to the refine regression-tail fix.

No submission was made 2026-06-04 (opening-wait diagnostic was measurement-only).
Cross-branch (Rule 42): sibling pushes now evict one of {refine, computeByShips},
not adaptiveK — coordinate on the claim board before pushing.
