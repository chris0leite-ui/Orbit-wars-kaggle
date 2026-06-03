# FLAG — 2026-06-03: rolling pair holds TWO uncalibrated agents; known-good μ1170 is OUT

**Risk to the rolling-pair floor.** After submitting refine (`53336920`), the
Kaggle rolling-last-2 pair is:

- `53336920` champ_refine_adaptivek — **uncalibrated** (local 70% h2h, no live μ yet)
- `53332500` champ_computeByShips_on — **uncalibrated / weak local** (7/16 solo A/B)

The known-good **champ_adaptiveK_on (μ~1170.4) was EVICTED** from the window — it
still exists (sub `53324164`) and is recoverable by resubmit, but it is NOT
currently one of the two agents scored for final evaluation.

**Why this is a flag:** if BOTH rolling agents settle below ~1170, the pair floor
drops and we've traded a known 1170 for two unknowns. This was a PI-explicit
"submit now" override (informed). 

**Action / watch:**
- Check `53336920` settling μ next session (see questions doc). If it converges
  below ~1140 (and computeByShips is also weak), **resubmit `champ_adaptiveK_on`**
  to restore the 1170 floor — that resubmit evicts the weaker of the current pair.
- Cross-branch note (Rule 42): other branches submitting will now evict one of
  these two uncalibrated agents, not adaptiveK — coordinate via the push claim
  board before pushing.
