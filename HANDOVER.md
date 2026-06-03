# HANDOVER.md — next-session brief

_Refreshed 2026-06-03 (`claude/kaggle-submission-strategy-JzIAr`). Read this first
(Rule 15). Also read `state/MULTI_BRANCH.md` (live rolling pair / track registry +
the joint-coordination REVERSAL block) per Rule 44._

## This session's outcome (the unlock)

**Reversed the "teamwork doesn't matter" closure and shipped the first agent that
uses it.** The 2026-06-02 session closed the joint-coordination axis (Rule 37) on
"the sync-coalition generator produces nothing." That was a **weak-opponent
confound** (Rule 41): it was only ever measured vs v7_0 / v7_minimax — opponents the
champion crushes, so every planet is source-saturated and the "two planets needed to
take one target" regime never arises. (This was the prior handover's NEXT-STEP #1,
"team-up attacks" — now validated, not null.)

Re-measured the generator directly on **champion-vs-strong-opponent** boards: it
yields **~100+ two-source coalitions per game** (driven by the *resource ratio* —
they appear when my planets are contested / out-resourced, i.e. the games that
decide the ladder). And **exploiting them wins**: the augment-not-replace refiner
(`BASELINE_CHOOSER=refine`: run the champion verbatim, then add only oracle-positive
2-source coalitions that don't conflict with the champion's locks) scored **70.2%
h2h vs the adaptive-K champion** (n=57, Wilson-lo 0.573; the n=32 read was 78%),
**paired +13/−4/+9 net** on 16 matched seeds.

**SUBMITTED** as sub **`53336920`** (`champ_refine_adaptivek.py` = champion +
adaptive-K + refiner + kinematic-table, all baked). New rolling pair = `53336920`
(refine) + `53332500` (computeByShips). **Evicted `53324164` champ_adaptiveK_on
μ~1170.4** from the rolling window (recoverable by resubmit). Rule 46c + timing GREEN
(max 777 ms, 0 over cap). Rule 43 weak-opp panel legs were in-flight at submit (PI
override).

## ⚠️ FIRST THING NEXT SESSION — does refine settle ≥ 1170?

The 70% is **local**; local→live μ is noisy (our notes: 88–94% local → 1150 live).
TrueSkill warm-up starts ~600 and climbs. Check the settling μ of sub `53336920`:
- **Settles ≥ 1170** → real gain; proceed to the regression-tail fix (#1 below).
- **Settles low** → resubmit the adaptive-K champion (`champ_adaptiveK_on`, μ1170,
  just behind the window — recoverable) and re-diagnose refine before building on it.

## NEXT STEPS (priority order — PI-noted 2026-06-03)

**1. Regression tail (highest-leverage, most localized).** Refine **broke 4 of 32
games the champion won** (long contested seeds: 2P1, 9P0, 13P0, 14P0). The
*generator* is fine — the misfire is in **which coalition the oracle picks** in long
games. Hypothesis: the marginal-gain horizon is too short to see the downside (the
teamwork strike commits ships that leave a planet undefended / get recaptured
later). Fixing this could turn **+9 → +13 net** without touching what works.
- *Replays were in `/tmp` and the container restart wiped them — step 1 is re-run
  `scripts/_step3b_adaptivek_winrate.sh` (saves to `/tmp/refine_adaptivek_replays`)
  to regenerate the 4 broken-seed games, then trace the bad coalition.*

**2. compute-by-ships × refine ("best of both worlds").** Full plan in
`knowledge-base/concepts/refine-x-computebyships-compatibility.md`. Hypothesis:
*complementary* — compute-by-ships (`BASELINE_COMPUTE_BY_SHIPS=1`, per-source search
breadth/depth scaled by ship surplus) helps **high-ship** planets solo-expand;
refine helps **low-ship/contested** planets coordinate. Test: (a) combined wallclock
re-bench (both add compute), (b) clean_ab combined vs refine-alone (n≥32),
(c) coalition-count with compute-by-ships on vs off (cannibalization check).
*Caveat: compute-by-ships was 7/16 standalone — not a guaranteed add.*

**3. Finish the Rule 43 panel (cheap loose end).** The weak-opp legs
(v7_0 / v4_planner / v3.5.1) never completed before the restart. Refine ≈ champion vs
weak opponents and the champion crushes them → should pass fast. Closes the gate:
`python fast.py eval submissions/champ_refine_adaptivek.py --vs-panel default --require-h2h submissions/baseline.py --workers 4`

**4. Bigger bets (only if refine settles well):**
- **Extend the AUGMENT path, not the replace path** (greedy-replace already failed
  9/16). Add: **3-source coalitions**, **defensive coalitions** (two planets jointly
  *hold* a threatened target), **wait-coordinated strikes**. Re-justifies the Rule-49
  planner doctrine via the augment framing.
- **Resource-ratio as a strategic-mode signal.** Coalitions arise exactly when
  out-resourced (my sources ≤ defended targets). Detect "I'm being out-resourced"
  and shift mode (more coordination / defense) — a higher-level lever than per-turn
  launch scoring.

## Still-untested shelved levers (from 2026-06-02 handover — table-ON re-test owed)

These were judged null/parity under a table-OFF or singleton-corrupted A/B; both
confounds are gone. Reassess after the refine line above. **Flag-flip, not rebuild.**
- **Optimized opening** — `BASELINE_OPENING_MILP=1` (`lib/joint_solver/opening_planner.py`); prior parity 4/8 (n=8), table OFF.
- **Smart position value head** — `BASELINE_VALUE_HEAD=composite` (`lib/value_heads.py`); never cleanly measured.
- Deferred (reassess WITH PI): 2-hop redeploy (reverted `5ec6a0d`, rebuild from `727e1bf`); reach-frontier chooser (`agents/reach_frontier/`, hold=0 bug, whole-agent eval).
- **Do NOT re-open:** H44 "fleets die in flight" (false — sun/OOB only), 4p-cushion (4/32), b5 reward-axis (0/32), flat expansion-credit (refuted loss mode).

## Tooling built this session (committed; bundles are gitignored, rebuild via script)

- `scripts/refine_seam_contested.py` — direct sync-coalition generator count on real
  boards (Step-2; bypasses the kaggle stderr sandbox).
- `scripts/_step3b_adaptivek_winrate.sh` — refine vs adaptive-K champion, matched-seed
  paired A/B (the validated 78%/70%). `_step3_refine_winrate.sh` = the wrong-base
  static version, kept for the lesson.
- `scripts/_build_refine_adaptivek_bundle.sh` — reproducible build of the submitted bundle.

## Standing gotchas (carry forward)

- **"I didn't set the flag" ≠ "the flag is off."** The kinematic table defaults ON
  (`main.py:896`, `get(...,"1")`). Check defaults before claiming a config.
- **A/B on the LIVE champion config, not the repo default.** The first refine A/B ran
  adaptive-K OFF (non-champion base) and gave a misleading number — PI caught it.
  Confirm the parity anchor is ~50% before trusting a lift.
- **Mid-run win-rate is noisy** (called a wash at 58%, finished 68.8%). Hold verdicts
  to full n (Rule 45).
- **`clean_ab.py` for any refine A/B** — env-var contamination otherwise (refine and
  trajectory both read `BASELINE_CHOOSER`); use a frozen-bundle opponent (immune).
- 5 submits/day, deadline **2026-06-23**.
