# HANDOVER.md — next-session brief

_Refreshed 2026-06-04 (`claude/kaggle-submission-strategy-JzIAr`). **Read
`state/STRATEGY.md` first** — it is the canonical "what we're running" doc. This file
is the next-session brief only._

## ⚠️ Docs reorganized 2026-06-04 — single-strategy, observation-driven mode

Merged main's minimal-docs structure into this branch (PI directive): `CLAUDE.md`
slimmed to ~11 binding rules (full 49-rule set archived at
`state/_archive/CLAUDE-JzIAr-full-49rules.md`, re-promote on need), and added
`state/STRATEGY.md` as the read-first canonical doc. Working mode is now one strategy,
one observation → one mechanism → one push (no multi-axis exploration).

## Live status (2026-06-04)

Rolling pair: `53336920` champ_refine_adaptivek (**μ ≈ 1179**, active) + `53332500`
champ_computeByShips_on (**μ ≈ 1180**, active). The adaptive-K base `53324164`
(μ 1188) is **evicted/frozen** — its μ is NOT comparable to the active pair (older,
weaker field). Refine's local edge (+9 net, 70% h2h, single-opponent) is
live-uncontradicted but **not yet live-confirmed** — see `state/STRATEGY.md` evidence
box. To confirm: a same-field resubmit of the base, or a multi-opponent panel.

## 2026-06-04 session outcome — opening-wait diagnostic (the horizon hypothesis is dead)

PI flagged a live loss (seed **722289020**, perimeter-ring / central-sun map): we
sit idle early, Merchant API takes the whole ring by step 90. PI's read: sparse map →
targets out of reach within horizon K → no candidates → idle. **Tested it directly
(`scripts/opening_starvation.py`) and REFUTED it:**
- Step-0 scan, 160 seat-boards: **0%** have nearest neutral past K_OPEN=20; 1% past
  the static floor 10. The map's "sparseness" is cross-map arc distance around the
  sun — adjacent-ring grabs are cheap (ETA 4–10). The horizon never zeroes candidates.
- Full opening trace (seed 722289020): **LAUNCHED 4 / WAITED-with-candidates 27 /
  STARVED-by-horizon 0** over 31 turns. The proposer offered 2–12 candidates *every*
  turn; the agent declined them. **The lever is the value function's early-expansion
  appetite, NOT the horizon constant K.** Full record: `audit/2026-06-04-opening-wait-diagnostic.md`.
- **Caveat (do not over-read):** this is local self-play (waiting is symmetric/even
  there). Proven: *we* wait 27/31. NOT proven: that it's *why* we lost — needs an
  aggressive early-expander to show the waiting is exploited. And "launch more early"
  already regressed — but in self-play, the confounded cohort. No submit this session.

## NEXT STEPS (priority order)

**1. Opening-appetite experiment (NEW — the decisive, cheap, no-build test).** Pit
the agent vs a deliberately **aggressive early-expander** (NOT self-play; build/borrow
one — the champion mirror cannot differentiate, it's a hoarder too — friction
`wrong-ab-instrument-champion-mirror`). Measure: (a) opening launch-rate gap, (b)
territory/production gap by step 30, (c) whether a higher early-launch appetite (lower
chooser launch threshold OR early-game expansion bonus) **wins vs that class while
staying neutral in self-play**. Cut the A/B by opponent class (Rule 41) — that splits
"waiting is exploited" from "launching more overextends," which the prior self-play
regression conflated. Tool ready: `scripts/opening_starvation.py` (step-0 scan +
launch overlay). Questions: `knowledge-base/questions/2026-06-04-opening-appetite.md`.

**2. Refine regression tail (only if refine settles well).** Refine **broke 4 of 32
games the champion won** (long contested seeds: 2P1, 9P0, 13P0, 14P0). The
*generator* is fine — the misfire is in **which coalition the oracle picks** in long
games. Hypothesis: the marginal-gain horizon is too short to see the downside (the
teamwork strike commits ships that leave a planet undefended / get recaptured
later). Fixing this could turn **+9 → +13 net** without touching what works.
- *Replays were in `/tmp` and the container restart wiped them — step 1 is re-run
  `scripts/_step3b_adaptivek_winrate.sh` (saves to `/tmp/refine_adaptivek_replays`)
  to regenerate the 4 broken-seed games, then trace the bad coalition.*

**3. compute-by-ships × refine ("best of both worlds").** Full plan in
`knowledge-base/concepts/refine-x-computebyships-compatibility.md`. Hypothesis:
*complementary* — compute-by-ships (`BASELINE_COMPUTE_BY_SHIPS=1`, per-source search
breadth/depth scaled by ship surplus) helps **high-ship** planets solo-expand;
refine helps **low-ship/contested** planets coordinate. Test: (a) combined wallclock
re-bench (both add compute), (b) clean_ab combined vs refine-alone (n≥32),
(c) coalition-count with compute-by-ships on vs off (cannibalization check).
*Caveat: compute-by-ships was 7/16 standalone — not a guaranteed add.*

**4. Finish the Rule 43 panel (cheap loose end).** The weak-opp legs
(v7_0 / v4_planner / v3.5.1) never completed before the restart. Refine ≈ champion vs
weak opponents and the champion crushes them → should pass fast. Closes the gate:
`python fast.py eval submissions/champ_refine_adaptivek.py --vs-panel default --require-h2h submissions/baseline.py --workers 4`

**5. Bigger bets (only if refine settles well):**
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
- `scripts/opening_starvation.py` (2026-06-04) — opening diagnostic: `--scan START N`
  for the cheap step-0 map-sparsity scan (no game), or `<opp> <seed> <window> <trace_seed>`
  for the per-turn candidate-availability vs actual-launch overlay. Refuted the
  horizon-K-starvation hypothesis (`audit/2026-06-04-opening-wait-diagnostic.md`).

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
