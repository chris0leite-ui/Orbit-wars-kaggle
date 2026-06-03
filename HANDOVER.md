# HANDOVER.md — next-session brief

_Refreshed 2026-06-03. Read this first (Rule 15). Also read `state/MULTI_BRANCH.md`
(live rolling pair / track registry / region-mvp open track) per Rule 44._

## This session's outcome — region/chunk-aware MVP (parity, banked)

Branch **`claude/region-mvp`** (off champion HEAD; commits `788af05` code +
`929ff94` docs, both pushed). Built and validated a region/chunk-aware agent: the
decision unit becomes a **cluster of planets** (orbital-parameter buckets), not a
single planet. Three verbs behind `BASELINE_REGION` (default OFF, byte-identical
champion): **bias** candidates toward high-value *predictable* contested regions,
**advance** idle rear mass to the frontier region (own→own redeploy), and a gated
**GAIN** stub. Plus a separate `BASELINE_HORIZON_DECAY` adapter (rollout-depth
floor deep early → champion-late). Design law obeyed (from the reach-frontier /
analytical-slice closures): **feed the rollout, never replace it.**

**Result: PARITY with the champion** — region-only 15/32=46.9% [0.31,0.64];
region+horizon 7/16=43.8% [0.23,0.67]; both INCONCLUSIVE. Timing clean
(region max 929 ms < champion's own 1084). Off-is-identical proven twice
(216-call replay + 80-state proposer-default parity, 0 mismatch). **No submission**
(parity isn't worth a rolling-pair slot, Rule 42/43).

**Why it's only parity (the structural diagnosis):** the chooser selects by its
**rollout score**, not by candidate cheap-delta — so biasing candidate *order*
only changes which candidates get validated under the cap; it cannot override the
rollout. The bias is gentle by construction; the advance pass moves ships
net-neutrally.

**Bonus finding (banked):** idle-source probe (`scripts/probe_idle_sources.py`,
1922 rows) — the champion fires ~1 of ~13 eligible planets/turn (**~90% idle**)
even in close mid-game. This **refutes the "source-saturated" premise** that
closed the joint-coordination axis on 2026-06-02 (that null was only measured in
blowout wins). The idle capacity is real; whether deploying it is correct
(hoarding) or a conversion gap is still open.

## NEXT-SESSION PLAN

**Top lever (the one this session's diagnosis points to):** add region value as an
additive **term in the chooser's final score** (`chooser_trajectory.score_candidate_v4`
leaf/delta), so a high-value-region capture is preferred at equal rollout delta —
"feed the rollout" at the *scoring* layer instead of candidate reordering. This is
the most likely path from parity to lift; ~1-hour experiment. Gate behind a new
flag, off-is-identical, single-variable A/B at n≥32 vs the table-ON champion using
**bundle-vs-bundle** (self-contained modules avoid the `clean_ab`/module-constant
contamination — see the session's harness notes).

**Secondary, if the score-term lever is null:**
- Tune the **advance pass** aggressiveness (reserve thresholds, max launches,
  improvement floor in `region_advance_pass`) — it's the real behavioral lever.
- A/B **horizon-decay in isolation** (never tested alone; only stacked → parity).
- The three shelved features from the prior handover (team-up `BASELINE_JOINT_SYNC`,
  opening MILP `BASELINE_OPENING_MILP`, composite value head) remain flag-flip
  candidates **with the table ON**.

**Method reminders:** one lever at a time (Rule 37); A/B bundle-vs-bundle to avoid
in-process env contamination; `fast.py bench` (not eval) for timing; champion
control = identical config minus the new flag; submit only on Wilson-lo ≥ 0.55
panel + n≥32 h2h + Rule 42 eviction check.


**Deferred — reassess WITH THE PI after the three above (do NOT start this session):**
- **2-hop redeploy** (shuffle forward so a follow-up can capture) — reverted
  (`5ec6a0d`); needs rebuild from spec `727e1bf`.
- **Reach-frontier chooser** (from-scratch value chooser) — separate agent
  (`agents/reach_frontier/`), had a hold=0 bug; it *replaces* the chooser, so
  evaluate whole-agent after the fix, don't stack.
- Cheap re-checks: H41 pv_horizon floor (`9ebd311`), PV_ETA tuning.

**Do NOT re-open** (dead for non-table reasons): H44 "fleets die in flight" (false —
sun/OOB only), 4p-cushion (4/32), b5 reward-axis (0/32), flat expansion-credit
(targeted the hoarding loss mode we refuted; real loss mode is the expansion race).

Full provenance (commit hashes per feature, the two confound mechanisms, deferred
detail) is in the session plan, mirrored into git history of this file.
