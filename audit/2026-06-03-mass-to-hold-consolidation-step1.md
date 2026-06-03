# 2026-06-03 — mass-to-HOLD consolidation: STEP 1 inert-check (Plan v5)

Branch: `claude/champion-strategy-rules-00JzI`. Session: PI greenlit building a
"consolidation / massed-strike proposer" for `holdgrab` (the new denial-centric
rollout champion). Recon flipped the build into a **pre-registered inert-check
first**, because the axis is mostly-falsified.

## Why we did NOT just build the proposer

Recon (3 Explore agents + 1 Plan agent) surfaced that the general
**capture-coalition** axis was built and falsified 2026-06-02 on the sibling
`baseline` lineage (`audit/2026-06-02-lever-retest-null.md`,
`knowledge-base/thoughts/2026-06-03-coordination-closed-value-leaf-negative.md`):
- greedy-as-replacement HURT (9/16 vs champion 16/16 vs v7_0);
- the augment-not-replace coalition refiner was **completely INERT** —
  `generate_sync_coalitions` yields ZERO candidates in real games, because
  sources accumulate enough ships to **solo-capture** their targets. Closed per
  Rule 37.

The ONE untested residual: `knowledge-base/thoughts/2026-05-31-sync-coalition-
k-gate-fix.md` found 2 of 4 coalitions **captured-but-didn't-HOLD** →
*"the next ceiling is HOLD, not capture."* So this session targets **mass-to-
HOLD only**: pool ship budget across sources so a high-value ENEMY planet
(worth DOUBLE under the denial value) that NO single source can HOLD — only
PRESSURE — can be captured-and-held. Note `sizing.py`: Orbit Wars is Lanchester
**linear**, no concentration bonus, so this is **pure budget-pooling** — narrow
by construction, hence the inert-check gate.

## What was built (committed `2195b0b`, all default-OFF, agent byte-identical)

- `agents/holdgrab/config.py`: `consolidate_*` flags (master switch
  `consolidate_hold=False`, `max_targets=3`, `max_legs=2`, `max_eta_gap=3`,
  `consolidate_neutral=False`, `instrument_consolidation=False`).
- `agents/holdgrab/chooser.py`: `Coalition` dataclass +
  `consolidation_opportunities(view, cfg, spendable)` — the shared enumerator
  for both the census and (if GO) the STEP-3 proposer. Reuses the EXACT
  `_capture_candidates` sizing primitives (`_seed_eta`, `model.ships_at`,
  `contest_force`, `ships_to_capture_and_hold`, `opp_reach_tick`). Opportunity =
  enemy target where (1) no single source solo-HOLDs, (2) ≥1 source can solo-
  CAPTURE (Tier-2 today), (3) nearest 2..max_legs sources' pooled budget clears
  `need_hold` at a synced `common_eta` (ETA-gap ≤ 3), ranked by `planet_value`,
  top-M. NOT yet wired into `select`/`choose`.
- `scripts/probe_consolidation.py`: the census harness with FROZEN thresholds.
  Plays the focal seat **closed-form** (`use_rollout=False`, ~20× faster than the
  800ms rollout; the opportunity enumeration is rollout-independent), both seat
  orders, over the geometry panel.

## Pre-registered verdict thresholds (frozen, no goalpost-shift)

- **NO-GO / STOP:** opportunities on `< 1%` of focal turns AND median `< 2`/game
  → inert like the baseline refiner; close the axis (Rule 37), do not build STEP 3.
- **GO / build:** `≥ 3%` of turns OR median `≥ 5`/game, in top planet_value tercile.
- **MARGINAL (1–3%):** surface the census to the PI before spending build budget.

## Result

- **Early read (4 games vs v7_0, closed-form):** 2 opportunities in 608 focal
  turns = **0.33%**, median **0/game**. Both were high-value enemy planets
  (value ~2618 / ~2992) → the mechanism *fires correctly*; the *opportunity* is
  rare. Points **NO-GO**, consistent with the 2026-06-02 capture falsification.
- **Full panel census (32 geometry seeds × {v7_0, v4_planner, v3.5.1} × both
  seat orders = 192 games):** launched in-flight at wrap; verdict to be read
  from the committed, reproducible tool next session (`python
  scripts/probe_consolidation.py`). <!-- VERDICT-PENDING: paste final tally here -->

## Disposition

If the full census confirms NO-GO (expected): close the mass-to-HOLD axis per
Rule 37, keep the enumerator as default-OFF latent capability (zero agent
impact), and pivot — the higher-EV lever per today's cross-cutting lesson is
fixing A/B throughput / submit-and-measure, not more proposer-side coordination
mechanisms (the coordination seam is empirically small at the champion's level).
