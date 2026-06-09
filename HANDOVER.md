# HANDOVER.md — next-session brief

## Mode

**Multi-tick + recap shipped (sub 53390700, PENDING). Strategic-value
bonuses (denial + opening) regressed at default weights and were
deferred for empirical calibration.** Next session opens with reading
the live μ for sub 53390700 after warm-up, then deciding among
(a) calibrating + re-shipping strategic, (b) the multi-tick wallclock
optimization path, or (c) the force-concentration restructure of
greedy selection.

## Live status (after 2026-06-05 11:26 UTC submit)

- **Newest (#1):** `producer_plus_multi_tick_recap_on.py`, sub
  **53390700** (2026-06-05 11:26 UTC), 272 870 B. Multi_size + opp_proj
  + multi-tick (K_4P=3, K_2P=2) + recapture penalty. Status: **PENDING**.
  Will climb in TrueSkill warm-up over ~24 h.
- **Backstop (#2):** `producer_plus_multi_opp_def_on.py`, sub **53384340**,
  live **μ = 1285.0** (climbed monotonically 947 → 1159 → 1287 over 28 h).
- **Evicted by 53390700:** sub 53373322 coalitions_on, μ = 1266.6.
- Older subs via `kaggle competitions submissions orbit-wars`.

## What landed this session

1. **Multi-tick opp projection** (commits `4e7c840` → `d714943`).
   `predict_opp_launches_via_mirror` runs K rounds per turn,
   accumulating cumulative_bg threaded into each inner call.
   Eta-frame correctness fix, parse_obs hoist, truncation policy
   fix all from code review.

2. **Recapture penalty** (commits `5ac1803` → `173c952`).
   Per-candidate scorer discount for thin captures opp can recapture.
   Subtracts `prod[T] × turns_lost` in ship units. `K_recap_eff =
   max(1, K_recap - K_opp)` avoids double-counting multi-tick window.

3. **Strategic-value bonuses (DEFERRED)** (commit `8c0c228`).
   Denial bonus (rewards captures opp values) + opening bonus (rewards
   early-game captures, decays to 0 at opening window). Math is sound;
   default weights 0.1 made bonuses 2-10× the typical competitive_score
   magnitude, dominating instead of nudging. Clean A/B vs producer:
   0/4 wins. Code is shipped but the gates default OFF. **Re-tune
   weight to ~0.005-0.02 in next session.**

4. **Submission**: sub 53390700 multi_tick_recap. Bundle sha256
   `049915a4ecdf...`. Rule 45 explicit PI override (n=16 Wilson-lo 0.28
   under 2-worker CPU contention contaminated the measurement).
   Rule 42 GREEN (predicted ~1290 > evicted 1266.6).

## The lesson carried forward

**Default weights for new additive scorer terms must be calibrated
empirically, not estimated by hand.** The strategic regression came
from sizing weights to "comparable" magnitude based on rough math
instead of dumping the actual competitive_score distribution. Before
shipping the next scorer term, do a 5-minute probe:

1. Run one full game with logging enabled
2. Dump per-candidate competitive_score values
3. Set new term default weight so its typical magnitude is 5-15% of
   the existing-score median

For the strategic bonuses specifically: target weight ≈ 0.005-0.02
based on the calibration.

## PI guidance for next session

- **First diagnostic:** clean single-process A/B for multi_tick_recap
  vs producer (workers=1, n=4 or n=8). Confirms whether the mechanism
  lifts cleanly over the base. ~3 minutes of work; closes a real
  question.
- **Live μ check:** read sub 53390700 after 2026-06-06 ~11:30 UTC
  (24 h after submit). Compare to the 1285 backstop.
- **Decision branch:**
  - If multi_tick_recap clean A/B is ≥60% → strategic bundle is the
    natural next bet **after** weight re-calibration.
  - If multi_tick_recap clean A/B is parity (50%) → the scorer
    framework may have reached its ceiling; consider force-concentration
    (the deeper restructure of greedy_select) before another scorer
    term.
- **Submission budget tomorrow:** 5/day. Use 1-2 max while sub 53390700 settles.

## Locked for later sessions

- **Strategic-value bonuses re-tuned.** Weight 0.005-0.02 after
  empirical calibration probe. The math + module + tests are all
  shipped (gated OFF); just re-tune and re-A/B.
- **Multi-tick wallclock optimization.** Skip regroup/greedy in opp
  planner calls (1.5-2× free), batch opps within a round (3×), eta-
  shifted reuse for rounds 1+ (further 2×). Cumulative ~3-5× speedup.
  Pre-requisite for heavier mechanism stacks.
- **Force-concentration.** Restructure greedy_select to pool sources
  onto top targets. Bigger surgery; addresses the "spread thin"
  failure mode at a different altitude than scorer terms.
- **Value-head ML** (locked further out per
  `knowledge-base/thoughts/2026-06-05-ml-next-steps-locked.md`).

## Next action checklist

1. Read live μ for sub 53390700 after ~24 h (target 2026-06-06 11:30 UTC).
2. Run clean single-process n=4 A/B for multi_tick_recap vs producer
   (`python fast.py eval submissions/producer_plus_multi_tick_recap_on.py
   --vs agents/producer/producer_agent.py --max-seeds 2 --workers 1`).
3. Based on the two results above + PI input, pick from the
   "decision branch" in the PI guidance section.

## Files of note touched this session

- `agents/producer/orbit_lite/opp_projection.py` — multi-tick K-round
  extension; eta-frame fix; parse_obs hoist; truncation policy.
- `agents/producer/orbit_lite/recapture.py` (new) — recapture-penalty
  leaf-scorer term.
- `agents/producer/orbit_lite/strategic_value.py` (new) — denial +
  opening bonus leaf-scorer terms; shared `_compute_captures()` helper.
- `agents/producer_plus/main.py` — env-var getters + call sites for
  all three new mechanisms.
- `scripts/bundle_producer_plus.py` — added "recapture",
  "strategic_value" to ORBIT_LITE_ORDER; 6 new ENV_VARIANTS entries
  (recapture_penalty, multi_tick_recap, denial, opening, strategic,
  multi_tick_strategic).
- `agents/producer_plus/producer_plus_*.py` — 6 new shim files.
- `tests/test_recapture_penalty.py` (new) + `tests/test_strategic_value.py`
  (new) — env-getter tests, synthetic unit tests, drift checks across
  all 9 shipped shims.
- `audit/2026-06-05-postmortem-recap-shipped-strategic-regressed.md`
  (new) — session postmortem.
- `knowledge-base/thoughts/2026-06-05-strategic-bonuses-regressed-
  recap-shipped.md` (new) — second-brain entry with the
  calibration-lesson and the re-tune plan.
- `state/MULTI_BRANCH.md` — push-claim rows for both the
  multi_tick_strategic (declined) and multi_tick_recap (submitted)
  attempts.

## Rules in force unchanged

- Rule 0: plain English with PI.
- Rule 1 + 12 + 42 + 45 + 46: submission discipline; Rule 45 explicit
  PI override on this submit, documented.
- Rule 38: env knobs default OFF, bit-identical to baseline.
- Rule 40: modeling-correctness over restriction-tuning.

## Open questions (carry to next session)

1. Does multi_tick_recap lift μ over multi_opp_def's 1285 base on the
   live ladder?
2. What weight makes strategic-value bonuses non-regressive? Empirical
   probe expected to show ~0.005-0.02.
3. Does the 4P cycle stalemate break with multi_tick_recap? 1-hour 4P
   self-match this session was contention-stuck and unrecoverable.
4. After multi-tick wallclock optimization, what mechanism does the
   freed budget unlock?
5. Is force-concentration the right next altitude lever, or does the
   scorer framework have more lift left at the term-tuning level?
