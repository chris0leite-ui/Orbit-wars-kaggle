# 2026-05-28 PM2 — compute variation as A/B noise, leaf_pv_2p climb start

## What happened this session

1. Shipped `BASELINE_LEAF_PV_2P=1` (sub 53117942) — re-enabled the 2P
   leaf production-PV term disabled since 2026-05-18. Local A/B vs peak
   anchor: n=10, 7-3 (70%, Wilson-lo 0.40). Mechanism check vs v4_planner:
   5-0, including seed=2 (silent-turns smoking gun) flipped L→W.
2. Submitted despite Rule 45 n=32 gate not cleared (PI explicit override).
   Evicted PEAK RESTORE μ=1114.5; PV_ETA μ=1154.8→1163.5 backstops.
3. **Same-seed A/B non-determinism investigation.** Triggered by seed=3
   step-count drift between n=5 and n=10 runs (250→183, same outcome).
   Hypothesis tested: PYTHONHASHSEED randomization (rejected — fixing
   it did not eliminate variance). Confirmed hypothesis: **wallclock
   coupling in `affordable_validate_cap()`**. The chooser measures
   `per_cand_ms` on the live CPU each turn and computes
   `n_aff = (wallclock_ms − 50) / per_cand_ms` — so the candidate count
   is a function of CPU speed. Slow CPU → fewer candidates → different
   move → different game.
4. Confirmation A/B: 4 parallel runs of seed=3 with `BASELINE_WALLCLOCK_MS=2000`
   + `OMP_NUM_THREADS=1` + taskset-pinned. Outcomes converged to 4/4
   DRAWS (vs 3W/1L before). Step-count range shrank 4× (141→32).
   CPU-variance is the dominant non-determinism source.
5. **Sub 53117942 (leaf_pv_2p) reading at session end: μ=921.3.**
   **This is a climb starting point, NOT a verdict.** All new submits
   enter at μ₀=600 and climb as they play games. PV_ETA at μ=1163.5
   has had hours more games to settle.
6. **PV_ETA confirmed as new peak on the ladder** (μ=1163.5, above the
   historical 1144-1165 band).

## What I'm carrying out of session

- **PI corrected me on "scores don't settle"** — promoted to
  `comp-context.md::SCORES DO NOT SETTLE` block. This is the
  single highest-frequency recurring agent misread in this comp.
  Future sessions must not call a same-day reading a verdict.
- **Compute variation is a real A/B confound, BUT ladder μ drift is
  multi-causal.** Opponent pool churns, σ shrinks with games, our
  σ resets on resubmit. Cannot attribute ladder noise solely to
  local compute variance.
- **The fix surface is mapped.** Production chooser is
  `chooser_trajectory.py` (not `chooser.py`); same wallclock pattern.
  Patch 2 (fixed `n_aff` constant) is the smallest robust change.
  Calibration of the constant is the open question — too low
  shallow-searches, too high (`n_aff=60`) caused both agents to draw
  in the diagnostic.
- **leaf_pv_2p's lifetime live μ is unknown.** Next session: check
  back, decide if a follow-up submission is needed. If μ climbs to
  ≥1100 the leaf-PV thesis is partially vindicated; if it stalls at
  ~950 the 2026-05-18 calibration-debt warning was correct.

## What I would NOT do again

- Treat a sub's first-hour reading as a verdict (PI has corrected
  this multiple sessions; comp-context.md now states it loudly).
- Spend a submission slot on n=10 evidence when n=32 was 19 min away.
  In retrospect, the wallclock-investigation we did AFTER the submit
  would have been more valuable BEFORE — it would have re-framed the
  A/B noise floor and possibly raised the n=32 priority.
