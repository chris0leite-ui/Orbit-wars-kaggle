# FLAG — local A/B noise floor inflated by CPU-variance coupling

**Raised:** 2026-05-28 PM2 by claude/kaggle-submission-review-gZsCu
**Action:** carry forward; address as a deliberate work item in a future session.

## The flag

Our local A/B harnesses (fast.py play, scripts/ab_quick.py) use the
production chooser, which couples its search depth to live CPU
performance via `affordable_validate_cap()` in
`agents/baseline/chooser.py` (and the analogous block in
`agents/baseline/chooser_trajectory.py`, the production default).
On every turn the chooser measures `per_cand_ms` on the live CPU
and sets `n_aff = (wallclock_ms − 50) / per_cand_ms`. Same seed,
same agent, different CPU load → different `n_aff` → different
candidate set considered → different chosen move → different game.

**Empirical:** 4 runs of seed=3 vs peak_anchor gave 3W/1L and step
counts spanning 218-359 in production mode. Pinning `n_aff` to 60
(via WALLCLOCK_MS=2000 + OMP=1 + taskset) converged outcomes to
4/4 DRAW with step counts spanning 204-236.

## Why we are not fixing now

1. **It alters playstyle.** Pinning `n_aff` high (60) caused the
   confirmation diagnostic to converge on DRAW — both agents at
   higher capacity played to stalemate. The constant we ship must
   be calibrated against ladder performance, not arbitrary.
2. **The leaf_pv_2p submission is still climbing.** Pushing a
   "stable-chooser" variant now would consume submit budget and
   evict our backstop (PV_ETA μ=1163.5) before its lifetime read
   is known.
3. **Ladder μ drift has multiple causes** (opponent pool churn,
   σ shrinkage, scores still climbing per comp-context.md). We
   cannot attribute the 50μ peak-resubmit drift purely to compute
   variance, so a "robustness fix" pitched on ladder-stability
   grounds is not yet evidence-justified.

## Implications for A/B methodology today

- A 70% n=10 A/B with Wilson-lo 0.40 should NOT be treated as the
  same kind of evidence as a 70% n=10 A/B on a deterministic agent.
  The actual sampling unit is "(seed, CPU state) pair," not "seed."
- When an A/B comes in inconclusive (Wilson-lo in [0.35, 0.55]) and
  the next step is "submit anyway," recall this flag.
- Possible local mitigation while waiting to address: set
  `OMP_NUM_THREADS=1` + `taskset -c <core>` for repeatable local
  runs. Does not fix the n_aff-vs-CPU coupling but removes the
  BLAS-parallel layer.

## When to revisit

After leaf_pv_2p's lifetime μ stabilizes (next 1-2 sessions). If our
ladder position is plateauing and other levers feel mechanical,
shipping Patch 2 (constant `n_aff` calibrated via sweep) becomes
attractive both as a stability play and as a determinism guarantee
for future A/Bs.

Related: `audit/2026-05-28-postmortem-pv-eta-pm-pv-eta-and-silent-turns.md`,
`knowledge-base/thoughts/2026-05-28-pm2-compute-variation-and-leaf-pv-2p.md`.
