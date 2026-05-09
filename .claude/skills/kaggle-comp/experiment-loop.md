# Experiment-loop — full spec

Nested inside Day-loop step 3. One run per candidate mechanism. See
`loops.md` for the router and loop-interaction context.

## Trigger

An experiment is selected (Day-loop step 2).

## 1-hour cap (revised 2026-05-04)

The cap applies to a **SINGLE FOLD's actual wall time on full data**,
not to the extrapolated 5-fold-both-anchor projection. If one fold
completes within 1h on production hardware, run it, see the result,
then decide whether to pursue the full 5-fold or shrink. This unblocks
heavyweight mechanisms (NN architectures, deep CatBoost on CPU) for
at least an exploratory single-fold probe.

Mandatory complement: **Rule 13** — heavy NN, deep CatBoost
(depth ≥ 8) 5-fold, and any 5-fold whose local-CPU projection > 1h
go to Kaggle GPU notebooks (P100 / T4×2). Don't declare
"not cost-justified" on local CPU alone.

## Steps

```
1. Heuristic baseline (skip if N/A):
   - closed-form rule / threshold / hand-coded
   - bound the lift available
   - if heuristic already clears the candidate's predicted lift,
     stop here

2. Smoke (5-min cap):
   - 1 fold / 1 trial / 50k subsample
   - verify: COMPLETE status, expected artifacts emitted, no shape
     mismatch, no permission errors
   - if smoke fails, fix and re-smoke; do NOT push to production

3. 1-fold time-probe:
   - full feature set, full data, fold 0
   - measure wall time on same hardware as production
   - if 5-fold projection ≥ 1h on local CPU, EITHER shrink config
     (folds, n_ens, epochs, subsample) OR port to Kaggle GPU per
     Rule 13. Don't kill the mechanism on cost grounds without
     evaluating both options.

4. 5-fold production:
   - emit OOF + test .npy artifacts to scripts/artifacts/
   - compute candidate OOF on metric

5. 4-gate filter:
   - G1 standalone OOF clears anchor at recipe-bias
   - G2 blend with anchor lifts at α* > 0
   - G3 net rare-class-flip ratio ≥ 0.5
   - G4 direction asymmetry: more correct flips than incorrect

6. Minimal-input meta sanity check (if stacking candidate):
   - train candidate meta with ONLY 2 components (anchor + new)
   - if 2-comp OOF < anchor at recipe-bias, STOP. Do not LB-probe.
   - if pool > 5 bases, also run an L1-coef prune sweep before
     LB-probing the full pool (Day-3 m5h finding)

7. Reviewer audit:
   - subagent invocation, no parent-context anchoring
   - emit gate-result audit entry to audit/YYYY-MM-DD-<name>.md

8. Ask PI to submit:
   - present: candidate name, OOF, gate results, predicted LB lift,
     remaining slots today (out of 10/day per Rule 12)
   - PI confirms → single-shot kaggle competitions submit (Rule 1)
   - log result, update calibration ladder
```

## Stop conditions

- Any gate fails (G1-G4 or 6 minimal-input-meta sanity).
- PI declines submission.
- LB result lands (loop completes; control returns to Day-loop step 4).

## Common Experiment-loop failure modes

Distilled from `audit/friction.md`. These are the patterns that have
killed runs more than once:

- **Probe-extrapolation drift** — 1-fold probe time × 5 underestimates
  full-anchor 5-fold by 2-3× when high-cardinality cats are present
  (e.g., Driver=887 with depth=8). Apply a 2-3× safety factor on
  cat-heavy mechanisms before "fits in 1h" verdict.
- **Tail-pipe buffering** — `python script.py | tail -40` buffers all
  output until pipe close. On timeout-kill, no log reaches disk.
  Always redirect to file: `python script.py > log 2>&1 &`.
- **Subagent monitor truncation** — Subagents that launch a long
  python via Monitor and rely on completion notifications return
  prematurely with truncated output. Contract: "run python > log
  2>&1; wait for exit; read log; summarize". Forbid Monitor +
  early-exit.
- **Schema-grep before FE** — Always check `train.columns` and the
  brief data dictionary before deriving a feature. Re-deriving
  existing columns is no-op.
- **`kaggle kernels init` string-bool defaults** — Kaggle silently
  treats string-quoted booleans as `false`. Edit metadata to bare
  `true` / `false` before pushing. Use `Path('/kaggle/input').rglob`
  for data discovery rather than hardcoded paths.
