# FLAG: bundle decision stack shelved (NOT deleted)

**Date raised**: 2026-05-19
**Branch**: `claude/ml-competition-strategy-PFhzM`
**Scope**: working tree state, future-session expectations

## What the flag says

After PI's ROI pivot, `agents/bundle/` stays in the repo but **is not
iterated further**. Files remain readable as reference; env vars
(`BUNDLE_*`) default to off / no-op.

This is a deliberate freeze, not a deletion. If next-session
exploration finds a primitive in bundle worth reusing in
`agents/trajectory_roi/`, lift it surgically. Do NOT:
- Import from `agents/bundle/` in `agents/trajectory_roi/`.
- Re-enable bundle env vars in new tests / oracles.
- Run new A/Bs that include bundle as a competitor (it's been
  characterised; further data adds nothing).

## What it does NOT mean

- It does NOT mean bundle's `BundleEvaluator.score` (path-integrated
  production weighting) was wrong as a SHAPE. The shape — value
  along a horizon, not just at one instant — is correct and should
  reappear in `trajectory_roi.score()`. The shape stays, the
  coefficients and the chooser-axis variants don't.
- It does NOT mean bundle's failure-mode diagnostic work is invalid.
  The 21.3% bounce rate measurement (5/18 Phase E Phase 0) is real
  data, just not enough lever to move bundle into the champion class.

## Action for next session

When implementing `agents/trajectory_roi/`, the first review point on
the score primitive is: does it reuse bundle's path-integration
shape correctly, or has it accidentally regressed to per-instant
scoring (like the original `agents/simple/roi.py`)? Path-integration
is one of the few bundle insights worth carrying forward.

## Related

- `audit/2026-05-19-phase-3-sweep-and-roi-pivot.md` — session
  audit + verdict
- `knowledge-base/thoughts/2026-05-19-roi-pivot-scenario-gated-clean-architecture.md`
