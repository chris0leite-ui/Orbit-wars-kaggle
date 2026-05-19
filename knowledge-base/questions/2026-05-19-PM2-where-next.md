# Open questions — 2026-05-19 PM2

## Q1. Should next session start with primitive verification, or with wrapping?

Two competing signals from this session:

- **Promoted Rules 41+42** point at: verify each primitive against the
  env's ground-truth physics before any chooser work. Concrete
  deliverable: a `tests/test_primitives_on_replays.py` that runs each
  capture-math + ETA primitive against N real replay positions and
  cross-checks vs `predict_fleet_fate` / env-step.
- **The 12/32 baseline_veto data point** points at: wrap baseline,
  add advisor signals, don't try to replace. Concrete deliverable:
  a richer wrapper that uses our predicate/portfolio as a priority
  bump on baseline's existing emits rather than as a replacement.

PI decision required next session: which is the higher-EV next step?

## Q2. Is the cluster tablebase salvageable as an advisor signal?

29 audited clusters with depth-8 minimax verdicts. We tried using
them as a veto on trajectory_roi (no signal) and on baseline (12/32
INCONCLUSIVE). Could the same data be useful as a PROBABILITY signal
("planet is in a bouncing-launch cluster") that baseline's chooser
uses to demote candidates rather than reject them? Or is the cluster
abstraction itself wrong (cluster-local sub-game is locality-blind)?

## Q3. What's the right "real-position oracle" cadence?

Rule 42 candidate says every primitive should be checked on real
replays. Per-commit? Per-session? Per-architecture-iteration? The
`scripts/probe_emits_via_fate.py` diagnostic is a starting point but
it just counts outcome buckets — we need a per-primitive verification
contract.
