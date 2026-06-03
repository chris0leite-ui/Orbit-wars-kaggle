# FLAG 2026-06-03 — the "regions / chunks" abstraction is parked; don't re-pitch it

The region/chunk idea has now been falsified at **both** places it can attach to
the decision pipeline:
- **Enumeration layer** (bias hook, `BASELINE_REGION`, 2026-06-03 AM): parity.
- **Scoring layer** (additive score term, `BASELINE_REGION_SCORE`, 2026-06-03 PM):
  null — 0.10/0.20/0.40 = 51.6% / 40.6% / 53.1% vs champion, all below the gate.

Root cause is not placement or tuning: **the rollout already prices the spatial
structure the region heuristic encodes.** A static hand-built board abstraction
adds no signal to an agent that simulates forward. Future sessions will be
tempted to re-pitch "think in regions / clusters / territories" because it's how
humans narrate the board — resist it. The code stays in-tree (default OFF,
byte-identical) as a documented dead-end. If region reasoning ever returns, it
must be as something the rollout genuinely *lacks* (opponent intent, multi-turn
coordination), not a re-skinned spatial prior.
