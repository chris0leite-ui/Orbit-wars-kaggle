# 2026-06-02 — marco opening-exploit axis closed, pivot required

**Status:** ACTIVE — relevant to next-session planning.

**Flag:** the `claude/game-theory-winning-strategy-SEU7P` branch has
exhausted axis-1 (marco-EAM opp model + adversarial rerank) per Rule 37
(3-variant axis cap, partial — this is the FIRST attempt). Next
opening-exploit attempt MUST be a different axis (swarm tactic,
deception layer, multi-turn opp prediction) — not another tuning of
this rerank.

**Durable artefacts kept in tree (don't churn):**
- `lib/opp_marco.py` (marco port — faithful, useful as a Tier-3
  component for any future opp-pool model).
- `lib/opp_model.py` Tier 3 wiring.
- `scripts/probe_rerank_*` diagnostics.
- The wait-turn + horizon fixes in `agents/baseline/chooser_trajectory.py`.

**Don't iterate:** the adversarial-rerank-vs-one-ply-opp-model design.
v2 of this would just be more tuning.

**Pointer to update:** `state/MULTI_BRANCH.md` Closed tracks table.
Add row with:
- Axis: "Marco-EAM opp model + adversarial rerank (axis-1 of
  opening-exploit family)"
- Branch: `claude/game-theory-winning-strategy-SEU7P`
- Verdict date: 2026-06-02
- Evidence: parity gate PASS 84.6%; rerank fires 60%, promotes 14%;
  n=8 vs anchor 50% Wilson-lo 0.22; n=13 vs romantamrazov 38.5%
  Wilson-lo 0.18; both <50% gates Rule 43c/45.
