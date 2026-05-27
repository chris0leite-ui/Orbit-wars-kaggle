# reach_frontier

A closed-form chooser that targets the production-time integral
`S_i ≈ S_i(0) + Σ p̃·τ_p^i`. **Framework replacement** for the v9 K=10
rollout chooser — not a layer stacked on top.

Per turn it computes a reach-time frontier ρ_i(p) for each player,
classifies planets by Voronoi cell (mine / contested / opp), computes
hold-time h(p) = ρ_opp(p) − ρ_me(p), and picks launches by Hungarian
assignment maximising `Σ p̃·h(p) − costs`.

**References:**

- `knowledge-base/concepts/reach-frontier-doctrine.md` — the math.
- `knowledge-base/concepts/reach-frontier-chooser-design.md` — the
  implementation spec, including substrate references.
- `knowledge-base/concepts/evaluation-metrics.md` — Rule 48 protocol
  for evaluating this class of agent (production-share primary,
  hold_fraction secondary, mandatory 2P/4P split).

**Known v1 biases (documented, not fixed):**

1. **Symmetric-strength opp model.** ρ_opp assumes every opponent plays
   its best reach. Against weak opponents this is conservative and we
   leave value on the table (safe direction).
2. **Garrison-growth via WorldModel.ships_at.** Uses the combat-resolved
   timeline; correct under our own ledger but doesn't model opponent
   reinforcements arriving after we commit.
3. **2P-first.** 4P "kingmaker" mitigation (doctrine §8.3) ships in v2.
   v1 runs in 4P without crashing but per-turn decisions are 2P-shaped.
