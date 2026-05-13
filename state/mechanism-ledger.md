# state/mechanism-ledger.md — every agent-design family tried

> One row per family × variant. Rule 21: a family is only "dead"
> after ≥3 distinct configs of its key hyperparameter.

## Families

| Family | Variants tried | Best result | Status | Notes |
|--------|---------------|-------------|--------|-------|
| heuristic-greedy-nearest-target | shipped Nearest Planet Sniper | live Score=303 | retired (anchor) | Calibration probe; aim-at-current dies to orbit drift + sun. |
| heuristic-orbit-aware-greedy | v1 (orbit lead + RNG tiebreak) | live Score=568 | retired | Closes A.6; +265 vs baseline. |
| heuristic-orbit-aware-greedy-with-shared-mechanism-layer | v1.1 (+ arrival_size) | live Score=566 | retired | Strategy/Mechanism split; production-aware sizing. |
| simple-greedy-target-selection-variants | nearest, production, roi, weakest, enemy_first | v1.2/roi live Score=1006.9 | retired | ROI dominates panel; submitted as #52518060. |
| heuristic-physics-upgrade | 5-iter aim + safe-intercept + sun-safe + path_clears + oob_guard | rolled into v2+ | retired | Foundational physics — kept in lib/aim.py, lib/trajectory.py. |
| heuristic-worldmodel-aware | v2 (roi + WorldModel.owner_at dedup) | live Score=966.1 | retired | Block A physics + Block D worldmodel dedup; submitted #52532938. |
| mission-framework-snipe-only | v3.0 (Mission dataclass + propose_snipe + settle_plan) | bit-for-bit parity with v2 | retired | Block E MVP refactor; foundation for all later agents. |
| env-clone-forward-sim-scorer | Sim<K> via env.clone() + step | AUC=0.952 at K=50 | retired (substrate) | Phase 2 probe; superseded by fast_sim (183× faster). |
| lookahead-drop-one-candidates | v3_lookahead (env.clone + drop-one + Sim<K=10>) | 32-seed 50/50 vs v2 | retired | Framework worked; drop-one candidate set too narrow on its own. |
| full-trajectory-predict-fleet-fate | lib/trajectory.py ray-cast | capture probe 97.2% | live (substrate) | Replaces endpoint-only guards. |
| cost-aware-roi-additive-denominator | (production × time_to_hold) / (ships + distance + 1) | shipped in v3_snipe | live (substrate) | Additive cost avoids 1-ship attraction. |
| comet-lifetime-correction | time_to_hold capped at len(path) - path_index | shipped in v3_snipe | live (substrate) | Lifetime-aware comet scoring. |
| mission-framework-snipe-plus-reinforce | v3_snipe (snipe + reinforce + same-turn ledger + ray-cast guards) | live Score=1005.7 | retired (production baseline) | First μ>1000 agent; submitted #52544634. |
| aggressive-snipe-ship-sizing | v3.5.1 (lib/missions/snipe aggressive=True) | live Score=945.6 | **falsified** | Local 68.8% Wilson lo 56.6% passed; live regressed −60 vs v3_snipe. Local A/B didn't generalise. |
| sigma-equivariance-patches | planner score-round + sym_hypot + tie-break | live Score=1041.4 | live (substrate) | 16/16 v3-vs-v3 self-play draws (provable cannot-lose at v3-class). In lib/planner.py + lib/orbit.py; affects all later agents. |
| v7-maximin-search | v7_minimax (N=2 our × M=2 opp × Sim<K=3>) | live Score=1040.8 | retired | First true game-theory iteration; structurally weaker than drop-one. |
| v4-receding-horizon-mission-portfolio | v4_planner (5 portfolios × Sim<K=6-10> + value head) | live Score=1038.6 | retired | "Shoots later" pathology — noop in portfolios prefers wait when target eta>K. |
| **fast-brain (Snapshot + opp_model + v7_search)** | lib/fast_sim.py (183×) + lib/opp_model.py Tier-0/1 + lib/v7_search.py | substrate for v7_0_drop_one | **live (substrate)** | Bypasses Environment overhead; bit-exact via direct interpreter() call. Foundation for all forward-sim agents going forward. Permanent reference: knowledge-base/concepts/lookahead-simulator-architecture.md. |
| **v7-drop-one-fast-brain** | v7_0_drop_one (drop-one chooser + K=10 forward sim + Tier-1 opp + 4P→v3.5.1 fallback) | **live Score=1094.9 — TEAM PEAK** | **live (anchor)** | Submitted #52588156; +56 over v4_planner, +89 over v3_snipe. Local: 79.2% vs v7_minimax Wlo 59.5%, 75.0% vs v4_planner Wlo 55.1%. |
| v7-sweep-variants (4) | v7_1_target_swap, v7_2_ship_sweep, v7_3_archetype, v7_4_hungarian | all FAIL Wilson 55% | dead, pruned | Local A/Bs ranged 45.8–58.3%; none cleared gate. |
| v7-iteration-variants (6) | v7_1_minimax through v7_6_no_recapture, v7_combined | all FAIL or PARITY | dead, pruned | σ-equiv + symmetric scoring + maximin + recapture + 4P-aware; v7.6 bisect showed σ-equiv layer regresses drop-one architecture by −54pp. |
| v8-psro-self-play-pool | v8_minimal, v8_fastbrain, v8_psro_meta | DEGENERATE Nash (pure v7) | parked, pruned | Pool needs anti-v7 policies before PSRO converges. |
| v9-super-version-variants (4) | v9_inflight, v9_k15, v9_combined, v9_opening | none clear Wilson 55% | dead, pruned | Tried inflight-value head, K=15 deeper, combined stacks, opening-conditional bonuses — all variance, no signal. |
| v10-evaluate-value-head | v10_evaluate (drop-one + K=10 + evaluate_value head, no σ-equiv) | 62.5% Wilson lo 42.7% FAIL | dead, pruned | Missed by 2.3pp; value head alone doesn't beat v7_0's pure ship-delta. |
| meta-strategy-framework (infra only) | replay capture, fingerprint (15 features, FEATURE_VERSION=1), manifold diagnostic | 5-class RF 80.5% at K=100 — gate (90%) not cleared | paused | ROI-family is one basin with 12-17% mutual confusion; broad-class routing partially works. |
| shot-validator-labeling-pipeline | scripts/label_shot_outcomes.py + data/shot_validator/{schema.json, README.md} | 37k labeled examples (24-dim, 0/1) | parked | MLP training deferred. Tier-2 opp_model slot reserved. |

## Family taxonomy (seed list — expand as tried)

- **Heuristic** — hand-coded rules over observation features.
- **Search** — MCTS / minimax / A* over short horizons.
- **Imitation learning (IL)** — supervised on top-LB replays.
- **Reinforcement learning (RL)** — self-play, opponent-pool training,
  PPO / A2C / IMPALA / etc.
- **Hybrid** — heuristic policy with RL value head, or IL warm-start
  followed by RL fine-tuning.
- **Ensemble** — vote / stack of agent classes per game-state segment.
