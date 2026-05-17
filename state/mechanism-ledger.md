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
| v7_1-h11-h15-opening-comet-reject | v7_1_open_drop_comets (drop-one + H11 opening + H15 comet reject) | Scalar 32-game 53.1% Wilson lo 36.4%; JAX 64-game 35.9% Wilson lo 25.3% FAIL | dead, pruned | H11 wired into _build_incumbent_intents; H15 in snipe.py. 2026-05-14. |
| v7_2-depth2-maximin | v7_2_depth2 (depth-2 maximin over v3.5.1 drop-ones, K_tail=4) | 31.3% Wilson lo 18.0% FAIL | dead, pruned | choose_depth2 in v7_search.py. Forward-sim biased toward passive play. 2026-05-14. |
| v7_3-minregret-archetypes | v7_3_minregret (min-regret over 5 hand-crafted opp archetypes) | 28.1% Wilson lo 15.6% FAIL | dead, pruned | choose_archetype_minregret; opp archetypes module reusable. 2026-05-14. |
| v7_4-composite-capture-value | v7_4_capture_value (drop-one + composite_capture_value head: capture reward + waste penalty) | 40.6% Wilson lo 25.5% FAIL | dead, pruned | composite_capture_value in lib/value_heads.py; reusable. Best of v7_X session. 2026-05-14. |
| v7_5-add-one-widening | v7_5_drop_add_capture (drop-or-add-one + capture value) | 37.5% Wilson lo 22.9% FAIL | dead, pruned | ADD-one enumerator; reusable. 2026-05-14. |
| v7_6-split-source-multilaunch | v7_6_split_source (drop-or-split + capture value; same src appears twice in action) | 40.6% Wilson lo 25.5% FAIL | dead, pruned | Split-source enumerator; reusable. Multi-launch from one source — env supports, never used before. 2026-05-14. |
| v7_7-enemy-multiplier | v7_7_enemy_mult (drop-one + capture value + snipe ENEMY_MULTIPLIER=1.3) | 28.1% Wilson lo 15.6% FAIL | dead, pruned | H10 single-coef. Top-10's enemy-bias doesn't transfer as a blanket multiplier — likely emerges from BETTER targeting, not a priority shift. 2026-05-14. |
| meta-strategy-framework (infra only) | replay capture, fingerprint (15 features, FEATURE_VERSION=1), manifold diagnostic | 5-class RF 80.5% at K=100 — gate (90%) not cleared | paused | ROI-family is one basin with 12-17% mutual confusion; broad-class routing partially works. |
| shot-validator-labeling-pipeline | scripts/label_shot_outcomes.py + data/shot_validator/{schema.json, README.md} | 37k labeled examples (24-dim, 0/1) | parked | MLP training deferred. Tier-2 opp_model slot reserved. |
| **v21-compound-proposer-augmentation** | v21_compound (sun-safe pre-filter + rotation_alignment bonus + chain_bonus + MissionBook TTL persistence on top of v20 dogpile) | Rule 38 sun-bug VERIFIED: v20 sends 6 fleets/8 games to sun + 2 OOB; v21 sends 0/0. h2h vs v20 n=32: 46.9% Wlo=0.309 INCONCLUSIVE (within noise). | local-only | Three NEW axes vs all prior chooser/value-function exhaustion (Rule 37): proposer-time filtering, geometric rotation awareness, mission persistence. `agents/v21_compound/main.py` + `lib/{compound,mission_book}.py` + `lib/geo/rotation.py`. 2026-05-16. |
| v21-ablations-single-axis (a/b/c/d) | v21a (sun-only), v21b (rotation-only), v21c (chain-only), v21d (book-only) | A/B vs v20 not completed this session | local-only | Single-axis ablations to localize which compound component (if any) lifts. Built but not benchmarked. lib.compound.compound_bonus accepts use_rotation/use_chain/use_carry flags. 2026-05-17. |
| **v22-sun-on-v15-base** | v22_sun_on_v15 (v15 chooser + lib.compound.fleet_path_safe pre-filter; nothing else) | A/B vs v15 running | local-only (this session) | Strategy pivot 2026-05-17: live ladder shows v15=1115.5 vs v20=1095.5; v20 lost 20μ vs v15 live despite local 65.6%. Build defensive-only addition on the live winner. By construction v22 ≡ v15 OR marginally better (sun pre-filter never adds candidates). Bundle: submissions/v22_sun_on_v15.py (332 KB). Highest-EV submission candidate. 2026-05-17. |

## Family taxonomy (seed list — expand as tried)

- **Heuristic** — hand-coded rules over observation features.
- **Search** — MCTS / minimax / A* over short horizons.
- **Imitation learning (IL)** — supervised on top-LB replays.
- **Reinforcement learning (RL)** — self-play, opponent-pool training,
  PPO / A2C / IMPALA / etc.
- **Hybrid** — heuristic policy with RL value head, or IL warm-start
  followed by RL fine-tuning.
- **Ensemble** — vote / stack of agent classes per game-state segment.
