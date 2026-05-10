# state/mechanism-ledger.md — every agent-design family tried

> One row per family × variant. Rule 21: a family is only "dead"
> after ≥3 distinct configs of its key hyperparameter.

## Families

| Family | Variants tried | Best result | Status | Notes |
|--------|---------------|-------------|--------|-------|
| heuristic-greedy-nearest-target | shipped Nearest Planet Sniper | live μ=303 | live (anchor) | Calibration probe; aim-at-current dies to orbit drift + sun. |
| heuristic-orbit-aware-greedy | v1 (orbit lead + RNG tiebreak) | live μ=508 | live | Closes A.6; +205 μ vs baseline. |
| heuristic-orbit-aware-greedy-with-shared-mechanism-layer | v1.1 (+ arrival_size) | local 85% vs v1 | submitted, μ pending | Strategy/Mechanism split; production-aware sizing. |
| simple-greedy-target-selection-variants | nearest, production, roi, weakest, enemy_first (5 score functions, shared DEFAULT_MECHANISMS) | `roi` 97.1% panel WR, **100% (64/64) vs v1_orbitfix** at 32 seeds | local-only, awaiting v1.1 μ settle | 8-seed confirmed at 32: roi mean WR 96.9% → 97.1%; production 75% → 67.7%; nearest 56% → 56% (tied with v1 by construction). audit/tournaments/20260510T140907Z.json. Submission gate (i) clears (Wilson lo on 64/64 ≈ 0.94); blocked only on v1.1 μ. |
| meta-strategy-framework (Phase 1 infra only) | replay capture (`scripts/tournament.py::_build_replay`), behavioural fingerprint (`lib/fingerprint.py`, 15 features, FEATURE_VERSION=1), manifold diagnostic (`scripts/manifold_check.py`) | 5-class RF 80.5% at K=100 — gate (90%) ❌ NOT cleared | wip — Phase 2 paused on hypothesis-board verdict | weakest (89.7%) and enemy_first (83.4%) sit in own basins; ROI-family (nearest/production/roi) is one basin with 12-17% mutual confusion. Recommended next: H-coarsen-labels (merge ROI-family) or H-richer-fingerprint (add distribution-shape + temporal-split features). Verdict audit: audit/2026-05-10-phase1-manifold-verdict.md. |

## Family taxonomy (seed list — expand as tried)

- **Heuristic** — hand-coded rules over observation features.
- **Search** — MCTS / minimax / A* over short horizons.
- **Imitation learning (IL)** — supervised on top-LB replays.
- **Reinforcement learning (RL)** — self-play, opponent-pool training,
  PPO / A2C / IMPALA / etc.
- **Hybrid** — heuristic policy with RL value head, or IL warm-start
  followed by RL fine-tuning.
- **Ensemble** — vote / stack of agent classes per game-state segment.
