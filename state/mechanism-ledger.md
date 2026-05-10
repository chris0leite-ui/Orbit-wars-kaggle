# state/mechanism-ledger.md — every agent-design family tried

> One row per family × variant. Rule 21: a family is only "dead"
> after ≥3 distinct configs of its key hyperparameter.

## Families

| Family | Variants tried | Best result | Status | Notes |
|--------|---------------|-------------|--------|-------|
| heuristic-greedy-nearest-target | shipped Nearest Planet Sniper | live μ=303 | live (anchor) | Calibration probe; aim-at-current dies to orbit drift + sun. |
| heuristic-orbit-aware-greedy | v1 (orbit lead + RNG tiebreak) | live μ=508 | live | Closes A.6; +205 μ vs baseline. |
| heuristic-orbit-aware-greedy-with-shared-mechanism-layer | v1.1 (+ arrival_size) | local 85% vs v1 | submitted, μ pending | Strategy/Mechanism split; production-aware sizing. |
| simple-greedy-target-selection-variants | nearest, production, roi, weakest, enemy_first (5 score functions, shared DEFAULT_MECHANISMS) | `roi` 97% panel-winrate, 100% (16/16) vs v1_orbitfix at 8 seeds | local-only, awaiting 32-seed confirmation | Plan: read-the-handover-next-imperative-whisper.md. Run: `python -m scripts.strategy_panel`. JSON: audit/tournaments/20260510T123059Z.json. Promotes to live if 32-seed confirms ≥60% beat AND v1.1 μ has settled (rolling-last-2 economy). |

## Family taxonomy (seed list — expand as tried)

- **Heuristic** — hand-coded rules over observation features.
- **Search** — MCTS / minimax / A* over short horizons.
- **Imitation learning (IL)** — supervised on top-LB replays.
- **Reinforcement learning (RL)** — self-play, opponent-pool training,
  PPO / A2C / IMPALA / etc.
- **Hybrid** — heuristic policy with RL value head, or IL warm-start
  followed by RL fine-tuning.
- **Ensemble** — vote / stack of agent classes per game-state segment.
