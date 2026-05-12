# state/current.md — current submitted agent + tournament rank

> Day-1 agent: leave this empty until the first kernel push. Once
> populated it is updated at every wrap-up (WRAPUP.md step 3).

```yaml
date: 2026-05-12
days_to_deadline: 42                     # 2026-06-23 23:59 UTC minus today
current_submitted_agent: v7_minimax      # rolling-last-2 NOW: [v3.5.1 (52565976, μ=988.8), v7_minimax (52568317, PENDING)]. σ-equiv-v1 (52565034, μ=1041.8 — our peak before this push) evicted as oldest.
last_kernel_push: 2026-05-12 06:50:06 UTC
last_submission_id: 52568317
last_submission_status: PENDING          # v7_minimax = v3+σ-equiv base + K=3 maximin overlay (lib/lookahead.score_joint_action_symmetric, 2× cost to cancel env P1-bias). Bundle sha256:1393d32b1f4e691d. Headline: first iteration this session to beat BOTH v3.4 (75% W/D local) AND precision_v3 (75% W/D local). Real game theory (von Neumann minimax at action level). Audit: 2026-05-12-v7-minimax-submission.md.
last_submission_file: submissions/v7_minimax.py  # 81.8 KB bundle of agents/v7_minimax/main.py + lib/{geometry (with sym_hypot),fleet,orbit,aim,combat,world_model,intent,trajectory,mechanism,mission,missions/snipe (sym_hypot),missions/reinforce (sym_hypot),planner (σ-equiv tie-break + SCORE_ROUND=6),lookahead (score_joint_action + score_joint_action_symmetric)}. Strategy = v3-with-σ-equiv as base + K=3 maximin overlay choosing among (incumbent, drop-smallest) × (opp = v3-from-opp-POV, drop-smallest opp).
last_submission_message: |
  v7_minimax: K-step maximin agent (real game theory). v3+σ-equiv base
  + maximin overlay: enumerate N=2 our candidates × M=2 opp models ×
  Sim<K=3> with v3 as rollout policy, pick action whose worst-case
  payoff is highest. score_joint_action_symmetric averages over
  seat-flipped rollouts to cancel env's P1-favoring tie-break.
  Local 4-seed both-sides (8 games each): 75% W/D vs v3.4; 75% W/D
  vs precision_v3. First iteration to beat both v3 AND precision on
  local probes. Bundle sha256:1393d32b1f4e691d.
tournament_rank_today: v3.5.1=988.8 (in rolling-last-2), v7_minimax=PENDING. σ-equiv-v1 (1041.8 — our team peak), v3_4 (995.4), precision_v3 (1009.0), v3_snipe (1005.7) all evicted.
our_best_rank: μ=PENDING (#52568317). Predicted range: 1040-1090 — v7 = σ-equiv (1041.8 measured) + maximin overlay (75% W/D vs σ-equiv-equivalent v3.4 locally). Expected: ≥ σ-equiv μ, with small upside from maximin overrides on ~5% of turns.
lb_top10_cliff: 1447.6                   # ShunkiKyoya, 2026-05-11.
submissions_used_today: 2                # σ-equiv-v1 (04:39 UTC) + v7_minimax (06:50 UTC), both this branch. Plus parallel-branch pushes.
submissions_used_total: 9
plateau_days: 0
saturation_count: 0
session_log:
  - 2026-05-11 PM — analyze-submission-logs-dFHeS: (a) live data pull for v3_snipe (52544634, μ=1055.5 +90.2 over v2) + v2 (52532938, μ=965.3); (b) replay-driven instrumented postmortem (scripts/episode_postmortem.py) — fleet outcome attribution + drop telemetry; v3 bounces enemy 2x as often as v2 (14.7% vs 7.6%); (c) precision-physics A/B (parallel branch claude/precision-physics-engine-ymJkA) — v3_snipe wins 10/16, precision p95=1444ms > 1s actTimeout — NOT submittable as-is; (d) audit/2026-05-11-v3-snipe-critical-review.md; (e) parity gap closed: 53% match was instrumentation bugs (off-by-one + missing obs.step backfill), now 100% on all 34 v3_snipe replays + 998-turn self-play; (f) permanent gates: tests/test_replay_parity.py, scripts/bundle_agent.py post-bundle parity + sha256 hash; (g) v3.2 lib changes — arrival_size now consults WorldModel.ships_at for adversary stacking (lib/mechanism.py), DEFAULT_HORIZON 110→250 (lib/world_model.py), realize() takes optional model kwarg (lib/intent.py); (h) 32-seed 2P A/B v3.2 vs v3_snipe_frozen = 57.8% (37/64) Wilson [45.6, 69.1] (audit/tournaments/20260511T152648Z.json); 16-seed 4P FFA panel v3.2 93.8% vs frozen 90.6% (audit/tournaments/ffa-panel-20260511T185817Z.json); (i) v3.2 not yet submitted — PI to authorize a slot per Rule 1.
  - 2026-05-11 — bootstrap-agentic-systems-lqnm6 (prior): bootstrap infra, Block E mission framework MVP, lookahead probes (env.clone() Sim<K=50> AUC=0.952), v3.1 drop-one parity, live fleet-loss fix (predict_fleet_fate), ROI cost-awareness, four remaining ROI-doc shortcomings (comet lifetime / reinforce / arrival-ledger / mission classification), v3_snipe submitted as #52544634. Total: 14 commits on lqnm6 branch.
  - 2026-05-10 PM — competition-strategy-brainstorm-ZK6XT (prior): strategic-direction plan + Block A physics + Blocks C+D arrival-ledger + v2 strategy. v1.2/roi μ settled at 978.7. Top-10 cliff = 1460. (Older entries archived to audit/archive-2026-05-10-handover-prior-pm-sessions.md.)
mechanism_families_explored:
  - heuristic-greedy-nearest-target       # comp-shipped Nearest Planet Sniper; calibration anchor only
  - heuristic-orbit-aware-greedy          # v1: lead-prediction for orbiting non-comet targets + tie-break randomisation (closes A.6)
  - heuristic-orbit-aware-greedy-with-shared-mechanism-layer  # v1.1: above + Strategy/Mechanism split + arrival_size (production-aware sizing)
  - simple-greedy-target-selection-variants  # nearest, production, roi, weakest, enemy_first — ROI dominates panel at 32 seeds; v1.2 = roi submitted as #52518060
  - meta-strategy-framework-infra-only       # replay capture + behavioural fingerprint + manifold diagnostic; gate not cleared with v1 features
  - heuristic-physics-upgrade               # v1.3: 5-iter aim_with_prediction + search_safe_intercept fallback + sun-safe arrival-aware + path-clears-other-planets + oob_guard
  - heuristic-worldmodel-aware              # v2: roi target selection + WorldModel.owner_at predictive dedup; submitted as #52532938
  - mission-framework-snipe-only            # v3.0 (Block E MVP): Mission dataclass + propose_snipe_missions + settle_plan; bit-for-bit parity with v2 (32/32 draws at step 500)
  - env-clone-forward-sim-scorer            # Sim<K> scoring head via env.clone() + step; AUC 0.952 at K=50 from probe step 50 ≈ perfect oracle (audit/2026-05-11-lookahead-phase2-forward-sim.md)
  - lookahead-drop-one-candidates           # v3.1 v3_lookahead: env.clone() forward sim over drop-one candidate set; 32-seed 50/50 vs v2 — framework works, drop-one too narrow
  - full-trajectory-predict-fleet-fate      # lib/trajectory.py replaces endpoint-only guards with full-flight ray-cast; capture probe reached 77.2% → 97.2% across the 3 fix waves of 2026-05-11
  - cost-aware-roi-additive-denominator     # score = (production × time_to_hold) / (ships_to_send + distance + 1). Additive cost avoids pure-value/cost over-correction toward 1-ship targets
  - comet-lifetime-correction                # comet_remaining_lifetime helper; time_to_hold caps at len(path) - path_index for comet targets
  - mission-framework-snipe-plus-reinforce  # v3.1 v3_snipe: snipe + reinforce mission classes through settle_plan (with same-turn arrival ledger). 32-seed 2P 57.8% vs v2 (Wilson lo 45.6%); 16-seed 4P FFA parity. Submitted as #52544634.
gate_status: cleared                      # 228/228 tests green; bundle E.2 self-play 0/10 crashes; bundle-vs-unbundled parity 10/10; 32-seed 2P + 16-seed 4P panels run
headroom_to_top5pct: deprecated            # no longer the binding target — top-10 prize cliff is
headroom_to_top10_prize: +392 μ           # top-10 cliff at 1447.6 (ShunkiKyoya, 2026-05-11). v3_snipe at 1055.5 → +392μ.
headroom_to_roman_public: +250 μ          # Roman published 1224; we are 250 μ below his ceiling. Block E missions narrow this; v3.1 candidate enumerator for lookahead is the next ceiling-raiser.
```
