# state/current.md — current submitted agent + tournament rank

> Day-1 agent: leave this empty until the first kernel push. Once
> populated it is updated at every wrap-up (WRAPUP.md step 3).

```yaml
date: 2026-05-12
days_to_deadline: 42                     # 2026-06-23 23:59 UTC minus today
current_submitted_agent: v3_4            # unchanged from prior session — no submissions today
last_kernel_push: 2026-05-11 21:19:13 UTC
last_submission_id: 52556866
last_submission_status: PENDING          # v3.4 = v3.2 + 4P spoiler. Bundle sha256:410b3c2ee370f943.
last_submission_file: submissions/v3_snipe.py
last_submission_message: |
  v3_snipe: Block E missions (snipe + reinforce) + cost-aware ROI +
  comet-lifetime + same-turn ledger + full-trajectory ray-cast guards.
tournament_rank_today: v3_4=PENDING (no rank movement today; offline-only iteration)
our_best_rank: μ=PENDING (#52556866, v3_4)
lb_top10_cliff: 1447.6                   # ShunkiKyoya, 2026-05-11
submissions_used_today: 0                # offline iteration only; HELD slot per Rule 12 caveat (rolling-last-2 eviction risk)
submissions_used_total: 7
plateau_days: 1                          # +1: no live μ progress today
saturation_count: 0
session_log:
  - 2026-05-11/12 — optimize-ship-strategy-tDPXx (this session, ~12h overnight): (a) Phase-0 idle-source decomposition shipped — lib/{planner,intent}.py gain opt-in `reasons` out-param + scripts/episode_postmortem.py classifies idle sources into 5 buckets (NO_PROPOSALS / GATE_REJECTED / LEDGER_LOSS / MECHANISM_DROP / RESERVE_HELD), validated on 8 self-play games (audit/2026-05-11-idle-breakdown-v3-snipe-phase0.md); (b) v3.5 attempt — airtime penalty (AIRTIME_PENALTY_WEIGHT=1.0) + endgame neutral burn (ENDGAME_NEUTRAL_BONUS=1.5) in lib/missions/snipe.py — REGRESSED to 43.8% Wilson [32.3%, 55.9%] at 32-seed pair-level vs v3.4 baseline (audit/2026-05-11-v3.5-airtime-and-endgame-burn.md); (c) built scripts/ab_variants.py — bundle-isolated A/B harness (each variant gets its own .py copy of lib via scripts/bundle_agent.py, no module-state leak); ran 4 rounds (sanity, 32-seed v3.5, 8-seed 5-variant ablation, 64-seed confirmation); AIRTIME=0.5+ENDGAME=1.5 best at 32-seed (54.7%) but converged to 52.3% Wilson [43.7%, 60.8%] at 64-seed = TIE; (d) PROPOSER_AFFORDABILITY_FILTER added — 64-seed verdict 21.1% Wilson [14.9%, 29.0%] = BROKEN (deploys ships from idle defensive garrisons to small low-EV captures); (e) all four scoring/filter knobs REVERTED to identity defaults; constants kept in code for future ablation (audit/2026-05-11-v3.5-airtime-and-endgame-burn.md final verdict); (f) gang_up_size mechanism added (lib/mechanism.py — runs before validate, anchors on slowest source, throttles faster sources DOWN by reducing ship count, allocates proportional shares); 64-seed 4-variant A/B verdict: 43.8% / 45.3% / 43.8% = REGRESSED (audit/2026-05-12-gang-up-v1.md); Phase-0 falsifier evidence — substrate bug: arrival_size re-bumps throttled shares via `intent.ships = max(intent.ships, needed)`, undoing gang-up's work (validate drops -39%, arrival_size drops +31%, net total -2%); (g) reverted GANG_UP_ENABLED=0; (h) 7-step OR problem-solving framework applied (/root/.claude/plans/you-are-a-mathematician-resilient-papert.md) — meta-lesson: Phase-0 bucket-reduction was a misleading proxy (correlated in head, not in reality); next iteration should target retrospective fleet value, not bucket reduction.
  - 2026-05-11 PM — analyze-submission-logs-dFHeS (prior session): (a)-(i) per archive.
mechanism_families_explored:
  - heuristic-greedy-nearest-target
  - heuristic-orbit-aware-greedy
  - heuristic-orbit-aware-greedy-with-shared-mechanism-layer
  - simple-greedy-target-selection-variants
  - meta-strategy-framework-infra-only
  - heuristic-physics-upgrade
  - heuristic-worldmodel-aware
  - mission-framework-snipe-only
  - env-clone-forward-sim-scorer
  - lookahead-drop-one-candidates
  - full-trajectory-predict-fleet-fate
  - cost-aware-roi-additive-denominator
  - comet-lifetime-correction
  - mission-framework-snipe-plus-reinforce
  - phase-0-idle-source-decomposition          # diagnostic: 5-bucket classifier in settle_plan + realize via opt-in `reasons` param. ~96% of idle = MECHANISM_DROP from `intent.ships > src.ships` in validate/arrival_size. Phase-0 bucket-reduction proved a misleading proxy: 4 fixes targeted it, all tied/regressed at 64-seed.
  - airtime-penalty-denominator                # lib/missions/snipe.py AIRTIME_PENALTY_WEIGHT term. AIRTIME=1.0 regressed -12.5pp; AIRTIME=0.5 tied (52.3% Wilson [43.7, 60.8]). Default reverted to 0.0.
  - endgame-neutral-burn-priority              # lib/missions/snipe.py ENDGAME_NEUTRAL_BONUS at step>=470. Standalone produces stalemate (40/64 draws). Default reverted to 1.0.
  - proposer-affordability-filter              # lib/missions/snipe.py PROPOSER_AFFORDABILITY_FILTER. 64-seed verdict 21.1% [14.9, 29.0]. BROKEN: stripping unaffordable big-target attempts replaces high-variance captures with low-EV small ones; defensive garrison value lost. Default reverted to 0.
  - bundle-isolated-ab-harness                 # scripts/ab_variants.py — per-variant bundle, no module-state leak, auto-discovers constant source file. Reused across 4 sweeps tonight.
  - gang-up-shared-sizing-mechanism            # lib/mechanism.py gang_up_size: anchor-on-slowest, throttle faster, allocate proportional. v1 regressed (43.8% / 45.3% / 43.8%). Substrate bug: arrival_size re-inflates shares; needs sibling-aware sizing. Default reverted to 0.
gate_status: cleared                      # 224/224 non-bootstrap tests green; bundle/idle-trace flows validated
headroom_to_top5pct: deprecated
headroom_to_top10_prize: +392 μ           # unchanged today
headroom_to_roman_public: +250 μ          # unchanged today
```
