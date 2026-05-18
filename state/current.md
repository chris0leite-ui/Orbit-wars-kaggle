# state/current.md — current submitted agent + tournament rank

> **Score values are intentionally NOT recorded here.** They drift as the
> rolling μ settles and as the leaderboard moves. ALWAYS query Kaggle
> directly at session start (Rule 32):
>
>     export KAGGLE_USERNAME="$KaggleUserName" KAGGLE_KEY="$KaggleAPIToke" \
>            KAGGLE_API_TOKEN="$KaggleAPIToke"
>     kaggle competitions submissions orbit-wars
>
> Updated 2026-05-17 by `claude/kaggle-baseline-strategy-lO4mm`
> (clean modular re-baseline of v15).

```yaml
date: 2026-05-18
deadline: 2026-06-23 23:59 UTC
days_to_deadline: 36

# Most-recent submission (trajectory chooser v4 + wait_N + wallclock budget).
# Sets BASELINE_CHOOSER=trajectory + BASELINE_VALUE_HEAD=hybrid via setdefault
# in agents/baseline/main.py. SETTLED at mu=1271.8 (far above local A/B
# prediction of ~1140-1180). Spatial-leaf A/B 2026-05-18 was net-negative
# (2P 40.6%, 4P 9.4%, wallclock max 2541ms) — NO new submission.
last_submission_id: 52754310
last_submission_status: COMPLETE
last_submission_file: submissions/baseline.py
last_submission_agent: trajectory_chooser_v4_waitN_baseline
last_kernel_push: 2026-05-17 22:06:07 UTC
prior_error_submission_id: 52744234  # 5/17 earlier — bundler fix in commit 4094aa1
current_submitted_agent: trajectory_chooser_v4_waitN_baseline (5/17 evening; trajectory chooser default-on)

# Rolling-last-2 (Kaggle auto-keeps these two for final evaluation; the
# third push auto-evicts the previous oldest). v20 (1082.4) evicted by
# the 52754310 push. Pair becomes [composite_a2 52744856, trajectory 52754310].
rolling_last_2:
  - {agent: trajectory_chooser_v4_waitN_baseline, sub_id: 52754310, submitted: 2026-05-17T22:06Z, status: PENDING}
  - {agent: composite_a2_hybrid_baseline_rebundle, sub_id: 52744856, submitted: 2026-05-17T14:17Z, status: COMPLETE, mu_at_submit_time: 1158.6}

# Team peak as of 2026-05-17: v15_banded (the multi-wait-grid + banded
# (src, tgt, wait_band) dedup line). v15 source lives in git history at
# f315dc7:agents/v15/main.py — NOT in the working tree (the "nuke
# historical strategy code" reset removed it; we kept the audits).
# A clean modular re-implementation lives at agents/baseline/.
team_peak_agent: v15_banded

# Calibration WARNING (still active 5/17 — applies to the next submission):
# Multiple recent submissions over-predicted live. Local-vs-live mapping
# has been roughly -20 to -30 pp on every recent submission. Use a
# 3-opponent panel (`fast.py eval --vs-panel`) + h2h vs the current
# rolling agent (not just a fixed baseline) before any new push.

submissions_used_today: 3     # 5/17 — composite+A2 hybrid x2 (52744234 ERROR, 52744856 OK), trajectory v4+waitN (52754310)
submissions_used_total: 32    # see ladder list below; refresh via Kaggle CLI
plateau_days: 0
saturation_count: 0

# Live ladder — read from `kaggle competitions submissions orbit-wars`,
# NOT from this file. Submission IDs are stable; scores are not.
# Most recent ladder entries by submission id:
#   52754310  baseline.py (trajectory v4 + wait_N + wallclock) 2026-05-17 22:06 PENDING ← NEW
#   52744856  baseline.py (composite+A2 hybrid, re-bundle) 2026-05-17 14:17 COMPLETE μ≈1158 settling
#   52744234  baseline.py (composite+A2 hybrid, ERROR) 2026-05-17 13:57 ← failed: `from agents.baseline import` not inlined
#   52721807  v20.py            2026-05-16 21:57  COMPLETE
#   52710995  v15.py            2026-05-16 13:43  COMPLETE  ← team peak
#   52704189  v8_scavenge (v13) 2026-05-16 09:07  COMPLETE
#   52699232  v8_scavenge (v12) 2026-05-16 05:37  COMPLETE
#   52687411  v8_scavenge (v9)  2026-05-15 17:41  COMPLETE  ← highest-ever score; evicted from rolling
#   52684059  v8_scavenge (v8)  2026-05-15 15:05  COMPLETE
#   52678866  iter.py  (iter_v2) 2026-05-15 11:34 COMPLETE
#   52661990  iter.py  (iter_v1) 2026-05-14 21:48 COMPLETE
#   52643676  geo.py             2026-05-14 09:10 COMPLETE
#   52630118  v7_pv.py           2026-05-13 23:31 COMPLETE
#   52607699  v7_0_drop_one      2026-05-13 08:33 COMPLETE
#   52588156  v7_0_drop_one      2026-05-12 17:36 COMPLETE
#   52579863  v4_planner         2026-05-12 14:25 COMPLETE
#   52568317  v7_minimax         2026-05-12 06:50 COMPLETE
#   52565976  v3.5.1             2026-05-12 05:20 COMPLETE
#   52565034  v3_snipe (sigma)   2026-05-12 04:39 COMPLETE
#   52556866  v3_4               2026-05-11 21:19 COMPLETE
#   52552139  precision_v3       2026-05-11 17:00 COMPLETE
#   52544634  v3_snipe           2026-05-11 12:16 COMPLETE
#   52532938  v2                 2026-05-11 04:04 COMPLETE
#   52518060  roi (v1.2)         2026-05-10 14:59 COMPLETE
#   52509319  v1_orbitfix (1.1)  2026-05-10 09:28 COMPLETE
#   52507539  v1_orbitfix        2026-05-10 08:11 COMPLETE
#   52497828  day1_baseline      2026-05-10 00:09 COMPLETE

session_log:
  - 2026-05-17 PM — audit-workflow-performance-btjeK.
    Diagnostic + observe-loop foundations + composite head wired + A2 merged
    + submission bundled. Workflow fixes: kaggle-CLI shim (`~/.local/bin/kaggle`
    installed by session-start hook to persist KAGGLE_API_TOKEN across Bash
    calls); `fast.py eval --vs-panel` REFUSES (exit 2) unless `--require-h2h
    <champion>` is set; WRAPUP step 4c enforces Rule-36 flags/questions filing
    check. Pivot #1 (replay-mine): `scripts/replay_mine.py` walks live-episode
    replays and classifies fleets into PI-facing buckets. Surprise finding:
    PI's "vanished_in_space = comets" hypothesis was falsified (0.1% / 12 of
    9507 fleets); the 838 vanishes were misclassified planet hits because
    `attribute_fleets:290` used static distance from fleet-old to planet-NEW.
    Fix: swept-pair against every planet via `lib.game.interpreter.swept_pair_hit`
    — v15's real waste is ~17%, not 24%. Pivot #2 (composite head): wired
    `composite_capture_value` opt-in via `BASELINE_VALUE_HEAD=composite`, then
    panel A/B at n=32 cleared every opponent including the team peak
    (v9_scavenge 30/32 = 93.8% Wlo=0.799, v15 24/32 = 75% Wlo=0.579 PASS;
    n=64 retest 40/64 = 62.5% Wlo=0.503 INCONCLUSIVE — best estimate of true
    2P winrate ~63-67%, Wlo right at the 0.55 gate). Followups: pre-bail
    headroom + adaptive WorldModel horizon (`#1+#2 timing fixes`). Merged A2
    from claude/kaggle-baseline-strategy-lO4mm (`favor_hybrid` dispatcher:
    composite-in-2P + A2-favor-in-4P; A2 = 1.5× weakest-opp bias + +55
    elimination bonus, sourced from public notebook romantamrazov LB μ=1224).
    Submission bundled at `submissions/baseline.py` (286 KB, parity OK over
    712 turns; uses hybrid by default via `os.environ.setdefault`). Tests:
    53+ green across baseline value/chooser/proposer + new postmortem-comet
    + dispatcher + wallclock variants (favor + hybrid). 4P FFA panel running
    at session-end. NOT SUBMITTED — Rule 1, PI sign-off required.
  - 2026-05-17 — kaggle-baseline-strategy-lO4mm.
    Shipped agents/baseline/{main,proposer,chooser,value}.py — clean
    modular re-implementation of v15 (live champion, 5/16 push), 577 LOC
    across 4 files of ≤262 LOC each. Uses lib/fast_sim.py + lib/opp_model.py
    primitives (untouched). Added tests/test_baseline_*.py (5 files,
    26 unit/smoke/h2h test cases). Local validation:
      - unit (23 cases): all green in ~3 s
      - smoke vs random both seats + per-turn budget: pass in ~3 min
      - fast.py bench (3 games / 557 turns): p50/p95/max within v15 envelope
      - fast.py eval baseline vs v7_0_drop_one (n=64): PASS, Wilson lo > 0.55
      - fast.py eval baseline vs v15 (n=64): INCONCLUSIVE (CI brackets 0.50)
        → functional parity, which is the expected outcome for a clean re-impl
    Not submitted — Rule 1 (single-shot, PI-approved). The baseline is
    the foundation for architectural pivots (learned value head,
    portfolio search, IL warm-start), with each of value / proposer /
    chooser / opp_model swappable independently.
  - 2026-05-16 — recover-main-foundations / merge-2026-05-16-knowledge.
    Iterated chooser axis v13 → v14 → v15 → v16 → v17 → v18 → v19 → v20
    on top of the v8_scavenge line. v15 cleared panel gate and shipped
    as the new rolling-last-2 champion. v20 (chooser dogpile) is the
    most-recent push. Audits: 2026-05-16-{v13,v14,v15,v16-v20}-*.md.
    Rule 37 (3-variant axis cap) flagged on the chooser-saturation axis
    (state files / audits document the structural ceiling ~μ=1120).
  - 2026-05-14 — game-strategy-eda-roatN.
    Geo iteration; geo v3.1 submitted as #52643676 and settled at floor.
    Cluster-conditional opening overlay FALSIFIED. Detailed in
    audit/2026-05-14-postmortem-geo-session.md and
    knowledge-base/thoughts/2026-05-14-overlay-postmortem.md.
  - 2026-05-13/14 — research-competition-analysis-2R8I3.
    v7_pv (PV target valuation, γ=0.99) shipped at #52630118. Eight
    other interventions FALSIFIED. Architectural finding: v7 + PV is
    a tight local optimum for the K=10 drop-one chooser. Postmortem:
    audit/2026-05-14-postmortem-research-competition-analysis-2R8I3.md.

mechanism_families_explored:
  - heuristic-greedy-nearest-target          # day-1 baseline
  - heuristic-orbit-aware-greedy             # v1 lead-prediction
  - heuristic-orbit-aware-greedy-with-shared-mechanism-layer  # v1.1
  - simple-greedy-target-selection-variants  # nearest/production/roi
  - heuristic-physics-upgrade                # 5-iter aim + safe-intercept
  - heuristic-worldmodel-aware               # v2 WorldModel.owner_at dedup
  - mission-framework-snipe-only             # v3.0 Mission dataclass
  - env-clone-forward-sim-scorer             # Sim<K>
  - lookahead-drop-one-candidates            # v3.1 v3_lookahead
  - full-trajectory-predict-fleet-fate       # lib/trajectory.py
  - cost-aware-roi-additive-denominator      # ROI in v3_snipe
  - comet-lifetime-correction                # time_to_hold caps
  - mission-framework-snipe-plus-reinforce   # v3_snipe
  - aggressive-snipe-ship-sizing-v3.5.1      # FALSIFIED live
  - sigma-equivariance-patches               # 16/16 self-play draws
  - v7-maximin-search                        # v7_minimax
  - v4-receding-horizon-mission-portfolio    # v4_planner
  - v7-drop-one-fast-brain                   # v7_0_drop_one
  - v7-sweep-variants-failed                 # v7_1..v7_4 all FAIL
  - v7-iteration-variants-failed             # super-versions all FAIL
  - v8-psro-self-play-pool                   # parked
  - v9-super-version-failed                  # all FAIL
  - v10-evaluate-value-head                  # FAIL
  - pv-target-valuation                      # H16 v7_pv winner
  - chooser-axis-sweep-7-variants            # all FAIL (5/14 session)
  - geometric-strategy-with-lookahead        # geo v3.1
  - composite-capture-value-head             # lib/value_heads reusable
  - cluster-conditional-opening-overlay      # FALSIFIED
  - v8-scavenge-marginal-delta-chooser       # depth-0 chooser + idle baseline
  - v9-root-cause-fix-stack                  # 4 root-cause fixes on v8
  - v12-crn-opp_traj-state-function-fix      # opp_traj baseline + orbital aim
  - v13-reactive-opp-in-rollouts             # drop CRN; gain realistic counterplay
  - v15-multi-wait-grid-banded-dedup         # extra_surplus (0,5,12) + wait_band {0,1-7,>=8}
  - v16-v20-chooser-saturation-iteration     # F4 vulnerability / dogpile / reactive-step-0 — all HOLD per Rule 37
  - baseline-clean-modular-reimpl-v15        # agents/baseline/ (this branch)
  - composite-head-on-baseline-chooser       # 5/17 PM — first lift past v9_scavenge ceiling
  - timing-aware-validate-cap                # 5/17 PM — leaf-eval cost in chooser budget
  - swept-pair-vanish-classifier             # 5/17 PM — measurement-honesty for replay-mine
  - A2-4P-weakness-exploitation              # 5/17 PM — merged from kaggle-baseline-strategy
  - favor_hybrid-2P-composite-4P-A2          # 5/17 PM — production dispatcher

gate_status: cleared                          # 53+ tests + bundle parity 712 turns
```
