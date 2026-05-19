# state/current.md — current submitted agent + tournament rank

> **Score values are intentionally NOT recorded here.** They drift as the
> rolling μ DRIFTS as the leaderboard moves (it does NOT settle to a
> final value — TrueSkill keeps updating). ALWAYS query Kaggle
> directly at session start (Rule 32):
>
>     export KAGGLE_USERNAME="$KaggleUserName" KAGGLE_KEY="$KaggleAPIToke" \
>            KAGGLE_API_TOKEN="$KaggleAPIToke"
>     kaggle competitions submissions orbit-wars
>
> Updated 2026-05-17 by `claude/kaggle-baseline-strategy-lO4mm`
> (clean modular re-baseline of v15).

```yaml
date: 2026-05-19
deadline: 2026-06-23 23:59 UTC
days_to_deadline: 35

# Most-recent submission (hold-feasibility filter solo).
# Built on `claude/audit-workflow-performance-btjeK` HEAD. Trajectory chooser
# unchanged (BASELINE_CHOOSER=trajectory default at main.py:38). The filter
# (`_target_holdable_after_capture` at proposer.py:407, gated at :627) was
# default-on since 2026-05-18 PM but had never been the sole change in a
# submission. This is the calibration probe.
#
# Local validation:
#   B.3 h2h solo A/B (treatment vs control with filter disabled at line 627):
#     25/32 = 78.1% Wlo=0.612 Whi=0.890 PASS (early-stop at n=32)
#   B' panel A/B + champion h2h (all PASS at Wlo >= 0.55):
#     vs champion 52784853:  24/32 = 75.0% Wlo=0.579 (closest to gate)
#     vs v7_0:               30/32 = 93.8% Wlo=0.799
#     vs v4_planner:         29/32 = 90.6% Wlo=0.758
#     vs v3.5.1:             27/32 = 84.4% Wlo=0.682
#   Bundle parity OK over 574 turns.
#   Wallclock: focal p50=310ms p95=738ms max=1268ms (max > 1s soft cap,
#   matches current source's profile; not a new risk).
last_submission_id: 52827111
last_submission_status: PENDING
last_submission_mu: null  # no μ yet; refresh via kaggle CLI
last_submission_message: "comet-aim + reactor-aware: 2P A/B comet-only 64/96=66.7pct Wlo=0.568 PASS; 4P FFA combined 89/127=70.1pct, comet-only 89/128=69.5pct CI[61.1,76.8], no-reactor 80/128=62.5pct CI[53.9,70.4]; comet-aim alone is +7pp, reactor-aware adds +0.6pp on top (within noise); Rule-38 trace ep 77087563 vs Felix Truong confirms 40-ship OOB fixed"
last_submission_file: submissions/baseline.py
last_submission_agent: baseline_comet_aim_plus_reactor
last_submission_sha256: 90d2034141054d2022e968c081e3b466d1608347d63b1d56657b34bc4b0370ef
last_kernel_push: 2026-05-19 19:52:53 UTC
prior_submission_id: 52811320
current_submitted_agent: baseline_comet_aim_plus_reactor (5/19 evening)

# Rolling-last-2 (Kaggle auto-keeps these two for final evaluation; the
# third push auto-evicts the previous oldest). Per the literal
# "rolling LAST 2 submissions" rule. Verified via `kaggle competitions
# submissions orbit-wars` 2026-05-19 19:52:
#   52811320 (May 19 12:54, μ=1137.5) — hold-feasibility filter solo,
#     SETTLED at 1137.5 (was drifting around 1067 yesterday). Kept.
#   52784853 (May 18 17:42, μ=1130.4) — EVICTED by 52827111 push.
#   52827111 (May 19 19:52, μ=PENDING) — comet-aim + reactor-aware.
# Floor of the rolling pair after push: 1137.5 (52811320) until 52827111 settles.
# Calibration note: 52811320 climbed dramatically from drift-low ~1067
# to settled 1137.5 — TrueSkill needs 24h+ to settle reliably.
rolling_last_2:
  - {agent: baseline_comet_aim_plus_reactor, sub_id: 52827111, submitted: 2026-05-19T19:52Z, status: PENDING, mu_snapshot: null}
  - {agent: baseline_hold_feasibility_solo, sub_id: 52811320, submitted: 2026-05-19T12:54Z, status: COMPLETE, mu_snapshot: 1137.5}

# Team peak as of 2026-05-17: v15_banded (the multi-wait-grid + banded
# (src, tgt, wait_band) dedup line). v15 source lives in git history at
# f315dc7:agents/v15/main.py — NOT in the working tree (the "nuke
# historical strategy code" reset removed it; we kept the audits).
# A clean modular re-implementation lives at agents/baseline/.
team_peak_agent: v15_banded

# Calibration WARNING (still active 5/19 — applies to next submissions):
# Multiple recent submissions over-predicted live. Local-vs-live mapping
# has been roughly -20 to -30 pp on every recent submission. Use a
# 3-opponent panel (`fast.py eval --vs-panel`) + h2h vs the current
# rolling agent (not just a fixed baseline) before any new push. The
# 52811320 push followed this protocol (panel + champion h2h all PASS).

submissions_used_today: 2     # 5/19 — 52811320 (hold-feasibility solo) + 52827111 (comet-aim + reactor-aware)
submissions_used_total: 35    # see ladder list below; refresh via Kaggle CLI
plateau_days: 0
saturation_count: 0

# Live ladder — read from `kaggle competitions submissions orbit-wars`,
# NOT from this file. Submission IDs are stable; scores are not.
# Most recent ladder entries by submission id:
#   52827111  baseline.py (comet-aim + reactor-aware)      2026-05-19 19:52 PENDING ← NEW
#   52811320  baseline.py (hold-feasibility filter solo)   2026-05-19 12:54 COMPLETE μ=1137.5 ← rolling pair
#   52784853  baseline.py (PV-off + bug #3/#4/#12 fixes)   2026-05-18 17:42 COMPLETE μ=1130.4 EVICTED by 52827111
#   52766596  baseline.py (joint v3 2P-only)               2026-05-18 07:12 COMPLETE μ=1118.3 EVICTED by 52811320
#   52754310  baseline.py (trajectory v4 + wait_N)         2026-05-17 22:06 COMPLETE μ=1143.7
#   52744856  baseline.py (composite+A2 hybrid)            2026-05-17 14:17 COMPLETE μ=1149.2
#   52744234  baseline.py (composite+A2 hybrid, ERROR)     2026-05-17 13:57 ← failed: `from agents.baseline import` not inlined
#   52721807  v20.py            2026-05-16 21:57  COMPLETE μ=1076.7
#   52710995  v15.py            2026-05-16 13:43  COMPLETE μ=1115.0  ← team peak
#   52704189  v8_scavenge (v13) 2026-05-16 09:07  COMPLETE μ=1085.7
#   52699232  v8_scavenge (v12) 2026-05-16 05:37  COMPLETE μ=1095.4
#   52687411  v8_scavenge (v9)  2026-05-15 17:41  COMPLETE μ=1119.9
#   52684059  v8_scavenge (v8)  2026-05-15 15:05  COMPLETE μ=1065.8
#   52678866  iter.py  (iter_v2) 2026-05-15 11:34 COMPLETE μ=1036.0
#   52661990  iter.py  (iter_v1) 2026-05-14 21:48 COMPLETE μ=1034.7
#   52643676  geo.py             2026-05-14 09:10 COMPLETE μ=1004.9
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
  - 2026-05-19 evening — audit-workflow-performance-btjeK.
    Three commits + one push. (a) `037009b`: reactor-aware launch
    selection — cost-parity filter (reject candidates where the cheapest
    opp reactor pays < 70 pct of our cost) + reactor-candidate generator
    (propose our own race-to-recapture for opp fleets in flight). 2P A/B
    INCONCLUSIVE (61/128 = 47.7 pct Wlo=0.392 Whi=0.563), drilldown
    showed local "regression" was wallclock artifact under 6-worker
    contention on 4 CPUs — agent decisions identical when given enough
    wallclock; max=1332ms in contended A/B vs 442ms in serial bench.
    (b) `dbbc535`: comet-aim path-indexed lead — new aim_comet 5-iter
    fixed-point sibling of aim_orbiting that reads obs.comets[g].paths
    instead of predict_relative orbital rotation; predict_fleet_fate
    also updated to use the path. Env-var BASELINE_COMET_AIM=on default.
    Rule-38 trace on ep 77087563 (sub 52811320 vs Felix Truong): pre-fix
    angle 160.01° at step 51 → step-8 OOB at (-2.19, 80.49); post-fix
    angle 121.25° → trajectory filter correctly drops the candidate as
    wrong-planet. (c) `cb8b5aa`: 4P FFA panel JSON committed.
    Verifications: comet-only 2P A/B 64/96=66.7 pct Wlo=0.568 PASS;
    4P FFA panel vs {v7_0_drop_one, v3.5.1, roi}:
      combined (reactor+comet): 89/127 = 70.1 pct (partial; container
                                cycled at game 127)
      comet-only:               89/128 = 69.5 pct CI [61.1, 76.8]
      no-reactor (52811320):    80/128 = 62.5 pct CI [53.9, 70.4]
    Comet-aim is +7pp solo; reactor-aware adds +0.6pp (within noise).
    Pushed combined bundle 52827111 at PI direction ("submit with
    reactor — I want to see how it behaves"). 52811320 settled at
    μ=1137.5 (was drifting around 1067 yesterday — TrueSkill needs
    24h+ to settle).
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
  - reactor-aware-launch-selection           # 5/19 PM — cost-parity filter + reactor candidates (037009b)
  - comet-aim-path-indexed-lead              # 5/19 PM — fix comet motion model (dbbc535)

gate_status: cleared                          # 60+ proposer/aim/trajectory tests + bundle parity
```
