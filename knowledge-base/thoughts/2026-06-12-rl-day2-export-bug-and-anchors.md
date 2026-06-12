# 2026-06-12 — RL track day 2: export bug, real strength, anchors

## The silent-pacifist bug (root-caused + gated)

Night-1 ladder evals were VOID: the numpy port of the launch-angle
solver (`solve_intercept_rows`) indexed a (P,2) array as (P,P,2) —
IndexError on every turn the policy tried to launch, swallowed by the
`agent()` try/except → empty action list → the exported agent idled
through every real game while the JAX policy it mirrored was strong.
Cost: one void n=32 eval (0/32 vs v7_0 measured a bot that never
moved), several confusing "wins" vs producer agents that themselves
idled (bare-file loading breaks their orbit_lite package import).

Permanent gates added (tests/test_rl_numpy_parity.py):
- launch-angle numpy↔JAX parity (duplicate targets included; self-
  targets excluded — masked in the action space and arctan2(0,0) is
  float-noise-arbitrary);
- behavioral: an exported file must EMIT fleets in a kaggle-env game.

Lesson generalized: parity tests on features+forward were necessary
but NOT sufficient — every numpy function on the act() path needs
either a parity test or a behavioral end-to-end test.

## Real night-1 strength (pure mirror self-play, 33M env-steps)

- vs v7_0 (calibrated ~1100μ reference): **30/32, Wilson-lo 0.799**
- vs ledger_v1_4: 2/4 probe (n=32 running)
- vs producer engine: 0/4 — eliminated ~step 165-205 by sustained
  coordinated waves. THE current weakness.
- Turn time: ~80ms single-game, p50 222ms under 4-way contention.

## Scripted anchor results (24-game JAX head-to-heads)

- rusher (75% waves at weakest reachable enemy, neutrals first 40
  steps) beats greedy 21/24 → IN the league anchor set.
- producer_lite (defense-aware margins + reinforcement relays) LOSES
  to greedy 2/24 — too passive + self-relay loops → excluded.
- Lesson: validate every anchor by head-to-head before training
  against it; "looks stronger" isn't.

## Eval-harness traps (cost ~1 h of morning)

- producer family needs torch (`pip install torch --index-url
  https://download.pytorch.org/whl/cpu`).
- producer and producer_plus register the same `orbit_lite` module
  name with different contents → one game per child process
  (`max_tasks_per_child=1`).
- producer_plus has NO orbit_lite of its own on awesome-clarke; it
  needs the branch's newer producer orbit_lite → vendored a copy into
  agents/producer_plus/orbit_lite/.

## CPU-contention eval trap (caught before it poisoned decisions)

Parallel kaggle-env probes force-idle heavy opponents: live_garval
breached the enforced 1s turn budget under 4-way contention and "lost"
4/4 games it actually wins crushingly (had us at 2 planets by step
100). Kaggle gives each agent dedicated cores → sequential probes
(PROBE_WORKERS=1, now the default) are the faithful local read.
fast.py eval does NOT enforce the budget, so its results are
contention-safe pure-strength reads.

## Clean night-1 scorecard (sequential / fast.py)

- v7_0: 30/32, Wilson-lo 0.799 (PASS)
- ledger_v1_4: 24/32, Wilson-lo 0.579 (PASS)
- producer: 0/4 probe (n=32 running)
- live_garval (live-pair rebuild, sub 53588922 env config —
  external/live_garval/): 0/4, eliminated step 172-249

Read: night-1 ≈ v7_0+ class (~1150-1250 live guess); producer-engine
style is THE wall. Rule 45 gate clearly unmet → no submit on merit.

## Submission posture (as of ~09:00 UTC)

Rule 42 BLOCKED for an RL submit today without PI sign-off: would
evict sync_on (1254) and honest prediction is ~1100-1250 (beats
v7_0-class, loses producer-class). Calibration-value case surfaced to
PI; awaiting their call. 4 slots left today (shared with other
branches).

## Plan

- v6 league kernel completes ~14:40 UTC → 12-game probe → kernel v7
  tonight: resume v6, league with {rusher, greedy} anchors +
  snapshots, --eval-opp rusher, --greedy-frac 0.4.
- Afternoon lever if time: value-reranked inference (k samples ×
  1-step exact rollout × value head) — only after checking value-head
  outcome correlation.
- GPU quota: ~17h used of 30h/week after v6; v7 (~8h) fits.
