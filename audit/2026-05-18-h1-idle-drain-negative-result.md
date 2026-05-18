# H1 post-chooser idle drain — negative result (2026-05-18)

## TL;DR

`drain_idle_rear()` post-chooser heuristic in `agents/baseline/main.py`
targeted the 43.8% isolated ship-turns measured in
`audit/replays/idle-trajectory-2026-05-17.md`. A/B FAILED:

- **2P** (clean bundle, n=32, vs hybrid trajectory): **11/32 = 34.4%**
  Wlo=0.204 Whi=0.517 — clear FAIL.
- **Wallclock**: max=1528ms, p95=829ms — over the 1000ms env cap.
- **4P**: not completed (terminated after 2P failure).

**Default flipped to OFF.** Opt-in via `BASELINE_IDLE_DRAIN=1`. Live
production (52754310 trajectory+hybrid, μ=1271.8) preserved.

## What H1 did

After the chooser returned its tactical moves, for each of OUR planets
NOT in those moves, if ALL of:
- `source.ships > 30` (IDLE_DRAIN_THRESHOLD)
- min-distance to any non-our planet > 35 (IDLE_REAR_THRESHOLD)
- `WorldModel.time_to_enemy_threat(source)` returns None (no threat)
- exists own planet strictly closer to action

Then append a reinforce launch from source toward closest own planet,
with `source.ships - 5` ships.

Rationale: the chooser leaves rear sources idle because no positive-Δ
launch exists. H1 doesn't touch the chooser's Δ — only adds drainage
moves. Hypothesis: forward-deployed ships are higher-EV than idle
rear ships.

## Why H1 failed (hypothesis)

The chooser's decision to leave rear ships idle is NOT a leak — it's
correctly-held DEFENSIVE RESERVE. Single-step Δ scoring captures
this: a launch from rear → forward has Δ ≤ 0 because:

- `my_ships` is unchanged (ships in flight still count for me)
- `opp_ships` is unchanged
- `production` unchanged
- No capture occurred
- BUT now those ships are committed to flight for 10-30 ticks —
  they cannot defend the source if opponent later attacks it

The chooser correctly rejects these launches. H1 force-emits them
anyway, which:
1. Locks ships out of defense for 10-30 ticks per launch
2. Burns "optionality" — the rear source's ships could have been
   reserved for a future opportunity the chooser will eventually find
3. Doubles per-turn fleet count, doubling per-turn wallclock load
   (max=1528ms confirms)

Result: H1-augmented agent loses 65.6% of head-to-head 2P games
against unmodified chooser. The chooser correctly held the reserve.

## Generalizable conclusion

**The 43.8% isolated ship-turns is not a leak.** It's the natural
distribution of a calibrated reserve. The audit measured a SYMPTOM,
not a problem.

This adds to:
- Spatial leaf head (commit b5f5296) — failed 2P 40.6%, 4P 9.4%
- H1 post-chooser drain (commit 1b3f920) — failed 2P 34.4%

**Both attempts to "fix" idle ships hurt winrate.** The chooser at
μ=1271.8 is closer to optimal than the audit suggested.

## What this rules out

1. Spatial leaf modifications to the chooser's Δ
2. Post-chooser forced-emission heuristics

By extension (same root cause: single-step Δ doesn't see multi-step
value):
3. H2 (stage-route long launches via proposer) — stage-to-own has
   Δ=0, chooser would reject anyway.
4. H4 (garrison-cap forced launch) — same forced-emission family
   as H1.

## What remains worth trying

- **Direction B (joint candidates)**: multi-step planning is the
  ONLY way to give "stage-then-capture" positive Δ. The chooser
  needs to see the two-step plan as one unit.
- **H3 (cheap-rank floor raise for distant candidates)**: if
  cheap-rank is currently rejecting candidates that would actually
  capture, raising the floor unlocks them WITHOUT forcing emissions
  (chooser still validates). Low expected gain — cheap-rank only
  filters bounce candidates, which the chooser would reject anyway.
- **Mining top-5 LB notebooks (Rule 22)**: structural ideas beyond
  trajectory chooser.

## Code state

- `agents/baseline/main.drain_idle_rear` — function preserved, opt-in
  via `BASELINE_IDLE_DRAIN=1`. Default OFF.
- 8 unit tests in `tests/test_baseline_idle_drain.py` pass with
  autouse fixture (force-enable for test scope).
- Bundle `submissions/baseline.py` unchanged (live agent at μ=1271.8).

## Rule applications

- **Rule 38** (fix-verification reproduces failure): H1 unit-tested
  in isolation (8/8 green) BEFORE A/B; A/B is the integration
  reproduction. Failed there.
- **Rule 1** (submission discipline): A/B failed → no submission.
- **Rule 37** (3-variant axis cap): "post-chooser idle-emission"
  axis: 1/3 used (this H1). 2 variants remain BUT given the failure
  cause analysis (forced-emission breaks calibration), unlikely
  worth spending.
- **Rule 40** (modeling vs restriction-tuning): H1 IS a heuristic
  patch (restriction), not modeling. The right modeling fix would
  be Direction B (joint candidates), which gives reserve-then-deploy
  positive Δ at the chooser level.
