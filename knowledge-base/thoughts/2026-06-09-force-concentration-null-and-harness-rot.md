# 2026-06-09 — force-concentration null result + harness rot

(AI session entry, autonomous session on `claude/awesome-clarke-ixy57v`;
PI prompt was "improve the latest submitted strategy, check the work on
other branches".)

## The headline numbers

Clean n=32 seat-balanced A/Bs vs vanilla producer, one game per
subprocess, zero errors/timeouts:

- `multi_opp_def` (rebuilt from today's HEAD): **24/32 = 75.0%**,
  Wilson [0.579, 0.867] — exactly reproduces the 2026-06-05 measurement.
- `force_concentration` standalone: **6/32 = 18.8%**.
- `lean_force_concentration` (multi_size + recapture + FC): **7/32 = 21.9%**.
- `multi_tick_force_concentration` (full stack + FC): **5/32 = 15.6%**.

Force-concentration — the relaxed one-wave-per-target mutex committed
2026-06-05 12:27 after the session wrap, never A/B'd until today — is a
hard null. All three variants lose roughly 4 of 5 games to the very agent
the base stack beats 3 of 4. The standalone failure means the mechanism
itself (not a composition effect) is harmful. Hypotheses for a future
diagnosis session: the second wave double-drains sources the first wave
needed for defense, or the rescore closure's score offset re-adds bonuses
that no longer apply. Do not re-ship without a replay-level diagnosis.

## The live-ladder situation that motivated the session

The 06-07 manual resubmit of multi_size (sub 53450504, no description)
evicted our best-ever agent multi_opp_def (settled 1263–1287) and itself
settled at only 1181 — ~100 below what identical code settled at on
06-04. Two lessons:

1. **The ladder strengthens fast.** ~100 μ of field drift in 3 days.
   Never compare a resubmission's settle to a historical settle.
2. **A resubmit is an eviction.** The rolling-pair model means restoring
   an old file mid-pair can evict the best member. Rule 42 exists for
   exactly this; the 06-07 submit had no claim row.

## Harness rot: three layers of silent test invalidation

Current kaggle_environments (installed fresh by bootstrap) broke the
producer_plus test layer in three independent ways, all silent:

1. **`__file__` is not defined** when the loader execs an agent file.
   Every shim in `agents/producer_plus/` and both `producer_agent.py`
   wrappers crashed at load; with `debug=False` the focal agent then
   plays None every turn and the game "completes". Two different shims
   produced byte-identical 83-step losses — the tell that neither was
   actually playing. Fix: recover the agent dir from `sys.path[-1]`
   (the loader appends the exec dir to sys.path during exec).
2. **Env-var pollution across in-process games.** Shims set their gates
   via `os.environ.setdefault` at load; in a single pytest process the
   keys outlive the game and re-gate every later game. Fix: tests pop
   all `PRODUCER_PLUS_*` keys before each game.
3. **Rewards are now win/loss (±1), not ship counts.** The
   "changes_planner_output" tests compared final rewards; two variants
   that both beat producer read as "identical". Fix: compare the focal
   player's serialized action stream.

Rule 38 was the right instinct here: the first "fix" (env clearing)
passed its unit-level reasoning but the failure state persisted; only
reproducing the actual game and hashing the action streams exposed the
dead-agent layer underneath.

## Status at session end

`producer_plus_multi_opp_def_on.py` rebuilt, re-validated (75%),
Rule 46 green (15/15 bundle tests, max turn 89 ms at seed 7), claim row
on the board predicting ≈1180–1260 vs evicted 1099.3 — awaiting PI
sign-off to submit (Rule 1).


## Addendum (2026-06-10, same session)

After the restore submit (sub 53523036), two more mechanisms measured
and banked as nulls, n=32 clean A/B vs vanilla producer each:

- **Strategic bonuses at calibrated weights.** Probe first (one full
  game, 141k candidate scores): median acted-on score 48 ship units;
  denial at weight 1.0 has median 354, opening 60. Weights 0.01 / 0.04
  put the bonus at 5-7% of the acted-on median — the prescribed nudge
  band. Results: denial 16/32, opening 15/32, composed 18/32. A 7%
  nudge cost ~20-25 points of win rate: the terms point the planner at
  the wrong captures (racing for contested high-production planets
  produces thin captures), so the failure is shape, not scale. The
  2026-06-05 "re-tune to 0.005-0.02" plan is closed — calibration was
  necessary but not sufficient.
- **Scorer horizon 18 → 24**: 17/32. H=18 is co-calibrated with the
  engine's other constants; raising it alone regresses.

Nine mechanisms measured on the producer engine to date; the only
survivor is multi_size + opp_projection (the live agent). The engine is
at a local optimum w.r.t. the vs-producer yardstick. Next lift should
come from live-loss observation, not from another engine-side term.
