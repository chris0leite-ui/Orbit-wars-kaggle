# HANDOVER.md — next-session brief

## Mode

**Observation-driven iteration on a single strategy.** No parallel exploration.
One observation from the PI → one mechanism → one push.

## Strategy

`baseline_adaptive_k` — see `state/STRATEGY.md` for the full spec, the build
script, the smoke procedure, and the iteration protocol.

Read `state/STRATEGY.md` first thing every session.

## Live status

- **Latest submission (#1):** `champ_computeByShips_on.py`, sub **53332500**
  (2026-06-03 15:11 UTC), bundle sha256 `53bf813b...`, 697 927 B. Adaptive K
  + compute_by_ships lever both ON. Predicted μ ≈ 1170 (parity with sibling
  per local n=16 A/B).
- **Backstop (#2):** `champ_adaptiveK_on.py`, sub **53324164**, live
  **μ = 1185.2** (our anchor — stays in the rolling pair).
- **TrueSkill warm-up reminder:** starts at μ ≈ 600 and climbs over ~24 h. Do
  not interpret the first few hours of leaderboard data.
- Read the rolling pair on demand:
  `kaggle competitions submissions orbit-wars | head -5`.

## Today's progress (2026-06-04, session-level)

**Ship-utilization mechanism family parked — not killed.**

Pursued three implementations of pressure-gradient ship circulation as a
post-pass over the chooser. All three falsified:

- v1 (centroid scalar field) — 5/16 wins, wallclock max 2958 ms — `924b44a`
- v2 (Biel's distance-decayed enemy mass) — 8/16 wins, max 1424 ms — `24ac0d7`
- v3 (v2 + destination-usefulness filter) — 5/16 wins, max 1396 ms — `b836407`

**Root cause** (code-review diagnosis after v2): Biel's "Producer" agent on
Kaggle uses an identical mechanism successfully because his entire planner
thinks in pressure (same scalar feeds attack scoring AND regroup
destinations). Our chooser thinks in (source, target) trade scoring with no
pressure notion → pressure-routed ships land at destinations our chooser
ignores. Cannot be patched with thin filters.

**PI explicit at session end:** "this is not done yet. Just... we couldn't
transfer the results to our strategy." The observation — rear stockpiles
sit idle while front fights — remains real and PI-verified. What we
falsified is the post-pass mechanism shape, not the underlying need.

All code preserved behind `BASELINE_FRONTIER_CIRCULATION=1` (default OFF).
Champion `champ_computeByShips_on.py` unaffected.

Earlier in the session (2026-06-03 work, already shipped): compute_by_ships
parity, idle_stockpile parity-after-gate-tighten — both default-OFF in the
live champion.

## Next action

**Pivot.** Do NOT continue tuning frontier_circulation in v4 form. Two
non-trivial paths remain for the ship-utilization observation; do NOT
pursue without first replay-mining 3-5 concrete cases of "rear ships
could have been used":

1. Chooser-internal rewrite — port pressure-aware scoring into
   `cheap_marginal_value`. Multi-session build. See
   `knowledge-base/questions/2026-06-04-chooser-pressure-port-vs-2hop-targeting.md`.
2. Goal-directed 2-hop pre-positioning — identify a concrete (rear →
   mid-friendly → opp) sequence the chooser is one launch short of, and
   pre-position only for that play. Smaller; chooser-aware by construction.

For an immediate next-session move that is NOT in this family: wait for
the next PI observation from live games. The strategy doc's
observation-driven loop applies.

## Pointers

- `state/STRATEGY.md` — strategy, build, smoke, iteration protocol.
- `CLAUDE.md` — process rules (lean).
- `state/MULTI_BRANCH.md` — push-claim board (Rule 42).
- `audit/2026-06-04-postmortem-champion-ml-graft-majestic-storm.md` — full
  postmortem of today's circulation triplet.
- `knowledge-base/thoughts/2026-06-04-circulation-family-parked-not-killed.md`
  — diagnosis and unblock paths.
- `knowledge-base/flags/2026-06-04-ship-utilization-still-open.md` — watch
  flag for future opportunities.
- `knowledge-base/questions/2026-06-04-chooser-pressure-port-vs-2hop-targeting.md`
  — open design question on the two unblock paths.
