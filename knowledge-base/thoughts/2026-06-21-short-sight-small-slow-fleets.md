# 2026-06-21 — PI observation: short-sighted, small slow fleets, can't wait to mass

> Append-only (Rule 35). Transcribed from PI during replay watching. Plain English.

## The observation (PI's words, lightly cleaned)

> In the loss, we attack early and lose the ships instead of building up position.
> Our fleets are generally small. We seem to be short-sighted, we have difficulty
> waiting — even though waiting brings a speed benefit, since large fleets are faster.

## The load-bearing game fact (verified)

Fleet speed scales with fleet size (data/README.md, comp-context.md):

    speed = 1.0 + (maxSpeed - 1.0) * (log(ships) / log(1000)) ** 1.5   # maxSpeed=6

- 5 ships  -> ~1.6 / turn
- 20 ships -> ~2.4 / turn
- 100 ships-> ~3.7 / turn
- 1000 ships-> 6.0 / turn (max)

So a small fleet is **both weaker AND ~2-2.4x slower** than a massed one. A small early
attack crawls to the target, the defender has many turns to reinforce, and it bounces
(loses the ships). Waiting to mass is doubly good: more ships to overwhelm the defense
AND faster arrival = fewer turns for the defender to react. The PI's intuition is
physically correct.

## Diagnosis (modeling cause)

- The agent's physics is CORRECT: `lib.fleet.speed` / `geometry.fleet_speed` implement
  the exact size-speed formula and it is used for ETAs, threat reach, recapture, etc.
  The offensive/defensive reach in `native_forward.reachable_enemy_mass` even uses
  `fleet_speed(source_garrison)`, so massing a source IS credited with more reach.
- The likely real cause is **SIGHT, not physics.** The planning horizon is
  `LR_HORIZON_2P=18` turns (2P) / `LR_HORIZON_4P=13` (4P). To wait and mass costs
  several turns; the decisive capture then lands near or beyond that horizon edge, so
  the value function never sees the payoff of patience -> "launch a small fleet now"
  out-scores "hold, mass, strike later." That is exactly "short-sighted, can't wait."
- Secondary: the greedy plan-builder sizes/【selects launches with the producer
  ship-count scorer (no hold/speed/stick concept), so it offers small per-ship-efficient
  launches. `LR_NATIVE_BUILDER` (tested earlier) helps target selection but inherits the
  same short horizon, and went too passive when behind (broke seed 6031).

## Hypothesis to test (this session)

Give the agent **longer sight** (raise `LR_NATIVE_HORIZON` / `LR_HORIZON_2P`) so the
delayed-but-decisive massed strike becomes visible and patience can win the comparison.
Watch: does it fix the lead-then-collapse losses (6013, 6007) WITHOUT breaking the
comeback win (6031) or the clean wins — and crucially watch **turn-ms** (a longer
horizon is more compute; the ladder has a per-turn wall). Combine with the builder if
sight alone is insufficient.
