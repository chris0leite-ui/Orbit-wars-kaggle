# 2026-06-04 — circulation family parked, not killed (PI explicit)

PI on session close: "this is not done yet. Just... we couldn't transfer the
results to our strategy."

## The real observation stands

"Our agent does not use all our ships, especially rear stockpiles that sit
idle far from action" remains a PI-verified problem from live games. Not
falsified. Worth re-attacking under a different mechanism shape.

## What we tried (and why it didn't transfer)

Three implementations of pressure-gradient regroup as a post-pass after
the chooser:

- **v1 (centroid distance)**: 5/16 + wallclock 2958 ms max. Centroid is
  geometry-only; not enough signal.
- **v2 (Biel's distance-decayed enemy mass)**: 8/16 + max 1424. Right
  scalar field, ships go to "high pressure" friendlies — but those
  friendlies don't necessarily fire.
- **v3 (v2 + destination-must-be-able-to-capture-today)**: 5/16 + max
  1396. Stricter filter regressed.

## Why it doesn't transfer (the diagnosis)

Biel's "Producer" agent on Kaggle works because **his entire planner
thinks in pressure** — same garrison forecast feeds his attack waves AND
his regroup destinations. When his regroup pushes ships toward a high-
pressure friendly, his next-turn attack planner naturally fires from
there.

**Our chooser thinks in `(source planet, target planet)` trades**:
enumerate 8 nearest opp targets per source, score "is THIS attack
positive-EV". It has no notion of pressure. So a pressure-gradient
regroup routes ships to destinations our chooser doesn't act on.

The mismatch is structural — same source signal in his stack, different
representation in ours. Tightening the destination filter (v3) doesn't
help because the filter cuts off launches that were apparently providing
incidental value (defensive density, pre-positioning for future captures
the chooser eventually finds).

## What would unblock it

Two paths, both bigger than a single post-pass:

1. **Rewrite the chooser's scoring to be pressure-aware** — essentially
   port Biel's planner. Large project. High risk, high reward.
2. **Goal-directed 2-hop pre-positioning** — instead of "send to high
   pressure", identify a CONCRETE 2-hop attack the chooser is missing
   today, and ship only for that play. Smaller; chooser-aware by
   construction.

## Operational state

- All code preserved behind `BASELINE_FRONTIER_CIRCULATION=1` (default OFF).
  Three commits on this branch: 924b44a (v1), 24ac0d7 (v2), b836407 (v3).
- Live champion: `champ_computeByShips_on.py` sub 53332500. Unaffected.
- Idle stockpile spend-down code (also default OFF, from earlier in this
  session) similarly preserved as 0bb53f3 + e24a99f.
