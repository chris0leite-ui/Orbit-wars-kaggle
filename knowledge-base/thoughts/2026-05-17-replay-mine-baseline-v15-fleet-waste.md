# v15 fleet-waste baseline, measured from 92 live replays

Date: 2026-05-17
Branch: claude/audit-workflow-performance-btjeK
Source: `audit/replays/replay-mine-2026-05-17.{json,md}`
Tool: `scripts/replay_mine.py 52710995`

## Numbers

v15 launched **9,507 fleets / 454,392 ships** across 92 episodes
(live ladder, submission 52710995, μ~1108.4 settled).

| Bucket             | Fleets | %    | Ships  | %    |
|--------------------|-------:|-----:|-------:|-----:|
| `win` (captured)   | 4,068  | 42.8 | 199,117 | 43.8 |
| `defense` (own)    | 3,123  | 32.8 | 113,585 | 25.0 |
| `waste_attack`     | 1,398  | 14.7 |  50,054 | 11.0 |
| `waste_trajectory` |   851  |  9.0 |  81,703 | **18.0** |
| `inflight`         |    65  |  0.7 |   9,881 |  2.2 |
| `unknown`          |     2  |  0.0 |      52 |  0.0 |

Raw outcome breakdown of the two waste buckets:

- `bounced_enemy`  1,271   (attack failed: under-sized vs defender)
- `vanished_in_space`  838 (mid-flight disappearance — comet? OOB?)
- `arrived_but_lost`   82  (captured then lost on same tick)
- `bounced_neutral`    45  (under-sized vs neutral)
- `sun`                13  (direct sun collision)

## Reading

1. **24% of fleets / 29% of ships are wasted.** That is the
   compounding ceiling PI named — efficient agents would
   redirect that quarter of the budget at the win bucket.
2. **The dominant ship-leak is `vanished_in_space` (18% of
   ships, 8.8% of fleets).** Ship-weighted is 2x the
   fleet-weighted rate — we send the *bigger* fleets on the
   trajectories that vanish. This is the largest single
   inefficiency on the board.
3. **Sun-deaths are real but small** (13 fleets, 0.14%).
   PI's intuition was right that they exist; the magnitude is
   small enough that the sun-avoidance test
   (`tests/test_mech_sun_avoid.py`) is already mostly working.
   Pivot #5 in the diagnostic plan should be re-prioritised
   *after* the vanished-in-space root cause is known.
4. **`bounced_enemy` (1,271 fleets) is the second-largest
   leak.** This is "we attacked a defended planet with too
   few ships." `composite_capture_value`'s waste-penalty term
   targets exactly this — when `ships ≤ pred_ships`, subtract
   `waste_weight × ships`. The mechanism exists in `lib/
   value_heads.py`; wiring it into baseline is pivot #2.

## What this changes about the plan

- Pivot #1 (replay-mine) → **DONE**, with surprising finding:
  trajectory-waste is bigger than sun, and vanished-in-space
  is the dominant component. PI's sun-death hypothesis was
  directionally right but mis-sized; the real leak is one
  level deeper.
- Pivot #2 (composite head) → still the right next step. The
  waste-penalty term directly targets `bounced_enemy` (13.4%
  of fleets). 4P semantics unclear — flag filed.
- Pivot #5 (sun-fix) → **demoted** below "identify the
  vanish mechanism." 13 sun-deaths in 9,507 fleets is rare
  enough that the test suite probably already constrains it;
  the 838 mid-flight vanishes are what we should chase.
- Open question filed at `knowledge-base/questions/
  2026-05-17-vanished-in-space-cause.md`.

## How PI grades this

Re-run `scripts/replay_mine.py 52710995` any time the
replay cache is fresh. The same script applied to a new
candidate's submission will give a directly-comparable
table. The waste% delta is the primary KPI for the next
A/B cycle: a successful pivot drops the waste_attack +
waste_trajectory total below 20% (currently 23.7%) without
losing the 42.8% win rate.
