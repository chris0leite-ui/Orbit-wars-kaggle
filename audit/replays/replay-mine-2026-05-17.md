# replay-mine — 2026-05-17

PI buckets: `win`=captured, `defense`=reinforced own, `waste_attack`=bounced, `waste_comet`=killed by comet swept-pair, `waste_trajectory`=sun/oob/vanished, `inflight`=alive at end, `unknown`=other.

## per-submission roll-up

| sub_id | ep | fleets | win% | def% | waste_atk% | waste_comet% | waste_traj% | inflight% | unknown% |
|---|---|---|---|---|---|---|---|---|---|
| 52710995 | 92 | 9507 | 47.4 | 35.2 | 15.7 | 0.1 | 0.9 | 0.7 | 0.0 |

## cross-submission totals

- fleets launched: 9507 across 92 episodes in 1 submissions
- ships launched: 454392

By bucket (count and percentage):
- `win              `  4511 fleets (47.4%) — 247677 ships (54.5%)
- `defense          `  3346 fleets (35.2%) — 134572 ships (29.6%)
- `waste_attack     `  1489 fleets (15.7%) —  58225 ships (12.8%)
- `waste_comet      `     5 fleets ( 0.1%) —   1738 ships ( 0.4%)
- `waste_trajectory `    89 fleets ( 0.9%) —   2247 ships ( 0.5%)
- `inflight         `    65 fleets ( 0.7%) —   9881 ships ( 2.2%)
- `unknown          `     2 fleets ( 0.0%) —     52 ships ( 0.0%)

Raw outcomes (debug):
- `captured` 4511
- `reinforced_self` 3346
- `bounced_enemy` 1357
- `arrived_but_lost` 87
- `alive_at_end` 65
- `oob` 64
- `bounced_neutral` 45
- `sun` 13
- `vanished_in_space` 12
- `comet_collision` 5
- `hit_planet_unknown_flip` 2
