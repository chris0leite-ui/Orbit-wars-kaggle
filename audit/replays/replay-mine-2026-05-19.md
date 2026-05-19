# replay-mine — 2026-05-19

PI buckets: `win`=captured, `defense`=reinforced own, `waste_attack`=bounced, `waste_comet`=killed by comet swept-pair, `waste_trajectory`=sun/oob/vanished, `inflight`=alive at end, `unknown`=other.

## per-submission roll-up

| sub_id | ep | fleets | win% | def% | waste_atk% | waste_comet% | waste_traj% | inflight% | unknown% |
|---|---|---|---|---|---|---|---|---|---|
| 52784853 | 94 | 9018 | 41.0 | 43.2 | 13.7 | 0.0 | 1.2 | 0.9 | 0.0 |
| 52766596 | 102 | 11531 | 43.3 | 38.3 | 16.8 | 0.1 | 1.0 | 0.6 | 0.0 |
| 52754310 | 102 | 11005 | 44.2 | 36.3 | 17.6 | 0.0 | 1.2 | 0.7 | 0.0 |
| 52744856 | 94 | 11102 | 45.6 | 37.1 | 15.8 | 0.1 | 1.0 | 0.4 | 0.0 |
| 52721807 | 109 | 14186 | 40.6 | 43.5 | 14.3 | 0.0 | 1.0 | 0.6 | 0.0 |

## cross-submission totals

- fleets launched: 56842 across 501 episodes in 5 submissions
- ships launched: 2799723

By bucket (count and percentage):
- `win              ` 24358 fleets (42.9%) — 1444787 ships (51.6%)
- `defense          ` 22612 fleets (39.8%) — 955587 ships (34.1%)
- `waste_attack     `  8900 fleets (15.7%) — 343750 ships (12.3%)
- `waste_comet      `    24 fleets ( 0.0%) —  15330 ships ( 0.5%)
- `waste_trajectory `   599 fleets ( 1.1%) —  17724 ships ( 0.6%)
- `inflight         `   343 fleets ( 0.6%) —  22245 ships ( 0.8%)
- `unknown          `     6 fleets ( 0.0%) —    300 ships ( 0.0%)

Raw outcomes (debug):
- `captured` 24358
- `reinforced_self` 22612
- `bounced_enemy` 7913
- `arrived_but_lost` 554
- `oob` 453
- `bounced_neutral` 433
- `alive_at_end` 343
- `vanished_in_space` 94
- `sun` 52
- `comet_collision` 24
- `hit_planet_unknown_flip` 6
