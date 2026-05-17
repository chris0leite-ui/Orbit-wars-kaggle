# replay-mine — 2026-05-17

PI buckets: `win`=captured, `defense`=reinforced own, `waste_attack`=bounced, `waste_comet`=killed by comet swept-pair, `waste_trajectory`=sun/oob/vanished, `inflight`=alive at end, `unknown`=other.

## per-submission roll-up

| sub_id | ep | fleets | win% | def% | waste_atk% | waste_comet% | waste_traj% | inflight% | unknown% |
|---|---|---|---|---|---|---|---|---|---|
| 52744856 | 13 | 1823 | 47.7 | 40.2 | 10.8 | 0.2 | 0.8 | 0.3 | 0.1 |

## cross-submission totals

- fleets launched: 1823 across 13 episodes in 1 submissions
- ships launched: 162109

By bucket (count and percentage):
- `win              `   870 fleets (47.7%) — 102151 ships (63.0%)
- `defense          `   732 fleets (40.2%) —  47752 ships (29.5%)
- `waste_attack     `   197 fleets (10.8%) —   8890 ships ( 5.5%)
- `waste_comet      `     3 fleets ( 0.2%) —   2864 ships ( 1.8%)
- `waste_trajectory `    15 fleets ( 0.8%) —    185 ships ( 0.1%)
- `inflight         `     5 fleets ( 0.3%) —    184 ships ( 0.1%)
- `unknown          `     1 fleets ( 0.1%) —     83 ships ( 0.1%)

Raw outcomes (debug):
- `captured` 870
- `reinforced_self` 732
- `bounced_enemy` 166
- `arrived_but_lost` 20
- `bounced_neutral` 11
- `oob` 6
- `vanished_in_space` 5
- `alive_at_end` 5
- `sun` 4
- `comet_collision` 3
- `hit_planet_unknown_flip` 1
