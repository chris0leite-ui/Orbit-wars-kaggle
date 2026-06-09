# replay-mine — 2026-06-01

PI buckets: `win`=captured, `defense`=reinforced own, `waste_attack`=bounced, `waste_trajectory`=sun/oob/vanished, `inflight`=alive at end, `unknown`=other.

## per-submission roll-up

| sub_id | ep | fleets | win% | def% | waste_atk% | waste_traj% | inflight% | unknown% |
|---|---|---|---|---|---|---|---|---|
| 53182323 | 120 | 18117 | 22.1 | 55.8 | 11.4 | 10.1 | 0.7 | 0.0 |

## cross-submission totals

- fleets launched: 18117 across 120 episodes in 1 submissions
- ships launched: 1120717

By bucket (count and percentage):
- `win              `  4001 fleets (22.1%) — 239080 ships (21.3%)
- `defense          ` 10105 fleets (55.8%) — 453675 ships (40.5%)
- `waste_attack     `  2066 fleets (11.4%) —  34947 ships ( 3.1%)
- `waste_trajectory `  1821 fleets (10.1%) — 356887 ships (31.8%)
- `inflight         `   124 fleets ( 0.7%) —  36128 ships ( 3.2%)
- `unknown          `     0 fleets ( 0.0%) —      0 ships ( 0.0%)

Raw outcomes (debug):
- `reinforced_self` 10105
- `captured` 4001
- `bounced_enemy` 1894
- `vanished_in_space` 1821
- `alive_at_end` 124
- `arrived_but_lost` 124
- `bounced_neutral` 48
