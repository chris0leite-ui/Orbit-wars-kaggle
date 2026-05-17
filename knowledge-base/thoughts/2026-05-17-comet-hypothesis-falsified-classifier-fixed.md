# PI's comet hypothesis falsified — classifier bug masked planet hits

Date: 2026-05-17 (afternoon)
Branch: claude/audit-workflow-performance-btjeK
Source: `audit/replays/replay-mine-2026-05-17.{json,md}` (re-run)
Code: `scripts/episode_postmortem.py:_swept_pair_planet_hit`
Supersedes: numbers in `2026-05-17-replay-mine-baseline-v15-fleet-waste.md`

## What happened

The previous replay-mine showed v15 with `vanished_in_space = 8.8%` of
fleets (838 / 9,507) — the dominant trajectory-waste bucket. PI's
hypothesis: most are killed by comet collisions (engine swept-pair
check against comets at orbit_wars.py:593, since comets ARE planets
via `comet_planet_ids`).

I extended `attribute_fleets` with a swept-pair check using the
engine's own `swept_pair_hit` primitive and `COMET_RADIUS=1.0`.
Re-running on the 92 v15 episodes:

| bucket | before | after |
|---|---:|---:|
| `win` (captured) | 42.8% | **47.4%** |
| `defense` | 32.8% | **35.2%** |
| `waste_attack` (bounced) | 14.7% | **15.7%** |
| `waste_comet` (new) | — | **0.1%** |
| `waste_trajectory` | 9.0% | **0.9%** |
| `inflight` | 0.7% | 0.7% |

Only **12 of 9,507 fleets** are actual comet collisions (0.1%). The
remaining 8.8% migrated to `win` / `defense` / `waste_attack` — they
were *fleets actually hitting planets*, but the original
`best_d < 5.0` check at `attribute_fleets:290` measured distance from
the fleet's last-seen position to the planet's position in `obs_vanish`
— and the planet had **orbited out of range by then**.

## The classifier bug

Original predicate (line 290):
```python
for p in obs_vanish.get("planets", []):
    d = math.hypot(p[2] - last_entry[2], p[3] - last_entry[3])
    if d < 5.0:
        # hit a planet
```

- Used **fleet's OLD position** vs **planet's NEW position** — these
  are one tick apart, plenty for an orbital planet to move >5 units.
- Single-point distance, not a swept segment — missed glancing hits.
- Did not account for the engine's use of `swept_pair_hit` semantics
  (continuous collision against the segment, not point-to-point).

## The fix

`_swept_pair_planet_hit(obs_prev, obs_vanish, last_entry)` runs the
engine's exact primitive against every planet in `obs_prev`, with:
- Planet old position from obs_prev
- Planet new position from obs_vanish (or from the comet path for comets
  that expired during the killing tick)
- Fleet old/new from `_fleet_new_pos(last_entry)`

It returns `(planet_id, is_comet)` for the hit. The
`vanished_in_space` else-branch reuses ownership-flip logic to
classify orbital hits as captured / reinforced_self / bounced_*.

## What this changes about the plan

1. **v15 is meaningfully better than measured.** The "24% wasted"
   number was a classifier artifact. The real waste is **16.7%**,
   essentially all `waste_attack` (bounced).
2. **PI's comet hypothesis is essentially falsified.** Comets cause
   ~0.1% of fleet losses, not the 9% suggested by `vanished_in_space`.
3. **Pivot #3 (comet-collision penalty in composite_capture_value)
   is no longer justified.** Building it would target 0.1% of fleets.
   Cancelled from this session's plan.
4. **Pivot #2 (composite head A/B) still motivated.** `composite_capture_value`
   has a waste_penalty term that specifically targets `bounced_*` (now
   the dominant waste at 15.7%). A/B running now via
   `BASELINE_VALUE_HEAD=composite fast.py eval --vs-panel default
   --require-h2h agents/baseline --max-seeds 32`.
5. **Replay-mine pipeline is now meaningfully more accurate** — the
   swept-pair fix lifts win-rate measurement by ~4.6pp on v15. Any
   future agent's replay-mine output will be directly comparable.

## Reading

The PI's intuition that "vanished in space" is suspicious was correct.
The mechanism wasn't comets but the classifier's static-distance
shortcut. The fix is structural (use the engine's primitive, not a
proxy) and makes every subsequent replay-mine measurement more
trustworthy. Net: 0 strategy change, +1 measurement honesty.

## How PI grades this

```bash
python scripts/replay_mine.py 52710995
```

Expected: `win >45%`, `defense >34%`, `waste_attack ~16%`,
`waste_comet ~0.1%`, `waste_trajectory <2%`. Matches the table above.
