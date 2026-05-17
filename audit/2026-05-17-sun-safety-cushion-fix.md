# lib.trajectory SUN_SAFETY=0.5 → 0 fix (2026-05-17)

> Filed at the end of the Direction A trajectory iteration. Documents
> the bug we found while diagnosing v4's 3pp filter-on-vs-off gap, and
> the fix's measured impact on production-adjacent code.

## The bug

`lib/trajectory.py:49` (pre-fix):
```python
SUN_SAFETY = 0.5
```

Used by `predict_fleet_fate` at line 120 (pre-fix):
```python
if sun_d < SUN_RADIUS + SUN_SAFETY:
    return FleetFate("sun", None, step + 1)
```

The engine's check at `orbit_wars.py:607`:
```python
if point_to_segment_distance((CENTER, CENTER), old_pos, new_pos) < SUN_RADIUS:
    fleets_to_remove.append(fleet)
```

**Bare `< SUN_RADIUS`.** No safety margin. predict_fleet_fate
rejected every trajectory whose minimum sun-distance fell in
`[SUN_RADIUS, SUN_RADIUS + 0.5)` — exactly the band the engine
would have ACCEPTED.

The 0.5 was filed 2026-05-11 as a "float drift cushion." Empirically
the cushion was creating systematic false-rejects, not preventing
drift errors.

## Where the false-rejects bit production code

Three consumers of `predict_fleet_fate`:

1. **`lib.mechanism.sun_avoid`** — drops snipe/reinforce intents.
   In `v3_snipe` agent (still in lib/missions/snipe.py production
   path), this was dropping legal launches whenever a path passed
   within 0.5 units of the safety boundary.
2. **`lib.mechanism.path_clears_other_planets`** — drops intents
   whose path crosses a non-target planet. Used by the same
   missions stack.
3. **`agents/baseline/proposer.py:PROPOSER_TRAJECTORY_FILTER`** —
   Option 1 admissibility prefilter. Was env-var-gated (default off).

## Measured impact

### Option 1 prefilter vs v15 (`fast.py eval`, n=64, --workers 6)

| variant | wins | rate | Wlo | verdict |
|---|---:|---:|---:|---|
| Composite_a2 alone (current ship) | 40/64 | 62.5% | 0.503 | INCONCL |
| Option 1, PRE-fix `SUN_SAFETY=0.5` | 36/64 | 56.2% | 0.441 | INCONCL |
| **Option 1, POST-fix `SUN_SAFETY=0`** | **42/64** | **65.6%** | **0.534** | **INCONCL** |

**+9.4pp** on the same A/B (56.2 → 65.6). Roughly at parity with
composite_a2 alone, with the deterministic 0% sun/oob/comet failure
mode guarantees on top.

### Trajectory chooser v4 same-n A/B (incidental)

At n=32, hybrid leaf, 1-per-src emit:
- filter ON, pre-fix: 11/32 = 34.4%
- filter ON, post-fix: 12/32 = 37.5% (+1pp; within noise at n=32)
- filter OFF: 14/32 = 43.8%

Smaller lift here because v4 has other issues (joint-action /
sequence gaps the architectural reframe couldn't close).

## The fix

```diff
-SUN_SAFETY = 0.5
+SUN_SAFETY = 0.0
```

Plus updated `tests/test_mech_sun_avoid.py` — `test_sun_avoid_drops_
path_grazing_safety_margin` (which pinned the buggy 0.5 cushion as
"correct" behavior) renamed and re-asserted to match engine truth.

Committed at `29e0d27`.

## Pattern recognition (3rd recurrence)

This is the same class as `helper-reimplemented-inline-silently-
wrong` (audit/friction.md 2026-05-14):
- A primitive is REimplemented or WRAPPED for a new context.
- The re-implementation has a near-correct predicate.
- The divergence (off-by-one, strict-vs-nonstrict, extra margin) only
  matters at the boundary — and the boundary is what the agent gets
  asked about most.
- The bug ships silently because all the easy cases work.

Promotion candidate filed in friction.md.

## Default-on Option 1

After the fix, `PROPOSER_TRAJECTORY_FILTER` is **default-on**:
- Set the env var to `off` to bypass (testing / parity).
- Production behaviour: prefilter eliminates deterministic
  sun/oob/expired-comet/comet-collision failure modes; minor speedup
  on candidate evaluation (fewer doomed candidates surface to the
  K-step rollout).

Live submission `52744856` (composite_a2_hybrid) doesn't use the
prefilter (was env-off when bundled). The next push will include
it automatically.
