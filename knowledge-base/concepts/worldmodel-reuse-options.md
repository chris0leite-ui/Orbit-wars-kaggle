# WorldModel reuse — options for tightening composite head wallclock

> Filed 2026-05-17 PM after the composite-head A/B left max-turn-ms at
> 1196-1580ms (env hard cap 1000ms). The #1+#2 timing fixes
> (`affordable_validate_cap` leaf probe + adaptive WorldModel horizon)
> shipped in this session; this is the *next* tier of optimization,
> deferred for a follow-up session.

## Problem

`lib.value_heads.composite_capture_value` builds `World.from_obs(obs)`
and `WorldModel.from_world(world, horizon=...)` on every call. With
~40 `build_idle_baseline` iterations + N_VALIDATE candidate leaf evals
per turn, that's 50-100 builds × ~1.5-2ms = **100-200ms per turn** of
redundant work. No quality cost from eliminating — both objects are
deterministic functions of obs.

Combined with the existing #1+#2 fixes:
- p95 ≈ 720 ms (target <800)
- max ≈ 1196-1580 ms (target <1000)

WorldModel reuse is the next lever to drop max under 1000ms
deterministically.

## Three options, by effort

### Option B — thread `world` through `favor_fn` API (~2-3 h)

Chooser builds `World` once per leaf, passes it as a kwarg to
`favor_fn`. `composite_capture_value` uses the passed-in world if
given, else builds.

**Touched files:**
- `agents/baseline/value.py` — `favor`, `favor_composite`, `favor_hybrid`
  signatures (add `world=None`).
- `agents/baseline/chooser.py` — `build_idle_baseline`, `score_action`
  build World per snap, pass to favor_fn.
- `lib/value_heads.py:composite_capture_value` — accept passed-in
  world.

**Wins:** ~35-70 ms per turn (eliminates the World.from_obs build,
keeps WorldModel build).

**Risks:** API churn across all favor variants. Low.

**LOC estimate:** 30-50.

### Option D — `Snap.world` as cached property (~4-5 h, RECOMMENDED)

`favor_fn` takes the snap directly; snap exposes `.world` as a
`@cached_property`. World is computed lazily once per snap and
reused everywhere on that snap.

**Touched files:**
- `lib/fast_sim.py` — add `@cached_property` for `.world` on the Snap
  class (and possibly `.world_model` too).
- `agents/baseline/value.py` — favor variants take `snap` instead of
  raw `obs`; pull obs / world from snap.
- `agents/baseline/chooser.py` — call `favor_fn(snap, ...)` instead
  of `favor_fn(snap.state[me].observation, ...)`.

**Wins:** ~70-130 ms per turn (World + reduced inner work).

**Risks:** Cached-property on a struct that may be cloned/stepped
needs care — `fs_clone` and `fs_step in_place=True` must NOT leak a
stale world from the parent. Probably means dropping the cache on
clone/step.

**LOC estimate:** 50-80.

### Option E — composite reads snap directly, skip `World` entirely (~6-8 h)

Rewrite `composite_capture_value` to operate on `snap.state` (the
fast_sim representation) and replicate the parts of
`simulate_planet_timeline` it actually needs.

**Touched files:**
- `lib/value_heads.py` — new `composite_capture_value_snap(snap, my_id)`
  that doesn't build World or WorldModel.
- `agents/baseline/value.py` — `favor_composite` calls the snap variant.
- New helper to predict `owner_at(planet_id, eta)` and
  `ships_at(planet_id, eta)` from snap + the in-flight fleet list,
  without building a full WorldModel.

**Wins:** ~120-200 ms per turn (eliminates BOTH World and WorldModel
builds for the common path).

**Risks:** Re-implementing simulation logic that's already in
`lib/world_model.simulate_planet_timeline` risks subtle drift from
the established (tested) version. Bigger surface for bugs.

**LOC estimate:** 100-150 + replication of timeline math.

## Expected impact on max turn-ms

Starting from current p95 ≈ 720, max ≈ 1196-1580:

| option | max → target | clears 1000ms cap? |
|---|---|---|
| B | ~1100-1500 | sometimes |
| **D** | **~1050-1450** | **usually (with #1+#2 deadline-headroom enforcement)** |
| E | ~950-1300 | most reliably |

D is the sweet spot: meaningfully tighter than B, much smaller scope
and risk than E.

## When to do this

- **Before** PI signs off on a live submission if the current max-turn
  risk (engine drops over-budget actions) is unacceptable for the
  ladder dynamics.
- **After** a session where there's a half-day budget for a focused
  perf refactor. The #1+#2 fixes are already in place; this is the
  next-tier improvement, not a blocker.

## Pre-conditions / things to know before starting

- Snap object identity: a `fs_step` with `in_place=True` mutates the
  snap rather than returning a new one. The cached_property
  invalidation contract matters.
- `WorldModel.from_world` is the dominant cost inside composite —
  ~50% of the per-call ms. Option B leaves WorldModel building per
  call; only D / E address it.
- All three options are pure perf changes — no winrate Δ expected.
  An A/B at n=32 vs v15 after the change should match the prior
  62.5-75% range.

## Cross-references

- `audit/replays/composite-ab-2026-05-17.md` — A/B numbers that
  motivated this work (max 1196-1580ms).
- `audit/friction.md` tag `composite-head-wallclock-over-1000ms-on-
  heavy-turns` — what's already shipped (#1+#2 fixes).
- `agents/baseline/chooser.py:78-105` — `affordable_validate_cap` +
  pre-bail headroom (the existing fix; D extends this layer).
- `lib/value_heads.py:151-237` — `composite_capture_value` (the
  edit target).
- `lib/fast_sim.py` — Snap class definition (D/E touch this).
- `lib/world_model.py:42-298` — `fleet_target_planet`,
  `_comet_paths_by_id`, `simulate_planet_timeline` (E replicates
  this last one).
