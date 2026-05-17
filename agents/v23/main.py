"""v23 — opening proposer overlay on top of the clean baseline (v15 parity).

Pivot motivation. Five chooser-axis variants (v20/v21/v21_a/v21_ae/
v21_solo/v22) all failed at n=32 vs v15. The v15 chooser is a finely-
tuned local optimum; any single-component modification breaks calibration.
v23 attacks a different axis: it does NOT touch the chooser at all.
Instead, it overlays a specialized opening policy for turns 0..15 of
2P games, then hands off to the unmodified baseline (= v15 parity) at
turn 16+.

Empirical gap (audit/2026-05-14-opening-atlas.json + v15 live replays
in audit/live-episodes/52710995/):
- v15 launches 2.0 fleets in turns 0-15 (0.13/turn).
- Top-10 leaders launch 7-10 fleets in turns 0-15 (0.47-0.67/turn).
- Median first-launch step: top-10 = 4.1, v15-class = 10.5.
- Six wasted opening turns forfeit ~6 prod-units × ~30 ships of board
  control to the front-loader.

Why v15 is bad at the opening: its leaf `_favor` weights production at
`pv_horizon(γ=0.99) ≈ 99` units per prod-1 capture, which loses to F1's
ship-balance term for early small-fleet launches. The chooser values
"wait and hoard" over "grab and produce." This is structural — there's
no "front-load" coefficient anywhere in v15's evaluation.

The fix: bypass v15's chooser for the opening window, run `propose_
opening_missions` (lib/missions/opening.py — already exists, was wired
into v7_1's pipeline, validated against top-10 fingerprint corpus) at
extended window=15, settle_plan, realize, emit. Hand off to baseline
at step 16+ unchanged.

Why this isn't the same as the falsified `cluster-conditional-opening-
overlay` (game-strategy-eda-roatN, 2026-05-14): that overlay was board-
classified via a buggy KMeans (paraphrased `is_orbiting` formula). Hard
nearest-centroid on soft clusters (silhouette ≈0.17) routed marginal
boards to wrong templates. v23 is board-partition-free: ONE opening
policy, always on for turns 0..15 of 2P games. No classifier.
"""

from __future__ import annotations

import os

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

from lib.fast_sim import from_obs as fs_from_obs
from lib.intent import World, realize as intent_realize
from lib.mechanism import DEFAULT_MECHANISMS
from lib.missions.opening import propose_opening_missions
from lib.planner import settle_plan
from lib.world_model import WorldModel

from agents.baseline import chooser, proposer
from agents.baseline.main import _as_dict, _num_seats, _wallclock_ms, _gamma

# ---------------------------------------------------------------------------
# Opening-overlay knobs
# ---------------------------------------------------------------------------

# Inclusive — opening fires for steps 0..15 in 2P games. Window chosen
# from the audit gap (v15 still behind top-10 through ~turn 15; closes
# by turn 30). Larger window = more game time governed by the simpler
# proposer. Default 15; tune to 10 if the first n=32 panel fails by a
# narrow margin (per the plan's falsification path).
V23_OPENING_WINDOW = int(os.environ.get("V23_OPENING_WINDOW", "15"))

# Opening proposer is 2P-only — 4P opening dynamics weren't profiled in
# the same audit, and the proposer's `(remaining_steps)^1.5 / distance`
# scoring may not hold when there are 3 opps to position against.
V23_OPENING_2P_ONLY = True


def agent(obs, configuration=None):
    """v23 = opening short-circuit (turns 0..V23_OPENING_WINDOW, 2P only)
    + baseline chooser (= v15 parity) for everything else."""
    obs_d = _as_dict(obs)
    me = int(obs_d.get("player", 0))
    raw_planets = obs_d.get("planets", []) or []
    raw_fleets = obs_d.get("fleets", []) or []
    if not raw_planets:
        return []

    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]
    my_planets = [p for p in planets if int(p.owner) == me]
    other_planets = [p for p in planets if int(p.owner) != me]
    if not my_planets or not other_planets:
        return []

    world = World.from_obs(obs_d)
    model = WorldModel.from_world(world)
    omega = float(obs_d.get("angular_velocity", 0.0))
    num_seats = _num_seats(planets, fleets)

    # v23 — opening proposer short-circuit. 2P games only; turns
    # 0..V23_OPENING_WINDOW inclusive. Bypasses baseline's full two-
    # stage rollout for the opening and emits the proposer's front-
    # loaded landgrab plan directly. Hands off to baseline at turn 16+.
    step_now = int(obs_d.get("step", 0))
    if (
        (not V23_OPENING_2P_ONLY or num_seats == 2)
        and step_now <= V23_OPENING_WINDOW
    ):
        opening_missions = propose_opening_missions(
            world, model, window=V23_OPENING_WINDOW,
        )
        if opening_missions:
            intents = settle_plan(opening_missions, world, model)
            actions = intent_realize(
                intents, obs_d,
                mechanisms=DEFAULT_MECHANISMS, model=model,
            )
            if actions:
                return actions
        # Else: proposer returned [] (no sources with > MIN_LAUNCH_GARRISON
        # ships OR no neutral targets) OR realize filtered all intents
        # via the mechanism pipeline. Fall through to baseline rather
        # than emit nothing — baseline's chooser may still find a
        # productive defensive launch (e.g. reinforce against an
        # in-flight enemy spawned mid-opening).

    # Baseline path — identical to agents/baseline/main.py from here on.
    gamma = _gamma()
    wallclock_ms = _wallclock_ms()

    threatened_mine = [
        p for p in my_planets
        if model.time_to_enemy_threat(int(p.id), me, world) is not None
    ]
    target_pool = other_planets + threatened_mine

    snap_base = fs_from_obs(obs, num_seats=num_seats)

    baseline_favors = chooser.build_idle_baseline(
        snap_base, me, num_seats, proposer.MAX_HORIZON, gamma,
    )

    prerank = proposer.propose(
        my_planets, target_pool, world, model, me, omega,
        baseline_len=len(baseline_favors),
    )

    return chooser.choose(
        snap_base, prerank, baseline_favors,
        me, num_seats, wallclock_ms,
        proposer.MIN_HORIZON, proposer.MAX_HORIZON, gamma,
    )
