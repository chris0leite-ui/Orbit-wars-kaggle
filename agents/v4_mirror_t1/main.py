"""v4_mirror_t1 — Tier 1: lag-compensated mirror.

Tier 0 (pure mirror) was falsified at 0/8 vs v3_snipe. Root cause:
v3's `arrival_size` sizes fleets tightly at
`target.ships + target.production * eta + 1`. Mirror copies the same
size but arrives one step later, so the defender at σ(target) has
`+production` more ships when the mirror fleet arrives — mirror always
fails to capture by 1–5 ships per swap, cascading to wipeout.

Tier 1 fix: for each new opp fleet, identify its target via
`lib.trajectory.predict_fleet_fate` (full-trajectory ray-cast that
accounts for orbiting planets), look up σ(target).production, and bump
our mirror launch size by `prod` ships to cover the 1-step lag. If
the target can't be identified, fall back to MAX_PRODUCTION (=5) so
we still mirror — never skip a launch in Tier 1.

4P falls back to v3_snipe (no strict guarantee in n ≥ 3).

Plan reference: /root/.claude/plans/you-are-a-top-parallel-swan.md
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from lib.intent import World
from lib.mirror import (
    build_bijection,
    detect_num_players,
    diagonal_opponent,
    rotate_angle,
)
from lib.trajectory import predict_fleet_fate


_STATE: dict[int, dict] = {}
_V3_AGENT = None
MAX_PRODUCTION = 5  # env spec: production_range = [1, 5]


def _obs_get(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _load_v3_fallback():
    global _V3_AGENT
    if _V3_AGENT is not None:
        return _V3_AGENT
    path = Path(__file__).resolve().parents[2] / "agents" / "v3_snipe" / "main.py"
    spec = importlib.util.spec_from_file_location("_mirror_t1_v3_fallback", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _V3_AGENT = mod.agent
    return _V3_AGENT


def _reset_state(my_id: int, obs) -> dict:
    initial = _obs_get(obs, "initial_planets", []) or []
    planets = _obs_get(obs, "planets", []) or []
    n = detect_num_players(planets)
    st = {
        "num_players": n,
        "bijection": build_bijection(initial) if n == 2 else {},
        "opp_id": diagonal_opponent(my_id, n) if n in (2, 4) else None,
        "prev_fleet_ids": set(),
    }
    _STATE[my_id] = st
    return st


def _predict_opp_target(fleet_raw, world: World):
    """Identify which planet an opp fleet will hit.

    Uses `predict_fleet_fate` on a synthetic src = (fleet position, radius=0)
    so the ray-cast walks the full trajectory accounting for orbital motion.
    Returns the hit Planet or None.
    """
    fx, fy = float(fleet_raw[2]), float(fleet_raw[3])
    angle = float(fleet_raw[4])
    ships = int(fleet_raw[6])
    # Synthetic src — Planet tuple with radius 0 placed at fleet position.
    synth_src = Planet(-1, -1, fx, fy, 0.0, ships, 0)
    # Pick any planet as the "target hint" for the predictor; it ray-casts
    # everything and returns the FIRST hit regardless.
    any_planet = next(iter(world.planets_by_id.values()), None)
    if any_planet is None:
        return None
    fate = predict_fleet_fate(synth_src, any_planet, angle, ships, world)
    if fate.outcome in ("target", "planet") and fate.hit_planet_id is not None:
        return world.planets_by_id.get(fate.hit_planet_id)
    return None


def agent(obs):
    my_id = int(_obs_get(obs, "player", 0))
    step = int(_obs_get(obs, "step", 0))
    st = _STATE.get(my_id)
    if step == 0 or st is None:
        st = _reset_state(my_id, obs)

    if st["num_players"] != 2:
        return _load_v3_fallback()(obs)

    raw_planets = _obs_get(obs, "planets", []) or []
    raw_fleets = _obs_get(obs, "fleets", []) or []
    bij = st["bijection"]
    opp_id = st["opp_id"]
    prev_ids = st["prev_fleet_ids"]

    new_opp = [f for f in raw_fleets if f[0] not in prev_ids and f[1] == opp_id]
    st["prev_fleet_ids"] = {f[0] for f in raw_fleets}

    if not new_opp:
        return []

    # Build a World view for predict_fleet_fate (it consumes World).
    world = World.from_obs(obs)
    garrison: dict[int, int] = {p[0]: int(p[5]) for p in raw_planets if p[1] == my_id}

    actions: list[list] = []
    for f_raw in new_opp:
        opp_from = int(f_raw[5])
        our_src = bij.get(opp_from)
        if our_src is None:
            continue
        avail = garrison.get(our_src, 0)
        if avail <= 0:
            continue

        # Identify opp's target → lookup σ(target).production for lag bump.
        opp_target = _predict_opp_target(f_raw, world)
        if opp_target is not None:
            sigma_target_id = bij.get(opp_target.id)
            sigma_target = world.planets_by_id.get(sigma_target_id) if sigma_target_id else None
            bump = int(sigma_target.production) if sigma_target is not None else MAX_PRODUCTION
        else:
            # Couldn't pin a target (rare — most reachable fleets hit
            # something within the 200-step horizon). Use the safe upper
            # bound so we still mirror.
            bump = MAX_PRODUCTION

        desired = int(f_raw[6]) + bump
        emit_ships = min(desired, avail)
        if emit_ships <= 0:
            continue
        actions.append([our_src, rotate_angle(float(f_raw[4])), emit_ships])
        garrison[our_src] = avail - emit_ships

    return actions
