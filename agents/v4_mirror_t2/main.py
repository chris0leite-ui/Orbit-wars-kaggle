"""v4_mirror_t2 — Tier 2: lag-compensated mirror + self-evident-error veto.

Inherits Tier 1's lag compensation. Adds: when opp's launch is
predicted to crash (sun / OOB / 200-step timeout with no hit), we
skip the mirror copy entirely. The opp wasted ships; we don't have
to waste ours.

Floor preservation: refusing to copy an action with predicted value ≤ 0
cannot worsen our position vs. pure mirror.

Reuses `lib.trajectory.predict_fleet_fate` (97.2% capture-probe
accuracy per audit/2026-05-11-capture-success-probe.json).

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
MAX_PRODUCTION = 5


def _obs_get(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _load_v3_fallback():
    global _V3_AGENT
    if _V3_AGENT is not None:
        return _V3_AGENT
    path = Path(__file__).resolve().parents[2] / "agents" / "v3_snipe" / "main.py"
    spec = importlib.util.spec_from_file_location("_mirror_t2_v3_fallback", path)
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


def _predict_opp_fate(fleet_raw, world: World):
    """Predict the FATE of an opp fleet — returns (outcome, hit_planet_or_None).

    Outcomes: 'target'/'planet' (hits a planet — mirror it),
              'sun'/'oob' (self-evident error — skip),
              'timeout' (ambiguous; treat as planet hit with max bump).
    """
    fx, fy = float(fleet_raw[2]), float(fleet_raw[3])
    angle = float(fleet_raw[4])
    ships = int(fleet_raw[6])
    synth_src = Planet(-1, -1, fx, fy, 0.0, ships, 0)
    any_planet = next(iter(world.planets_by_id.values()), None)
    if any_planet is None:
        return ("oob", None)
    fate = predict_fleet_fate(synth_src, any_planet, angle, ships, world)
    hit = world.planets_by_id.get(fate.hit_planet_id) if fate.hit_planet_id else None
    return (fate.outcome, hit)


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

        outcome, hit_planet = _predict_opp_fate(f_raw, world)

        # ─── Tier 2 self-evident-error veto ───────────────────────────────
        if outcome in ("sun", "oob"):
            # Opp's fleet is self-defeating. Skip mirror.
            continue

        # ─── Tier 1 lag-compensated size ──────────────────────────────────
        if hit_planet is not None:
            sigma_target_id = bij.get(hit_planet.id)
            sigma_target = world.planets_by_id.get(sigma_target_id) if sigma_target_id else None
            bump = int(sigma_target.production) if sigma_target is not None else MAX_PRODUCTION
        else:
            bump = MAX_PRODUCTION

        desired = int(f_raw[6]) + bump
        emit_ships = min(desired, avail)
        if emit_ships <= 0:
            continue
        actions.append([our_src, rotate_angle(float(f_raw[4])), emit_ships])
        garrison[our_src] = avail - emit_ships

    return actions
