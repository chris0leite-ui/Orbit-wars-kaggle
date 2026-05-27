"""minimal — stripped-down v15-style agent.

Pipeline (per turn):
  1. parse obs (dict-or-attr), detect 2P/4P from max owner id
  2. WorldModel.from_world → predict per-planet threat ETAs
  3. proposer.propose       attack candidates (capture-size) plus
                            reinforce candidates for threatened-mine planets
  4. chooser.choose         idle-baseline + Δ rollout against reactive opp,
                            greedy per-source emit (Δ > 0 only)

Designed as a foundation to swap pieces against (value / proposer /
chooser / opp_model). Same agent(obs, configuration) entry contract
and same lib/ primitives as agents/baseline/.
"""

from __future__ import annotations

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

from lib.fast_sim import from_obs as fs_from_obs
from lib.intent import World
from lib.world_model import WorldModel

from agents.minimal import chooser, proposer


def _as_dict(obs) -> dict:
    if isinstance(obs, dict):
        return obs
    return {
        "player": getattr(obs, "player", 0),
        "step": getattr(obs, "step", 0),
        "planets": list(getattr(obs, "planets", []) or []),
        "fleets": list(getattr(obs, "fleets", []) or []),
        "comets": list(getattr(obs, "comets", []) or []),
        "comet_planet_ids": list(getattr(obs, "comet_planet_ids", []) or []),
        "angular_velocity": float(getattr(obs, "angular_velocity", 0.0)),
    }


def _num_seats(planets, fleets) -> int:
    m = -1
    for x in planets:
        if int(x[1]) > m:
            m = int(x[1])
    for x in fleets:
        if int(x[1]) > m:
            m = int(x[1])
    return 4 if m >= 2 else 2


def agent(obs, configuration=None):
    obs_d = _as_dict(obs)
    me = int(obs_d.get("player", 0))
    raw_p = obs_d.get("planets") or []
    raw_f = obs_d.get("fleets") or []
    if not raw_p:
        return []

    planets = [Planet(*p) for p in raw_p]
    fleets = [Fleet(*f) for f in raw_f]
    my_planets = [p for p in planets if int(p.owner) == me]
    enemy_planets = [p for p in planets if int(p.owner) != me]
    if not my_planets or not enemy_planets:
        return []

    num_seats = _num_seats(planets, fleets)

    world = World.from_obs(obs_d)
    model = WorldModel.from_world(world)
    threatened_mine = [
        p for p in my_planets
        if model.time_to_enemy_threat(int(p.id), me, world) is not None
    ]

    cands = proposer.propose(
        my_planets, enemy_planets,
        threatened_mine=threatened_mine, model=model, me=me, world=world,
    )
    if not cands:
        return []

    snap = fs_from_obs(obs, num_seats=num_seats)
    return chooser.choose(snap, cands, me, num_seats)
