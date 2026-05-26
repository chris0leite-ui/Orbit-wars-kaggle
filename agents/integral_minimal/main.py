"""integral_minimal — closed-form horizon-weighted ROI, no rollout, no chooser.

PI 2026-05-26: the 5,400-LoC baseline_integral package loses to the 100-line
simple/roi heuristic on the archetype bench. Build the minimum-viable agent
from first principles using only closed-form physics; benchmark on the same
panel. If this beats ROI, the proposer/chooser/rollout/leaf stack is provably
overhead.

Per turn, for each owned planet:
    for each non-mine target:
        cost  = target.ships + 1               # exact closed-form capture cost
        aim   = aim_orbiting / aim_comet(...)  # closed-form intercept angle
        fate  = predict_fleet_fate(...)        # closed-form physics filter
        if fate.outcome != "target": skip
        score = π_target · max(0, T - t - eta) · opp_bonus / cost
    fire at the best positive-EV target

The score is the literal "terminal-ship-integral contribution per ship spent":
  - π · (T - t - eta) = future ship production captured by this action
  - / cost            = per-ship efficiency
  - · opp_bonus       = 2× urgency for opponent-owned targets (derived earlier:
                          their garrison grows, so delay costs both their gain
                          AND our delayed production)

No rollout. No leaf value function. No env-var feature flags except T_END
and the opponent bonus. Pure function of obs.
"""
from __future__ import annotations

import os

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet  # noqa: F401

from lib.aim import aim_orbiting, aim_comet
from lib.intent import World
from lib.trajectory import predict_fleet_fate
from lib.world_model import _comet_paths_by_id

EPISODE_STEPS_DEFAULT = 500
OPP_BONUS_DEFAULT = 2.0


def _as_dict(obs):
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


def agent(obs, configuration=None):
    obs_d = _as_dict(obs)
    me = int(obs_d.get("player", 0))
    t = int(obs_d.get("step", 0))
    if not obs_d.get("planets"):
        return []

    world = World.from_obs(obs_d)
    comet_paths = _comet_paths_by_id(world) if world.comet_ids else {}
    t_end = int(os.environ.get("INTEGRAL_T_END", str(EPISODE_STEPS_DEFAULT)))
    opp_bonus_val = float(os.environ.get("INTEGRAL_OPP_BONUS",
                                          str(OPP_BONUS_DEFAULT)))

    actions = []
    for src_id, src in world.planets_by_id.items():
        if int(src.owner) != me or int(src.ships) < 1:
            continue

        best_score = 0.0
        best_action = None
        for tgt_id, tgt in world.planets_by_id.items():
            if tgt_id == src_id or int(tgt.owner) == me:
                continue
            cost = max(1, int(tgt.ships) + 1)
            if cost > int(src.ships):
                continue

            tgt_tuple = [tgt.id, tgt.owner, tgt.x, tgt.y,
                         tgt.radius, tgt.ships, tgt.production]
            if int(tgt_id) in world.comet_ids and int(tgt_id) in comet_paths:
                cpath, cidx = comet_paths[int(tgt_id)]
                aim = aim_comet((src.x, src.y), src.radius, tgt_tuple,
                                tgt.radius, cost, cpath, cidx)
            else:
                aim = aim_orbiting((src.x, src.y), src.radius, tgt_tuple,
                                    tgt.radius, cost, world.omega)
            if aim is None:
                continue
            angle, _arrival_xy, _eta_hint = aim

            fate = predict_fleet_fate(src, tgt, angle, cost, world)
            if fate.outcome != "target":
                continue

            time_remaining = max(0, t_end - t - int(fate.step))
            opp_bonus = opp_bonus_val if int(tgt.owner) >= 0 else 1.0
            score = float(tgt.production) * time_remaining * opp_bonus / cost
            if score > best_score:
                best_score = score
                best_action = [int(src.id), float(angle), int(cost)]

        if best_action is not None:
            actions.append(best_action)

    return actions
