"""baseline — clean modular re-implementation of v15 (live champion μ=1115.5).

Pipeline (per turn):
  1. proposer.propose       enumerate fire-now + multi-wait grid, cheap-rank,
                            dedup by (src, tgt, wait_band).
  2. chooser.build_idle_baseline   precompute favor under (me-idle, opp-reactive).
  3. chooser.choose         validate top candidates with fast_sim K-step rollout,
                            emit greedy non-dogpile moves.

Knobs (env var overrides, all optional):
  BASELINE_GAMMA              PV-discount γ for favor() and cheap-rank.   default 0.99
  BASELINE_WALLCLOCK_MS       per-turn validate budget (env actTimeout=1000).
                                                                          default 600
  BASELINE_LAMBDA_ALPHA       weight on the static per-class priority prior.
                              0 disables; design default 3.0.             default 3.0
  BASELINE_LAMBDA_GAP         weight on the opponent in-flight posterior gap.
                              0 disables; design default 2.0.             default 2.0
  BASELINE_ROI_ENABLED        when 1/true, divide positive cheap_marginal_value
                              by (ships + eta + roi_denom_floor) -- v3_snipe
                              additive cost denominator.                  default 1
  BASELINE_ROI_DENOM_FLOOR    additive floor on the ROI cost denominator. default 1.0
  ORBIT_WARS_PARITY_WALLCLOCK_MS    bundle-parity override (very large
                                    value disables mid-loop deadline bail
                                    so the agent is a pure function of obs).
"""

from __future__ import annotations

import os

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

from lib.fast_sim import from_obs as fs_from_obs
from lib.intent import World
from lib.world_model import WorldModel
from lib import priority_prior

from agents.baseline import chooser, proposer


_PARITY_ENV_VAR = "ORBIT_WARS_PARITY_WALLCLOCK_MS"


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
    max_owner = -1
    for p in planets:
        if int(p.owner) > max_owner:
            max_owner = int(p.owner)
    for f in fleets:
        if int(f.owner) > max_owner:
            max_owner = int(f.owner)
    return 4 if max_owner >= 2 else 2


def _wallclock_ms() -> float:
    override = os.environ.get(_PARITY_ENV_VAR)
    if override:
        try:
            return float(override)
        except ValueError:
            pass
    try:
        return float(os.environ.get("BASELINE_WALLCLOCK_MS", chooser.WALLCLOCK_BUDGET_MS))
    except ValueError:
        return chooser.WALLCLOCK_BUDGET_MS


def _gamma() -> float:
    try:
        return float(os.environ.get("BASELINE_GAMMA", 0.99))
    except ValueError:
        return 0.99


def _lambda_alpha() -> float:
    try:
        return float(os.environ.get("BASELINE_LAMBDA_ALPHA", 3.0))
    except ValueError:
        return 3.0


def _lambda_gap() -> float:
    try:
        return float(os.environ.get("BASELINE_LAMBDA_GAP", 2.0))
    except ValueError:
        return 2.0


def _roi_enabled() -> bool:
    val = os.environ.get("BASELINE_ROI_ENABLED")
    if val is None:
        return True
    return val.strip().lower() not in ("0", "false", "no", "off", "")


def _roi_denom_floor() -> float:
    try:
        return float(os.environ.get("BASELINE_ROI_DENOM_FLOOR", 1.0))
    except ValueError:
        return 1.0


def agent(obs, configuration=None):
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
    gamma = _gamma()
    wallclock_ms = _wallclock_ms()

    lam_a = _lambda_alpha()
    lam_g = _lambda_gap()
    class_of = priority_prior.compute_class_of(raw_planets)
    opp_share = priority_prior.compute_opp_share_in_flight(model, me, class_of)
    priority_dict = priority_prior.priority_by_planet(
        class_of, opp_share, lam_a, lam_g,
    )
    roi_on = _roi_enabled()
    roi_floor = _roi_denom_floor()

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
        priority_by_planet=priority_dict,
        roi_enabled=roi_on,
        roi_denom_floor=roi_floor,
    )

    return chooser.choose(
        snap_base, prerank, baseline_favors,
        me, num_seats, wallclock_ms,
        proposer.MIN_HORIZON, proposer.MAX_HORIZON, gamma,
    )
