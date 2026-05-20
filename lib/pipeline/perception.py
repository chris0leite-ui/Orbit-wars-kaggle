"""Stage 1 — Perception.

obs (raw kaggle observation dict-or-Struct) → TurnContext.

The reference implementation parses planets/fleets, infers num_seats,
builds the World + WorldModel snapshots. Bit-exact: closed-form only.
"""

from __future__ import annotations

from typing import Any, Optional

from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet

from lib.intent import World
from lib.pipeline.types import TurnContext
from lib.world_model import WorldModel


def _as_dict(obs):
    """Coerce an obs (dict or Struct) into a plain dict.

    Matches `lib.joint_solver.mpc._as_dict` so parity is preserved.
    """
    if isinstance(obs, dict):
        return obs
    return {k: getattr(obs, k) for k in dir(obs) if not k.startswith("_")}


def _num_seats(planets, fleets) -> int:
    """Infer num_seats from owner ids present in obs.

    Mirror of `lib.joint_solver.mpc._num_seats`.
    """
    owners = {int(p.owner) for p in planets if int(p.owner) >= 0}
    owners.update(int(f.owner) for f in fleets if int(f.owner) >= 0)
    if not owners:
        return 2
    return max(2, max(owners) + 1)


def perception_default(obs, configuration: Optional[Any] = None) -> TurnContext:
    """Reference Stage-1 implementation (parity with mpc.solve_turn)."""
    obs_d = _as_dict(obs)
    me = int(obs_d.get("player", 0))
    raw_planets = obs_d.get("planets", []) or []
    raw_fleets = obs_d.get("fleets", []) or []

    # Empty-obs short-circuit (mirrors mpc.solve_turn:147-154).
    if not raw_planets:
        return TurnContext(
            obs_d=obs_d, configuration=configuration,
            me=me, num_seats=2, step_now=int(obs_d.get("step", 0) or 0),
            omega=0.0,
            planets=[], fleets=[], my_planets=[], other_planets=[],
            world=None, model=None,  # type: ignore[arg-type]
            is_empty_obs=True,
        )

    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]
    my_planets = [p for p in planets if int(p.owner) == me]
    other_planets = [p for p in planets if int(p.owner) != me]
    num_seats = _num_seats(planets, fleets)
    step_now = int(obs_d.get("step", 0) or 0)

    # No-targets / no-sources short-circuit (mirrors mpc.solve_turn:160-167).
    if not my_planets or not other_planets:
        return TurnContext(
            obs_d=obs_d, configuration=configuration,
            me=me, num_seats=num_seats, step_now=step_now,
            omega=float(obs_d.get("angular_velocity", 0.0)),
            planets=planets, fleets=fleets,
            my_planets=my_planets, other_planets=other_planets,
            world=None, model=None,  # type: ignore[arg-type]
            is_no_targets=True,
        )

    world = World.from_obs(obs_d)
    model = WorldModel.from_world(world)
    omega = float(obs_d.get("angular_velocity", 0.0))

    return TurnContext(
        obs_d=obs_d, configuration=configuration,
        me=me, num_seats=num_seats, step_now=step_now,
        omega=omega,
        planets=planets, fleets=fleets,
        my_planets=my_planets, other_planets=other_planets,
        world=world, model=model,
    )
