"""Physics-validation gate for proposed launches.

The geometric primitives in `agents/trajectory_roi/main.py:_aim_and_eta`
(and consequently every agent in our experimental line that imports
them) compute aim angles but do NOT validate that the resulting fleet
trajectory is physically legal. Real failures: launches passing through
the sun at the board center, fleets going OOB, fleets hitting an
intervening planet before reaching the intended target.

Baseline.py uses `lib.trajectory.predict_fleet_fate` as a late drop-
filter for exactly this purpose (see lib/mechanism.py `sun_avoid`,
`path_clears_other_planets`, `oob_guard`). This module exposes the same
primitive for our experimental line.

Adapter note: `predict_fleet_fate` expects planets with `.x`/`.y`
attributes and a world with `.planets_by_id` (the `lib.world_model
._MiniWorld` shape). `lib.trajectory_layer.PlanetView` exposes
`.current_x`/`.current_y` instead. We wrap with lightweight duck-typed
objects here, mirroring how `lib.world_model.WorldModel.from_world`
adapts its inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

from lib.trajectory import predict_fleet_fate
from lib.trajectory_layer import World


@dataclass(frozen=True)
class _PlanetShim:
    """`predict_fleet_fate`-compatible view over a PlanetView."""
    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: float
    production: float


class _WorldShim:
    """`predict_fleet_fate`-compatible world (mirrors
    `lib.world_model._MiniWorld`)."""
    __slots__ = ("omega", "planets_by_id", "step")

    def __init__(self, world: World):
        self.omega = world.omega
        self.step = int(world.step)
        self.planets_by_id = {
            p.id: _PlanetShim(
                id=p.id, owner=p.owner,
                x=float(p.current_x), y=float(p.current_y),
                radius=float(p.radius),
                ships=float(p.ships), production=float(p.production),
            )
            for p in world.planets
        }


def launch_reaches_target(src, target, aim_angle: float, ships: int,
                            world: World) -> bool:
    """True iff a launch from `src` with the given aim/ships would
    actually land on `target` (outcome == 'target' AND the hit planet
    is in fact the intended one).

    `src` and `target` are `PlanetView` instances; this function
    handles the adapter conversion to `predict_fleet_fate`'s expected
    shape. Returns False on sun collision, OOB, intervening planet, or
    timeout."""
    wshim = _WorldShim(world)
    src_shim = wshim.planets_by_id[src.id]
    tgt_shim = wshim.planets_by_id[target.id]
    fate = predict_fleet_fate(src_shim, tgt_shim, aim_angle, ships, wshim)
    return fate.outcome == "target" and fate.hit_planet_id == target.id
