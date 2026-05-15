"""Foundation `Predictor` API — closed-form analytical predictions.

Two performance tiers:

- **O(1) per query** — position of one planet / comet / fleet at any
  future relative turn `t`, via closed-form orbital projection
  (`lib.orbit.predict_relative` semantics) or comet path-array
  lookup. No simulation, no collision check.
- **O(horizon) per query** — per-planet timeline `(owner, ships)`
  including the effect of hypothetical launches, via the arrival-
  ledger forward simulation in `lib.world_model`.

The cheap tier covers the user's `"when will planet/comet/fleet be
where?"` ask. The timeline tier covers `"how many ships at planet B
when an enemy fleet arrives?"` and `"if I launch X ships at planet
A, what happens?"`.

For full physics including collisions and combat (rather than the
analytical projection here), use `lib.foundation.jax_engine.step` /
`rollout_python` to forward-simulate and read the resulting state.

Pytree-immutable contract: the input `state` is never mutated. The
predictor lazily constructs scalar `Planet` / `Fleet` views on first
access; subsequent queries reuse those caches.

Step 6 of the plan; Step 9 extends `arrival_ledger(hypothetical=...)`
to non-zero `launch_turn` via JAX-engine forward simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet

from lib.fleet import speed as fleet_speed
from lib.foundation.actions import ActionSpec
from lib.foundation.memory import EmptyMemory, Memory
from lib.game.jax.jax_types import GameState, MAX_FLEETS, MAX_PLANETS
from lib.geometry import CENTER, ROTATION_RADIUS_LIMIT
from lib.world_model import (
    build_arrival_ledger,
    simulate_planet_timeline,
    state_at_timeline,
)


@dataclass(frozen=True)
class Arrival:
    """One arrival at a planet, as carried in the ledger.

    Wrapper around the `(eta, owner, ships)` tuple that
    `lib.world_model.build_arrival_ledger` returns; lets callers
    write `arr.eta` instead of `arr[0]`.
    """

    eta: int
    owner: int
    ships: int

    @classmethod
    def from_tuple(cls, t: tuple[int, int, int]) -> "Arrival":
        return cls(eta=int(t[0]), owner=int(t[1]), ships=int(t[2]))


class Predictor:
    """Analytical predictor for game state at future relative turns.

    Construct once per agent turn; query repeatedly. The closed-form
    queries are pure functions of the input state — no mutation
    possible.
    """

    def __init__(self, state: GameState, memory: Optional[Memory] = None):
        self._state = state
        self._memory = memory or EmptyMemory()
        self._omega = float(state.angular_velocity)
        self._planets_cache: Optional[list[Planet]] = None
        self._fleets_cache: Optional[list[Fleet]] = None

    # -- Lazy scalar adapters ---------------------------------------------

    def _planets_list(self) -> list[Planet]:
        if self._planets_cache is None:
            s = self._state
            alive = np.asarray(s.planets_alive)
            ids = np.asarray(s.planets_id)
            owner = np.asarray(s.planets_owner)
            x = np.asarray(s.planets_x)
            y = np.asarray(s.planets_y)
            radius = np.asarray(s.planets_radius)
            ships = np.asarray(s.planets_ships)
            prod = np.asarray(s.planets_prod)
            self._planets_cache = [
                Planet(
                    id=int(ids[i]),
                    owner=int(owner[i]),
                    x=float(x[i]),
                    y=float(y[i]),
                    radius=float(radius[i]),
                    ships=int(ships[i]),
                    production=int(prod[i]),
                )
                for i in range(MAX_PLANETS)
                if bool(alive[i])
            ]
        return self._planets_cache

    def _fleets_list(self) -> list[Fleet]:
        if self._fleets_cache is None:
            s = self._state
            alive = np.asarray(s.fleets_alive)
            ids = np.asarray(s.fleets_id)
            owner = np.asarray(s.fleets_owner)
            x = np.asarray(s.fleets_x)
            y = np.asarray(s.fleets_y)
            angle = np.asarray(s.fleets_angle)
            frm = np.asarray(s.fleets_from_planet)
            ships = np.asarray(s.fleets_ships)
            self._fleets_cache = [
                Fleet(
                    id=int(ids[i]),
                    owner=int(owner[i]),
                    x=float(x[i]),
                    y=float(y[i]),
                    angle=float(angle[i]),
                    from_planet_id=int(frm[i]),
                    ships=int(ships[i]),
                )
                for i in range(MAX_FLEETS)
                if bool(alive[i])
            ]
        return self._fleets_cache

    def _find_planet_idx(self, planet_id: int) -> Optional[int]:
        """Find array index of a planet by id. Returns None if unknown
        / dead."""
        s = self._state
        alive = np.asarray(s.planets_alive)
        ids = np.asarray(s.planets_id)
        for i in range(MAX_PLANETS):
            if alive[i] and int(ids[i]) == planet_id:
                return i
        return None

    def _find_fleet_idx(self, fleet_id: int) -> Optional[int]:
        """Find array index of a fleet by id. Returns None if unknown
        / dead."""
        s = self._state
        alive = np.asarray(s.fleets_alive)
        ids = np.asarray(s.fleets_id)
        for i in range(MAX_FLEETS):
            if alive[i] and int(ids[i]) == fleet_id:
                return i
        return None

    # -- O(1) position queries --------------------------------------------

    def planet_position(
        self,
        planet_id_or_idx: int,
        t: int,
        *,
        by_id: bool = True,
    ) -> Optional[tuple[float, float]]:
        """`(x, y)` of planet at relative turn `t`. O(1) closed-form.

        `t=0` returns the current position. Static planets (orbital
        ring outside `ROTATION_RADIUS_LIMIT`) return the same
        position for all `t`. Comets are handled separately — this
        method delegates to `comet_position` when `is_comet[idx]`.

        `by_id=True` (default) treats the first arg as a planet id;
        `by_id=False` treats it as a raw array index.

        Returns `None` if the planet doesn't exist or its slot isn't
        alive.
        """
        s = self._state
        if by_id:
            idx = self._find_planet_idx(planet_id_or_idx)
            if idx is None:
                return None
        else:
            idx = int(planet_id_or_idx)
            if idx < 0 or idx >= MAX_PLANETS or not bool(s.planets_alive[idx]):
                return None

        if bool(s.is_comet[idx]):
            return self._comet_position_by_idx(idx, t)

        cx = float(s.planets_x[idx])
        cy = float(s.planets_y[idx])
        radius = float(s.planets_radius[idx])

        if t == 0 or self._omega == 0.0:
            return (cx, cy)

        dx = cx - CENTER
        dy = cy - CENTER
        orb_r = math.hypot(dx, dy)
        if orb_r + radius >= ROTATION_RADIUS_LIMIT:
            return (cx, cy)

        # Off-by-one accounting: `jax_step.planet_path_compute` runs
        # BEFORE `step += 1`, so at `state.step = S` the planets_x
        # field reflects `angle(initial + omega * (S - 1))` for S >= 1,
        # and `initial` for S = 0 (no rotation yet applied). To predict
        # planets_x at `state.step = S + t`, the rotation delta is:
        #   S = 0, t = 0 → 0 (current = initial)
        #   S = 0, t >= 1 → omega * (t - 1)  (future planets_x at angle(omega*(t-1)))
        #   S >= 1 → omega * t  (future angle(omega*(S+t-1)) vs current angle(omega*(S-1)))
        s_step = int(self._state.step)
        if s_step == 0:
            effective_t = max(0, t - 1)
        else:
            effective_t = t

        cur_angle = math.atan2(dy, dx)
        new_angle = cur_angle + self._omega * effective_t
        return (
            CENTER + orb_r * math.cos(new_angle),
            CENTER + orb_r * math.sin(new_angle),
        )

    def comet_position(self, comet_id: int, t: int) -> Optional[tuple[float, float]]:
        """`(x, y)` of a comet at relative turn `t`. O(1) lookup into
        the pre-computed comet path array.

        Returns `None` if the planet id isn't a comet, or if the
        comet has expired by turn `t` (path index out of bounds).
        """
        idx = self._find_planet_idx(comet_id)
        if idx is None:
            return None
        return self._comet_position_by_idx(idx, t)

    def _comet_position_by_idx(self, idx: int, t: int) -> Optional[tuple[float, float]]:
        s = self._state
        if not bool(s.is_comet[idx]):
            return None
        spawn_k = int(s.planet_comet_spawn[idx])
        path_j = int(s.planet_comet_path[idx])
        if spawn_k < 0 or path_j < 0:
            return None
        current_idx = int(s.comet_path_index[spawn_k])
        path_len = int(s.comet_paths_len[spawn_k, path_j])
        future_idx = current_idx + t
        if future_idx < 0 or future_idx >= path_len:
            return None  # Expired or not-yet-active
        return (
            float(s.comet_paths_xy[spawn_k, path_j, future_idx, 0]),
            float(s.comet_paths_xy[spawn_k, path_j, future_idx, 1]),
        )

    def fleet_position(self, fleet_id: int, t: int) -> Optional[tuple[float, float]]:
        """`(x, y)` of a fleet at relative turn `t`. O(1) straight-line
        projection.

        **IGNORES COLLISIONS** — returns the would-be position if the
        fleet were to fly unobstructed. For collision-aware
        prediction (where did the fleet actually go?), use
        `lib.foundation.jax_engine.step` / `rollout_python` to
        forward-simulate and read the resulting state.

        Returns `None` if `fleet_id` isn't an alive fleet.
        """
        s = self._state
        idx = self._find_fleet_idx(fleet_id)
        if idx is None:
            return None
        cx = float(s.fleets_x[idx])
        cy = float(s.fleets_y[idx])
        angle = float(s.fleets_angle[idx])
        ships = int(s.fleets_ships[idx])
        spd = fleet_speed(ships)
        return (cx + math.cos(angle) * spd * t, cy + math.sin(angle) * spd * t)

    def positions_at(self, t: int) -> dict[str, np.ndarray]:
        """All entity positions at relative turn `t` as numpy arrays.

        Keys:
            "planets" — shape `(P, 2)` non-comet planet positions
            "comets"  — shape `(C, 2)` comet positions (active ones
                        only; expired comets are dropped)
            "fleets"  — shape `(F, 2)` fleet straight-line projections
                        (collisions ignored)

        Order within each group follows array order; pair with the
        corresponding `_planets_list()` / `_fleets_list()` if id
        attribution matters.
        """
        s = self._state
        planets_alive = np.asarray(s.planets_alive)
        fleets_alive = np.asarray(s.fleets_alive)
        is_comet = np.asarray(s.is_comet)

        planet_pos: list[tuple[float, float]] = []
        comet_pos: list[tuple[float, float]] = []
        fleet_pos: list[tuple[float, float]] = []

        for i in range(MAX_PLANETS):
            if not planets_alive[i]:
                continue
            pos = self.planet_position(i, t, by_id=False)
            if pos is None:
                continue
            if is_comet[i]:
                comet_pos.append(pos)
            else:
                planet_pos.append(pos)

        for i in range(MAX_FLEETS):
            if not fleets_alive[i]:
                continue
            cx = float(s.fleets_x[i])
            cy = float(s.fleets_y[i])
            angle = float(s.fleets_angle[i])
            ships = int(s.fleets_ships[i])
            spd = fleet_speed(ships)
            fleet_pos.append(
                (cx + math.cos(angle) * spd * t, cy + math.sin(angle) * spd * t)
            )

        def _arr(lst):
            return np.array(lst, dtype=np.float32).reshape(-1, 2)

        return {
            "planets": _arr(planet_pos),
            "comets": _arr(comet_pos),
            "fleets": _arr(fleet_pos),
        }

    # -- O(horizon) timeline queries --------------------------------------

    def arrival_ledger(
        self,
        horizon: int,
        hypothetical: Iterable[ActionSpec] = (),
    ) -> dict[int, list[tuple[int, int, int]]]:
        """Per-planet `[(eta, owner, ships), ...]` arrival ledger
        including hypothetical launches.

        `hypothetical` is a list of `ActionSpec`. Each spec with
        `launch_turn=0` is materialised as a virtual `Fleet` at the
        spawn position (source planet + radius offset along
        `dir_angle`) and added to the in-flight fleet set before
        `build_arrival_ledger` runs.

        `launch_turn > 0` raises `NotImplementedError` — for now,
        callers wanting "what if I launch on turn 3" should
        `jax_engine.step` forward to turn 3 first, then query.
        Step 9 extends this with a forward-sim helper.
        """
        planets = self._planets_list()
        fleets = list(self._fleets_list())

        for spec in hypothetical:
            if spec.launch_turn != 0:
                raise NotImplementedError(
                    f"Predictor.arrival_ledger: launch_turn>0 not yet "
                    f"supported (got spec={spec}). Use "
                    f"jax_engine.step to forward-simulate to the "
                    f"launch turn first, then build a new Predictor."
                )
            virtual = self._synthesise_fleet(spec, planets)
            if virtual is not None:
                fleets.append(virtual)

        return build_arrival_ledger(
            fleets, planets, horizon=horizon, omega=self._omega,
        )

    def _synthesise_fleet(
        self,
        spec: ActionSpec,
        planets: list[Planet],
    ) -> Optional[Fleet]:
        """Convert a launch `ActionSpec` to a virtual `Fleet` at the
        spawn position. Returns `None` if the source planet is
        unknown."""
        src = next((p for p in planets if p.id == spec.from_planet_id), None)
        if src is None:
            return None
        cos_a = math.cos(spec.dir_angle)
        sin_a = math.sin(spec.dir_angle)
        spawn_x = src.x + cos_a * (src.radius + 0.1)
        spawn_y = src.y + sin_a * (src.radius + 0.1)
        return Fleet(
            id=-1,  # sentinel for hypothetical
            owner=spec.agent_id,
            x=spawn_x,
            y=spawn_y,
            angle=spec.dir_angle,
            from_planet_id=spec.from_planet_id,
            ships=spec.ships,
        )

    def ships_at_planet(
        self,
        planet_id: int,
        t: int,
        hypothetical: Iterable[ActionSpec] = (),
    ) -> tuple[int, float]:
        """`(owner, ships)` at `planet_id` at relative turn `t`.

        Accounts for production AND the listed hypothetical
        launches. `owner = -1` for neutral. Returns `(-1, 0.0)` if
        the planet id isn't known.
        """
        if t < 0:
            raise ValueError(f"Predictor.ships_at_planet: t must be >= 0, got {t}")
        hypothetical_list = list(hypothetical)
        ledger = self.arrival_ledger(
            horizon=max(t + 1, 1), hypothetical=hypothetical_list,
        )
        planets = self._planets_list()
        planet = next((p for p in planets if p.id == planet_id), None)
        if planet is None:
            return (-1, 0.0)
        timeline = simulate_planet_timeline(
            planet, ledger.get(planet_id, []), horizon=t + 1,
        )
        owner, ships = state_at_timeline(timeline, t)
        return (int(owner), float(ships))

    # -- Introspection -----------------------------------------------------

    @property
    def state(self) -> GameState:
        """The input state. Read-only — Pytree immutable."""
        return self._state

    @property
    def memory(self) -> Memory:
        """The optional memory passed in at construction."""
        return self._memory

    @property
    def omega(self) -> float:
        """Game's angular velocity (rotation rate). 0.0 → no
        planets rotate."""
        return self._omega
