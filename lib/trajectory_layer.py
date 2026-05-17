"""lib/trajectory_layer.py — sparse, 100% accurate trajectory layer.

The canonical "where will every entity be at relative turn t" oracle.
Closed-form positions in O(1) for planets / comets / fleets. Built
once per turn at the top of `agent()`; queried wherever the chooser /
value head / mission scorer needs future state. No simulation, no
combat resolution at this phase — Phase 1 covers positions only.

Subsequent phases add: arrival ledger (Phase 2), hypothetical-launch
overlay (Phase 3), SunFilter (Phase 4), lazy combat (Phase 5),
differential parity harness (Phase 6).

Parity invariant: every position returned MUST match the env's
`lib/game/interpreter.py` exactly. The env rotates planets BEFORE
incrementing `obs.step`, so:

  obs.step = 0  → planets at INITIAL angle (atan2 of initial coords)
  obs.step = S ≥ 1 → planets at INITIAL + omega·(S - 1)

A query for relative turn `t` from current step `S` asks for the
position at `obs.step = S + t`, i.e.:

  S + t = 0       → init
  S + t ≥ 1       → init + omega·(S + t - 1)

Equivalently, working from the CURRENT (already-rotated) position:

  S = 0, t = 0   → current (= init)
  S = 0, t ≥ 1   → current + omega·(t - 1)
  S ≥ 1, any t   → current + omega·t

The `_effective_t` helper below encodes exactly this. Pinned by
`tests/test_trajectory_layer_positions.py::test_planet_position_off_by_one_step_zero`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

# ---------------------------------------------------------------------------
# Constants — must match `lib/game/interpreter.py` exactly
# ---------------------------------------------------------------------------

BOARD_SIZE: float = 100.0
CENTER: float = 50.0
SUN_RADIUS: float = 10.0
ROTATION_RADIUS_LIMIT: float = 50.0  # orbital_radius + planet_radius < this → rotating

# Default game configuration (mirrors `lib/fast_sim.DEFAULT_CONFIG`).
_DEFAULT_CFG = {
    "episodeSteps": 500,
    "shipSpeed": 6.0,
    "sunRadius": 10.0,
    "boardSize": 100.0,
    "cometSpeed": 4.0,
    "actTimeout": 1.0,
}


@dataclass(frozen=True)
class GameConfig:
    """Frozen view of the env's runtime configuration."""
    episode_steps: int = 500
    ship_speed: float = 6.0
    sun_radius: float = 10.0
    board_size: float = 100.0
    comet_speed: float = 4.0
    act_timeout: float = 1.0

    @classmethod
    def from_configuration(cls, configuration: Any) -> "GameConfig":
        """Accept dict / Struct / SimpleNamespace; coerce to frozen view."""
        get = (lambda k, default:
               configuration.get(k, default) if isinstance(configuration, dict)
               else getattr(configuration, k, default))
        if configuration is None:
            return cls()
        return cls(
            episode_steps=int(get("episodeSteps", _DEFAULT_CFG["episodeSteps"])),
            ship_speed=float(get("shipSpeed", _DEFAULT_CFG["shipSpeed"])),
            sun_radius=float(get("sunRadius", _DEFAULT_CFG["sunRadius"])),
            board_size=float(get("boardSize", _DEFAULT_CFG["boardSize"])),
            comet_speed=float(get("cometSpeed", _DEFAULT_CFG["cometSpeed"])),
            act_timeout=float(get("actTimeout", _DEFAULT_CFG["actTimeout"])),
        )


# ---------------------------------------------------------------------------
# Per-entity views — projection-ready, immutable
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanetView:
    """Per-planet projection-ready view.

    `current_x/y` is the position at `obs.step` (after the interpreter
    rotated planets for that step). `init_x/y` is the position the env
    stored in `obs.initial_planets` at game-start.

    For non-rotating planets, `current == init` for all turns.
    For rotating planets, the orbital math reconstructs the closed-form
    position at any relative turn.
    """
    id: int
    owner: int
    current_x: float
    current_y: float
    init_x: float
    init_y: float
    radius: float
    ships: float
    production: float
    is_comet: bool

    # Pre-computed orbital invariants (computed once in from_obs):
    orbital_radius: float  # distance from CENTER (at init)
    init_angle: float      # atan2 of (init - CENTER)
    is_rotating: bool      # orbital_radius + radius < ROTATION_RADIUS_LIMIT


def _effective_t_for_orbital(current_step: int, t: int) -> int:
    """Translate a relative-turn-t into the equivalent "rotation delta
    relative to the current angle", accounting for the env's step-0
    quirk where rotation hasn't been applied yet.

    See module docstring for the derivation.
    """
    if current_step == 0:
        return max(0, t - 1)
    return t


def _planet_position_orbital(planet: PlanetView, omega: float,
                              current_step: int, t: int,
                              ) -> tuple[float, float]:
    """Closed-form orbital position at relative turn `t`.
    Caller is responsible for handling `is_rotating == False`
    (just return current_x/y) and t==0 (also return current).
    """
    eff = _effective_t_for_orbital(current_step, t)
    # The current angle is atan2 of CURRENT coords minus CENTER (NOT
    # init coords) — this respects the env's off-by-one rotation
    # accounting documented in `lib/foundation/predictor.py:208-220`.
    cur_angle = math.atan2(planet.current_y - CENTER, planet.current_x - CENTER)
    new_angle = cur_angle + omega * eff
    r = planet.orbital_radius
    return (CENTER + r * math.cos(new_angle),
            CENTER + r * math.sin(new_angle))


@dataclass(frozen=True)
class FleetView:
    """Per-fleet projection-ready view. Straight-line motion at
    `speed(ships)`. Collisions are NOT modelled at this phase — Phase 2
    adds the arrival ledger; Phase 4 adds the SunFilter.
    """
    id: int
    owner: int
    current_x: float
    current_y: float
    angle: float
    ships: int
    from_planet_id: int
    speed: float

    def position_at(self, t: int) -> tuple[float, float]:
        return (self.current_x + math.cos(self.angle) * self.speed * t,
                self.current_y + math.sin(self.angle) * self.speed * t)


def _fleet_speed(ships: int, ship_speed: float = 6.0) -> float:
    """Mirrors `lib.fleet.speed` and the env's per-step formula
    (`lib/game/interpreter.py:746`):

        speed = 1 + (max_speed - 1) * (log(ships) / log(1000)) ** 1.5

    Clipped to max_speed.
    """
    s = max(1, int(ships))
    raw = 1.0 + (ship_speed - 1.0) * (math.log(s) / math.log(1000.0)) ** 1.5
    return min(raw, ship_speed)


@dataclass(frozen=True)
class CometPathView:
    """Per-comet pre-computed XY path. `path[absolute_step]` is the
    position at that step; `path_index` is the entry corresponding to
    `obs.step` (the comet's current position).

    Position at relative turn `t` is `path[path_index + t]` (or `None`
    if past the path end — comet has expired).
    """
    planet_id: int
    path: tuple[tuple[float, float], ...]
    path_index: int

    def position_at(self, t: int) -> Optional[tuple[float, float]]:
        idx = self.path_index + t
        if idx < 0 or idx >= len(self.path):
            return None
        return self.path[idx]


# ---------------------------------------------------------------------------
# World — the top-level snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class World:
    """Per-turn immutable trajectory snapshot. Build once at the top of
    `agent()`; never mutate; share across all queries inside the turn.

    Hypothetical launches (Phase 3) return a NEW World; the receiver
    is unchanged. Caches (Phase 2+) are added as private fields holding
    `MappingProxyType` views over computed-once dicts.
    """
    step: int
    my_id: int
    omega: float
    episode_seed: Optional[int]
    cfg: GameConfig
    planets: tuple[PlanetView, ...]
    fleets: tuple[FleetView, ...]
    comet_paths: tuple[CometPathView, ...]
    # Indices (built in `from_obs`). Plain dicts; the immutability
    # invariant is that we never mutate them after construction.
    _planet_by_id: dict[int, PlanetView] = field(default_factory=dict, compare=False, repr=False)
    _fleet_by_id: dict[int, FleetView] = field(default_factory=dict, compare=False, repr=False)
    _comet_by_planet_id: dict[int, CometPathView] = field(default_factory=dict, compare=False, repr=False)

    # ---------------------------------------------------------------
    # Construction
    # ---------------------------------------------------------------

    @classmethod
    def from_obs(cls, obs: Any, configuration: Any = None,
                 *, episode_seed: Optional[int] = None,
                 ) -> "World":
        """Build a World from a Kaggle obs (dict or Struct).

        `episode_seed=None` is allowed; downstream queries past the
        next comet-spawn boundary will be marked `UNCERTAIN` (Phase 4).
        Today's Phase 1 ignores it — positions are deterministic from
        what's in the obs.
        """
        # Helper for dict/attr dual access.
        def _g(o: Any, k: str, default: Any = None) -> Any:
            if isinstance(o, dict):
                return o.get(k, default)
            return getattr(o, k, default)

        cfg = GameConfig.from_configuration(configuration)
        step = int(_g(obs, "step", 0) or 0)
        my_id = int(_g(obs, "player", 0) or 0)
        omega = float(_g(obs, "angular_velocity", 0.0) or 0.0)

        comet_pid_list = _g(obs, "comet_planet_ids", []) or []
        comet_pid_set: set[int] = {int(c) for c in comet_pid_list}

        raw_initial = _g(obs, "initial_planets", []) or []
        initial_by_id: dict[int, tuple] = {int(p[0]): p for p in raw_initial}

        raw_planets = _g(obs, "planets", []) or []
        planets: list[PlanetView] = []
        for p in raw_planets:
            pid = int(p[0])
            owner = int(p[1])
            cur_x = float(p[2])
            cur_y = float(p[3])
            radius = float(p[4])
            ships = float(p[5])
            production = float(p[6])
            is_comet = pid in comet_pid_set
            init = initial_by_id.get(pid)
            if init is not None:
                init_x = float(init[2])
                init_y = float(init[3])
            else:
                # Comets and any planet missing from initial_planets
                # fall back to current position. is_rotating will be
                # computed from current → orbital_radius is consistent
                # with env semantics (comets don't orbit, they path).
                init_x = cur_x
                init_y = cur_y
            dx = init_x - CENTER
            dy = init_y - CENTER
            orbital_radius = math.hypot(dx, dy)
            init_angle = math.atan2(dy, dx)
            # Comets never rotate; they're path-driven.
            is_rotating = (not is_comet
                           and (orbital_radius + radius) < ROTATION_RADIUS_LIMIT)
            planets.append(PlanetView(
                id=pid, owner=owner,
                current_x=cur_x, current_y=cur_y,
                init_x=init_x, init_y=init_y,
                radius=radius, ships=ships, production=production,
                is_comet=is_comet,
                orbital_radius=orbital_radius,
                init_angle=init_angle,
                is_rotating=is_rotating,
            ))

        raw_fleets = _g(obs, "fleets", []) or []
        fleets: list[FleetView] = []
        for f in raw_fleets:
            ships_i = int(f[6])
            fleets.append(FleetView(
                id=int(f[0]),
                owner=int(f[1]),
                current_x=float(f[2]),
                current_y=float(f[3]),
                angle=float(f[4]),
                ships=ships_i,
                from_planet_id=int(f[5]),
                speed=_fleet_speed(ships_i, cfg.ship_speed),
            ))

        raw_comets = _g(obs, "comets", []) or []
        comet_paths: list[CometPathView] = []
        for group in raw_comets:
            # group has 'planet_ids', 'paths', 'path_index'
            pids = (group.get("planet_ids", [])
                    if isinstance(group, dict)
                    else getattr(group, "planet_ids", []))
            paths = (group.get("paths", [])
                     if isinstance(group, dict)
                     else getattr(group, "paths", []))
            path_index = int(group.get("path_index", 0)
                             if isinstance(group, dict)
                             else getattr(group, "path_index", 0))
            for pid, path in zip(pids, paths):
                # path is a list of (x, y) pairs; freeze.
                frozen_path = tuple((float(pt[0]), float(pt[1])) for pt in path)
                comet_paths.append(CometPathView(
                    planet_id=int(pid),
                    path=frozen_path,
                    path_index=path_index,
                ))

        # Build the lookup indices.
        planet_by_id = {p.id: p for p in planets}
        fleet_by_id = {f.id: f for f in fleets}
        comet_by_pid = {c.planet_id: c for c in comet_paths}

        return cls(
            step=step,
            my_id=my_id,
            omega=omega,
            episode_seed=episode_seed,
            cfg=cfg,
            planets=tuple(planets),
            fleets=tuple(fleets),
            comet_paths=tuple(comet_paths),
            _planet_by_id=planet_by_id,
            _fleet_by_id=fleet_by_id,
            _comet_by_planet_id=comet_by_pid,
        )

    # ---------------------------------------------------------------
    # Position queries (closed-form, O(1))
    # ---------------------------------------------------------------

    def planet_position(self, planet_id: int, t: int,
                        ) -> Optional[tuple[float, float]]:
        """Position of planet at relative turn `t`. Returns the
        current (already-rotated) coords for static planets and t=0
        queries; the closed-form orbital projection otherwise.

        For comets, delegates to `comet_position`. Returns `None` if
        the planet id isn't present, or if it's an expired comet.
        """
        comet = self._comet_by_planet_id.get(planet_id)
        if comet is not None:
            return comet.position_at(t)
        planet = self._planet_by_id.get(planet_id)
        if planet is None:
            return None
        if t == 0:
            return (planet.current_x, planet.current_y)
        if not planet.is_rotating or self.omega == 0.0:
            return (planet.current_x, planet.current_y)
        return _planet_position_orbital(planet, self.omega, self.step, t)

    def fleet_position(self, fleet_id: int, t: int,
                       ) -> Optional[tuple[float, float]]:
        """Straight-line projection of a fleet at relative turn `t`.

        IGNORES COLLISIONS — returns the would-be position if the
        fleet flew unobstructed. Collision-aware queries come in
        Phase 2 (`ledger_for`) and Phase 4 (`SunFilter`).
        """
        f = self._fleet_by_id.get(fleet_id)
        if f is None:
            return None
        return f.position_at(t)

    def comet_position(self, planet_id: int, t: int,
                       ) -> Optional[tuple[float, float]]:
        """Comet position at relative turn `t` via path-array lookup.
        Returns `None` past the path end (comet expired).
        """
        c = self._comet_by_planet_id.get(planet_id)
        if c is None:
            return None
        return c.position_at(t)

    # ---------------------------------------------------------------
    # Convenience accessors
    # ---------------------------------------------------------------

    def planet_by_id(self, planet_id: int) -> Optional[PlanetView]:
        return self._planet_by_id.get(planet_id)

    def fleet_by_id(self, fleet_id: int) -> Optional[FleetView]:
        return self._fleet_by_id.get(fleet_id)

    def comet_by_planet_id(self, planet_id: int) -> Optional[CometPathView]:
        return self._comet_by_planet_id.get(planet_id)

    def is_comet(self, planet_id: int) -> bool:
        return planet_id in self._comet_by_planet_id


__all__ = [
    "BOARD_SIZE", "CENTER", "SUN_RADIUS", "ROTATION_RADIUS_LIMIT",
    "GameConfig",
    "PlanetView", "FleetView", "CometPathView",
    "World",
]
