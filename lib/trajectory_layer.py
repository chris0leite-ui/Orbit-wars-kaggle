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
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

from lib.aim import swept_pair_hit
from lib.combat import resolve_arrivals

# ---------------------------------------------------------------------------
# Constants — must match `lib/game/interpreter.py` exactly
# ---------------------------------------------------------------------------

BOARD_SIZE: float = 100.0
CENTER: float = 50.0
SUN_RADIUS: float = 10.0
ROTATION_RADIUS_LIMIT: float = 50.0  # orbital_radius + planet_radius < this → rotating

# Sun safety cushion for ray-casting (matches `lib/trajectory.SUN_SAFETY`).
# A fleet's swept segment touching within (SUN_RADIUS + SUN_SAFETY) of
# CENTER is considered sun-killed. The 0.5-unit cushion absorbs float
# drift on tangent paths so the new ray-cast doesn't disagree with the
# env's collision verdict at the edge.
SUN_SAFETY: float = 0.5

# Default horizon for ledger / timeline queries. Matches
# `lib/world_model.DEFAULT_HORIZON`.
DEFAULT_LEDGER_HORIZON: int = 250

# Max step depth for per-fleet ray-cast. Matches
# `lib/trajectory.DEFAULT_MAX_STEPS`. 1-ship fleets at speed 1.0
# cross the 141-unit board diagonal in ~142 steps; 200 leaves margin.
DEFAULT_RAYCAST_STEPS: int = 200

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
    # Phase 2 caches (lazily populated by the ledger methods):
    # `_ledger_cache_built` is a list with one bool (mutable through
    # the frozen wrapper). Indexed by horizon to avoid re-building
    # for different horizons on the same World.
    _ledger_cache: dict[int, dict[int, tuple["Arrival", ...]]] = field(
        default_factory=dict, compare=False, repr=False,
    )
    _timeline_cache: dict[tuple[int, int], dict] = field(
        default_factory=dict, compare=False, repr=False,
    )

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
            _ledger_cache={},
            _timeline_cache={},
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

    # ---------------------------------------------------------------
    # Phase 2 — Arrival ledger + per-planet timelines
    # ---------------------------------------------------------------

    def _full_ledger(self,
                     horizon: int,
                     ) -> dict[int, tuple[Arrival, ...]]:
        """Internal: get the full ledger for a horizon, building once
        and caching per-horizon."""
        cached = self._ledger_cache.get(horizon)
        if cached is not None:
            return cached
        built = _build_full_ledger(self, horizon)
        self._ledger_cache[horizon] = built
        return built

    def ledger_for(self, planet_id: int,
                   horizon: int = DEFAULT_LEDGER_HORIZON,
                   ) -> tuple[Arrival, ...]:
        """Arrivals destined for `planet_id` within `horizon` turns.

        Returns `()` for unknown `planet_id` or no inbound fleets.
        Arrivals are sorted by `(eta, fleet_id)` for deterministic
        iteration. The first call for a horizon builds the full
        ledger (eager); subsequent calls hit the cache.
        """
        return self._full_ledger(horizon).get(planet_id, ())

    def ledger_all(self,
                   horizon: int = DEFAULT_LEDGER_HORIZON,
                   ) -> Mapping[int, tuple[Arrival, ...]]:
        """Full ledger view as a read-only mapping. Built once per
        horizon; subsequent calls hit the cache. Returned mapping is
        the cached dict; callers MUST NOT mutate."""
        return self._full_ledger(horizon)

    def _timeline_for(self, planet_id: int,
                      horizon: int,
                      ) -> Optional[dict]:
        """Internal: get the per-planet timeline for a horizon,
        building once and caching per (planet, horizon)."""
        key = (planet_id, horizon)
        cached = self._timeline_cache.get(key)
        if cached is not None:
            return cached
        planet = self._planet_by_id.get(planet_id)
        if planet is None:
            # No PlanetView (could be a comet — comets don't carry
            # garrison/owner in a meaningful sense post-arrival;
            # combat for comets resolves the same way, so we still
            # need a timeline if asked).
            comet = self._comet_by_planet_id.get(planet_id)
            if comet is None:
                return None
            # Synthesise a minimal PlanetView for the comet from the
            # `planets` tuple (every comet IS a planet in obs.planets;
            # find it). The conditional ensures we don't reach here
            # unless the comet is in the comet_paths but missing from
            # `planets` — practically impossible in real obs, but
            # defensive.
            for p in self.planets:
                if p.id == planet_id:
                    planet = p
                    break
            if planet is None:
                return None
        arrivals = self.ledger_for(planet_id, horizon)
        timeline = _simulate_planet_timeline(planet, arrivals, horizon)
        self._timeline_cache[key] = timeline
        return timeline

    def ownership_at(self, planet_id: int, t: int,
                     horizon: int = DEFAULT_LEDGER_HORIZON,
                     ) -> tuple[int, float]:
        """`(owner, ships)` at relative turn `t`. Returns the current
        snapshot for `t=0`; walks the planet's timeline for `t>0`.

        Returns `(-1, 0.0)` for unknown `planet_id`. Clamps `t` to
        `[0, horizon]` (queries past `horizon` return the horizon
        endpoint's state — analogous to
        `lib/world_model.state_at_timeline`).
        """
        if t < 0:
            t = 0
        timeline = self._timeline_for(planet_id, horizon)
        if timeline is None:
            return (-1, 0.0)
        h = timeline["horizon"]
        clamped = min(t, h)
        return (timeline["owner_at"][clamped],
                float(timeline["ships_at"][clamped]))

    def incoming_enemy_eta(self, planet_id: int, my_id: int,
                           horizon: int = DEFAULT_LEDGER_HORIZON,
                           ) -> Optional[int]:
        """Min ETA among arrivals to `planet_id` not owned by `my_id`.
        Returns `None` if no enemy fleet is inbound within `horizon`.

        Mirrors `WorldModel.incoming_enemy_eta` semantics (drops
        zero-ship arrivals; counts neutral attackers as enemies)."""
        arrivals = self.ledger_for(planet_id, horizon)
        if not arrivals:
            return None
        enemy_etas = [a.eta for a in arrivals
                      if a.owner != my_id and a.ships > 0]
        if not enemy_etas:
            return None
        return min(enemy_etas)


__all__ = [
    "BOARD_SIZE", "CENTER", "SUN_RADIUS", "ROTATION_RADIUS_LIMIT",
    "SUN_SAFETY", "DEFAULT_LEDGER_HORIZON", "DEFAULT_RAYCAST_STEPS",
    "GameConfig",
    "PlanetView", "FleetView", "CometPathView", "Arrival",
    "World",
]


# ---------------------------------------------------------------------------
# PHASE 2 — Arrival ledger (sparse + eager) and per-planet timelines
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Arrival:
    """One predicted fleet arrival at a planet.

    `eta` is in turns relative to the snapshot's `obs.step`; eta>=1
    (a fleet arriving on the same step it was launched has eta=1
    in the env, because movement happens BEFORE rotation+combat).
    """
    eta: int
    owner: int
    ships: int
    fleet_id: int

    # Compatibility with the legacy ledger's `(eta, owner, ships)` triples.
    def to_legacy_tuple(self) -> tuple[int, int, int]:
        return (self.eta, self.owner, self.ships)


# Module-level helpers (fresh-build implementations of the orchestration;
# the combat primitive `lib.combat.resolve_arrivals` and the geometry
# primitive `lib.aim.swept_pair_hit` are reused unchanged — they're
# env-faithful and bit-tested).


def _fleet_target_planet(
    world: "World",
    fleet: "FleetView",
    *,
    max_steps: int = DEFAULT_RAYCAST_STEPS,
    sun_safety: float = SUN_SAFETY,
) -> tuple[Optional[int], Optional[int]]:
    """Per-fleet ray-cast: returns `(planet_id, eta_steps)` of the first
    planet the fleet collides with, or `(None, None)` if the fleet dies
    in sun / OOB / times out without hitting any planet.

    Walks the fleet forward step-by-step. At each turn `t in [1,
    max_steps]`:
      - Fleet position at t-1 and t (straight-line, no collisions yet).
      - Sun: point-to-segment distance from the fleet's swept segment to
        CENTER must be >= SUN_RADIUS + sun_safety. Else: dies in sun.
      - OOB: position at t outside [0, BOARD_SIZE]² → dies OOB.
      - Planet collision: for every planet, `swept_pair_hit` between
        the fleet's (old, new) segment and the planet's (old, new)
        chord, with the planet's radius. First hit wins.

    Spawn-step source-planet skip is NOT applied here — this function
    operates on already-in-flight fleets; the fleet's `current_x/y` is
    already past its source. (For NEW launches in Phase 3, the overlay
    constructor handles the spawn-offset and skip explicitly.)
    """
    if fleet.speed <= 0:
        return None, None

    # Pre-compute the planets we'll check against. Comets count as
    # planets for collision purposes (the env's interpreter checks
    # against every entry in `obs.planets`, including comet pids).
    planet_ids: list[int] = [p.id for p in world.planets]

    for t in range(1, max_steps + 1):
        f_old = fleet.position_at(t - 1)
        f_new = fleet.position_at(t)

        # Sun check (matches lib/trajectory.predict_fleet_fate).
        if _segment_to_point_distance(f_old, f_new, (CENTER, CENTER)) \
                < SUN_RADIUS + sun_safety:
            return None, None

        # OOB check.
        if (f_new[0] < 0.0 or f_new[0] > BOARD_SIZE
                or f_new[1] < 0.0 or f_new[1] > BOARD_SIZE):
            return None, None

        # Planet collision.
        for pid in planet_ids:
            p_old = world.planet_position(pid, t - 1)
            p_new = world.planet_position(pid, t)
            if p_old is None or p_new is None:
                # Comet that's expired or planet that doesn't exist; skip.
                continue
            p = world.planet_by_id(pid)
            if p is None:
                # Could be a comet (no PlanetView).
                comet = world.comet_by_planet_id(pid)
                if comet is None:
                    continue
                # Comets have a fixed radius of 1.0 per the env.
                prad = 1.0
            else:
                prad = p.radius
            if swept_pair_hit(f_old, f_new, p_old, p_new, prad):
                return pid, t

    return None, None  # timeout


def _segment_to_point_distance(
    a: tuple[float, float],
    b: tuple[float, float],
    p: tuple[float, float],
) -> float:
    """Shortest distance from segment a->b to point p. Mirrors
    `lib/trajectory._segment_to_point_distance` for parity."""
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    seg_len2 = dx * dx + dy * dy
    if seg_len2 == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / seg_len2
    t = max(0.0, min(1.0, t))
    cx = ax + t * dx
    cy = ay + t * dy
    return math.hypot(px - cx, py - cy)


def _build_full_ledger(
    world: "World",
    horizon: int,
) -> dict[int, tuple[Arrival, ...]]:
    """Build the per-planet ledger by ray-casting every in-flight fleet.

    Returns `{planet_id: (Arrival, ...), ...}` keyed by every planet
    that has at least one arrival; planets with no inbound fleet are
    NOT in the dict (callers query with `.get(pid, ())` for graceful
    empties — done by `World.ledger_for`).

    Fleets that don't hit any planet within `horizon` (sun / OOB /
    timeout) are dropped — they exit the simulation without affecting
    any planet's timeline.
    """
    buckets: dict[int, list[Arrival]] = defaultdict(list)
    max_steps = min(int(horizon), DEFAULT_RAYCAST_STEPS)
    for fleet in world.fleets:
        if fleet.ships <= 0:
            continue
        target_id, eta = _fleet_target_planet(
            world, fleet, max_steps=max_steps,
        )
        if target_id is None or eta is None:
            continue
        if eta > horizon:
            continue
        buckets[target_id].append(Arrival(
            eta=int(eta),
            owner=int(fleet.owner),
            ships=int(fleet.ships),
            fleet_id=int(fleet.id),
        ))
    # Sort arrivals per planet by eta for deterministic iteration.
    return {
        pid: tuple(sorted(arrs, key=lambda a: (a.eta, a.fleet_id)))
        for pid, arrs in buckets.items()
    }


def _simulate_planet_timeline(
    planet: "PlanetView",
    arrivals: Iterable[Arrival],
    horizon: int,
) -> dict:
    """Per-planet ownership/garrison timeline under integer-tick semantics.

    Mirrors `lib/world_model.simulate_planet_timeline` exactly:
      1. At t=0, record current (owner, ships).
      2. For t in [1, horizon]:
         a. If currently owned (owner != -1), garrison += production.
         b. Resolve same-step arrivals via `lib.combat.resolve_arrivals`.
         c. Record (owner_at[t], ships_at[t]).

    Returns `{owner_at: dict[int, int], ships_at: dict[int, float],
              horizon: int}`. The legacy format kept for parity.
    """
    h = max(0, int(math.ceil(horizon)))
    by_turn: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for arr in arrivals:
        if arr.ships <= 0:
            continue
        bucket = max(1, int(math.ceil(arr.eta)))
        if bucket > h:
            continue
        by_turn[bucket].append((arr.owner, int(arr.ships)))

    owner = planet.owner
    garrison = float(planet.ships)
    owner_at = {0: owner}
    ships_at = {0: max(0.0, garrison)}

    for t in range(1, h + 1):
        if owner != -1:
            garrison += planet.production
        group = by_turn.get(t, [])
        if group:
            owner, garrison = resolve_arrivals(owner, garrison, group)
        owner_at[t] = owner
        ships_at[t] = max(0.0, garrison)

    return {"owner_at": owner_at, "ships_at": ships_at, "horizon": h}
