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

import enum
import math
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Optional

from lib.aim import swept_pair_hit
from lib.combat import resolve_arrivals
from lib.opp_model import lite_greedy_policy

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

    `spawn_turn` is the relative turn (from snapshot's obs.step) at which
    the fleet exists. For in-flight fleets in the obs, `spawn_turn=0`
    (they exist now). For future-launch overlays (Phase 7), `spawn_turn`
    is the bundle's `launch_turn` — the fleet doesn't exist before that
    turn (`position_at(t<spawn_turn)` returns None).

    `current_x`/`current_y` is the position at `spawn_turn` — i.e. the
    spawn position for future launches, or the live position for
    in-flight fleets at `spawn_turn=0`.
    """
    id: int
    owner: int
    current_x: float
    current_y: float
    angle: float
    ships: int
    from_planet_id: int
    speed: float
    spawn_turn: int = 0

    def position_at(self, t: int) -> Optional[tuple[float, float]]:
        if t < self.spawn_turn:
            return None
        dt = t - self.spawn_turn
        return (self.current_x + math.cos(self.angle) * self.speed * dt,
                self.current_y + math.sin(self.angle) * self.speed * dt)


def _fleet_speed(ships: int, ship_speed: float = 6.0) -> float:
    """Mirrors `lib.fleet.speed` and the env's per-step formula
    (`lib/game/interpreter.py:746`):

        speed = 1 + (max_speed - 1) * (log(ships) / log(1000)) ** 1.5

    Clipped to max_speed.
    """
    s = max(1, int(ships))
    raw = 1.0 + (ship_speed - 1.0) * (math.log(s) / math.log(1000.0)) ** 1.5
    return min(raw, ship_speed)


def _raycast_first_planet_hit(
    src: "PlanetView", aim_angle: float, ships: int,
    planets: list, omega: float, rot_offset: int,
    max_steps: int = 60,
) -> tuple[Optional[int], Optional[int]]:
    """Phase E Phase 1: lightweight raycast for joint-detection.

    Walks the fleet forward from `src` along `aim_angle` and returns
    `(hit_planet_id, arrival_step)` for the first planet whose circle
    the fleet enters, or `(None, None)` if no hit within `max_steps`.

    For non-rotating planets, uses `current_x/current_y` as the fixed
    position. For rotating planets (omega > 0), advances each planet
    via `_planet_position_orbital` per step. Mirrors predict_fleet_fate
    geometry but on PlanetView (trajectory_layer's data shape), not on
    world_model.PlanetState.

    Skips the source planet itself. Does NOT check sun / OOB / comet —
    callers do their own SunFilter check before emitting.

    O(max_steps * planets) per call; ~1-2ms typical. Joint detection
    in `BundleEvaluator.score` runs this once per launch in the bundle.
    """
    if not planets:
        return None, None
    cos_a = math.cos(aim_angle)
    sin_a = math.sin(aim_angle)
    spawn_x = src.current_x + cos_a * (src.radius + 0.1)
    spawn_y = src.current_y + sin_a * (src.radius + 0.1)
    speed = _fleet_speed(ships)
    if speed <= 0:
        return None, None
    src_id = int(src.id)

    fx, fy = spawn_x, spawn_y
    for step in range(1, max_steps + 1):
        fx += cos_a * speed
        fy += sin_a * speed
        for p in planets:
            if int(p.id) == src_id:
                continue
            if p.is_rotating and omega != 0.0:
                t_eff = _effective_t_for_orbital(rot_offset, step)
                px, py = _planet_position_orbital(p, omega, rot_offset, t_eff)
            else:
                px, py = p.current_x, p.current_y
            dx = px - fx
            dy = py - fy
            if dx * dx + dy * dy <= (p.radius + 0.5) * (p.radius + 0.5):
                return int(p.id), step
    return None, None


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
    # Outgoing future launches scheduled via `with_candidate(launch_turn>0)`.
    # Each entry is `(src_id, launch_turn, ships)`. Phase 7 bundles cause
    # these to accumulate across multiple `with_candidate` calls. At
    # timeline-simulation time, each entry deducts `ships` from the source
    # planet's garrison at `launch_turn` BEFORE production accrues — same
    # ordering as `lib/game/interpreter.py`'s `process_moves` → production
    # phase sequence.
    _outgoing_launches: tuple[tuple[int, int, int], ...] = ()
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
    _combat_log_cache: dict[tuple[int, int], dict] = field(
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
            _combat_log_cache={},
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
        outgoing = self._outgoing_for(planet_id)
        timeline = _simulate_planet_timeline(planet, arrivals, horizon,
                                              outgoing=outgoing)
        self._timeline_cache[key] = timeline
        return timeline

    def _outgoing_for(self, planet_id: int,
                      ) -> tuple[tuple[int, int], ...]:
        """All outgoing future launches scheduled from `planet_id`.
        Returns `((turn, ships), ...)`. Empty if no future launches.
        Phase 1-6 worlds (no overlays / launch_turn=0 only) return ()."""
        return tuple((t, s) for src, t, s in self._outgoing_launches
                     if src == planet_id)

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

    # ---------------------------------------------------------------
    # Phase 5 — Per-planet combat outcomes (lazy)
    # ---------------------------------------------------------------

    def _combat_log_for(self, planet_id: int, horizon: int,
                        ) -> dict[int, "CombatOutcome"]:
        """Internal: lazy per-(planet, horizon) combat log build.

        Reuses the same arrival list as the timeline path; the combat
        log is a side-channel emitted by
        `_simulate_timeline_with_combat_log`. The plain
        `_timeline_cache` and the verbose `_combat_log_cache` are
        populated together on first call.
        """
        key = (planet_id, horizon)
        cached = self._combat_log_cache.get(key)
        if cached is not None:
            return cached
        planet = self._planet_by_id.get(planet_id)
        if planet is None:
            # Comet fallthrough (mirrors _timeline_for).
            comet = self._comet_by_planet_id.get(planet_id)
            if comet is None:
                return {}
            for p in self.planets:
                if p.id == planet_id:
                    planet = p
                    break
            if planet is None:
                return {}
        arrivals = self.ledger_for(planet_id, horizon)
        outgoing = self._outgoing_for(planet_id)
        timeline, log = _simulate_timeline_with_combat_log(
            planet, arrivals, horizon, outgoing=outgoing,
        )
        # Also pre-populate the plain timeline cache so a subsequent
        # `ownership_at` call doesn't redo the work.
        self._timeline_cache[key] = timeline
        self._combat_log_cache[key] = log
        return log

    def combat_at(self, planet_id: int, t: int,
                  horizon: int = DEFAULT_LEDGER_HORIZON,
                  ) -> Optional["CombatOutcome"]:
        """`CombatOutcome` at relative turn `t` on `planet_id`, or
        `None` if no arrivals at that turn (no combat). Bit-exact with
        `lib.combat.resolve_arrivals` on the same arrival group.

        Clamps `t` to `[1, horizon]`. `t <= 0` returns None (combat
        events are indexed from t=1 onward — t=0 is the snapshot).
        """
        if t <= 0:
            return None
        log = self._combat_log_for(planet_id, horizon)
        return log.get(int(t))

    # ---------------------------------------------------------------
    # Phase 3 — Hypothetical launch overlay
    # ---------------------------------------------------------------

    def with_candidate(self, spec: "LaunchSpec") -> "World":
        """Return a NEW World that includes a hypothetical launch.

        The receiver is unchanged. The new World has:
        - The source planet's ships decremented by `spec.ships`.
        - A synthetic FleetView added at the env-faithful spawn
          position (`source.center + (radius + 0.1) * direction`).
        - Fresh caches — the fleet set has changed, so the ledger
          and timelines must be recomputed on first query.

        Semantics: the overlay represents the state immediately
        AFTER the env's launch phase (`process_moves`) but BEFORE
        production / movement / rotation / combat. Querying
        `overlay.ledger_for(pid)` predicts where the synthetic
        fleet will arrive, with eta=1 corresponding to the fleet
        having traversed `speed` units from spawn.

        Compared with COMMITTING the launch (running fs_step with
        the action), the overlay is at step S; the committed state
        is at step S+1. For every arrival, `overlay.eta` ==
        `committed.eta + 1`.

        `spec.launch_turn > 0` represents a FUTURE-TURN commitment:
        the source's owner/ships are validated against the parent
        World's timeline at `launch_turn` (so chained overlays
        correctly account for prior launches in the same bundle), the
        spawn position is `planet_position(src, launch_turn) +
        (radius + 0.1) * direction`, and the synthetic FleetView
        carries `spawn_turn = launch_turn` so its ray-cast starts at
        the right time. The source planet's t=0 state is unchanged;
        the ship deduction is applied IN THE TIMELINE at the launch
        turn (via `_outgoing_launches`).

        Raises:
        - `ValueError` if `spec.src_id` isn't a known planet
        - `ValueError` if at `launch_turn` the source isn't owned by
          `spec.owner` or doesn't have enough ships
        - `ValueError` if `spec.ships <= 0` or `spec.launch_turn < 0`
        """
        if spec.ships <= 0:
            raise ValueError(f"spec.ships must be > 0 (got {spec.ships})")
        if spec.launch_turn < 0:
            raise ValueError(
                f"spec.launch_turn must be >= 0 (got {spec.launch_turn})"
            )
        src = self._planet_by_id.get(spec.src_id)
        if src is None:
            raise ValueError(f"unknown src_id: {spec.src_id}")

        # Validate ownership + ships AT launch_turn against the
        # current World's timeline. For launch_turn=0 this is the
        # current src state (owner / ships fields directly). For
        # future launches, query `ownership_at` so chained overlays
        # account for prior commitments.
        if spec.launch_turn == 0:
            src_owner_at_launch = src.owner
            src_ships_at_launch = float(src.ships)
        else:
            src_owner_at_launch, src_ships_at_launch = self.ownership_at(
                spec.src_id, spec.launch_turn,
            )
        if src_owner_at_launch != spec.owner:
            raise ValueError(
                f"source {spec.src_id} owned by {src_owner_at_launch} at "
                f"launch_turn={spec.launch_turn}, not {spec.owner}"
            )
        if spec.ships > src_ships_at_launch:
            raise ValueError(
                f"source {spec.src_id} has only {src_ships_at_launch} ships "
                f"at launch_turn={spec.launch_turn}, "
                f"cannot launch {spec.ships}"
            )

        # Pick a unique synthetic fleet id (negative, never collides
        # with real fleet ids which are non-negative).
        min_existing = min((f.id for f in self.fleets), default=0)
        virtual_id = min(min_existing, 0) - 1

        # Spawn position at launch_turn: planet_position(src, launch_turn)
        # + (radius + 0.1) * direction. For launch_turn=0 this is just
        # the src's current position (Phase 1's planet_position).
        cos_a = math.cos(spec.aim_angle)
        sin_a = math.sin(spec.aim_angle)
        src_pos = self.planet_position(spec.src_id, spec.launch_turn)
        if src_pos is None:
            # Source despawned (comet); shouldn't happen because the
            # ownership check above would have failed. Defensive only.
            raise ValueError(
                f"source {spec.src_id} has no position at "
                f"launch_turn={spec.launch_turn}"
            )
        spawn_x = src_pos[0] + cos_a * (src.radius + 0.1)
        spawn_y = src_pos[1] + sin_a * (src.radius + 0.1)

        new_fleet = FleetView(
            id=virtual_id,
            owner=spec.owner,
            current_x=spawn_x,
            current_y=spawn_y,
            angle=spec.aim_angle,
            ships=int(spec.ships),
            from_planet_id=spec.src_id,
            speed=_fleet_speed(spec.ships, self.cfg.ship_speed),
            spawn_turn=int(spec.launch_turn),
        )

        # Two cases for ship accounting:
        # - launch_turn == 0: deduct ships from src's PlanetView NOW
        #   (the post-process_moves snapshot semantics). DO NOT add
        #   to _outgoing_launches — the deduction is already baked
        #   into the t=0 PlanetView.
        # - launch_turn > 0: leave src.ships unchanged; record in
        #   _outgoing_launches so the timeline deducts at launch_turn
        #   before production.
        new_planets = self.planets
        new_planet_by_id = self._planet_by_id
        new_outgoing = self._outgoing_launches
        if spec.launch_turn == 0:
            new_src = replace(src, ships=src.ships - spec.ships)
            new_planets = tuple(
                new_src if p.id == spec.src_id else p
                for p in self.planets
            )
            new_planet_by_id = dict(self._planet_by_id)
            new_planet_by_id[spec.src_id] = new_src
        else:
            new_outgoing = self._outgoing_launches + (
                (int(spec.src_id), int(spec.launch_turn), int(spec.ships)),
            )

        # Append synthetic fleet.
        new_fleets = self.fleets + (new_fleet,)
        new_fleet_by_id = dict(self._fleet_by_id)
        new_fleet_by_id[virtual_id] = new_fleet

        # PERF (Phase 8): inherit parent's caches + incrementally
        # extend with the synthetic fleet's contribution. Without
        # this, every `bundle.apply(world)` paid O(N_fleets ×
        # N_planets × horizon) per child World — the dominant
        # BundleEvaluator cost (44 in-flight fleets × 24 planets ×
        # horizon turns = ~30 ms / score on mid-game states).
        #
        # The synthetic fleet is the ONLY fleet whose contribution
        # to the ledger is new; existing fleets' arrivals are
        # identical between parent and child because planet
        # positions, fleet positions, and sun geometry are
        # invariants of the (orbit_omega, step) snapshot. Affected
        # timelines: the source planet (ships decremented at t=0
        # for launch_turn=0, or via _outgoing_launches for
        # launch_turn>0) and the synthetic fleet's target planet
        # (new arrival). All other planet timelines are unchanged.
        affected: set[int] = {int(spec.src_id)}
        inherited_ledger: dict[int, dict[int, tuple[Arrival, ...]]] = {}
        # Compute the synthetic fleet's target once at the max
        # cached horizon, reuse across smaller horizons (target +
        # eta are physics-deterministic; smaller horizons just
        # drop arrivals whose eta is past the horizon).
        synth_target_id: Optional[int] = None
        synth_eta: Optional[int] = None
        if self._ledger_cache:
            max_h = max(self._ledger_cache.keys())
            max_steps = min(int(max_h), DEFAULT_RAYCAST_STEPS)
            synth_target_id, synth_eta = _fleet_target_planet(
                self, new_fleet, max_steps=max_steps,
            )
            if synth_target_id is not None:
                affected.add(int(synth_target_id))
            for h, parent_ledger in self._ledger_cache.items():
                new_ledger = dict(parent_ledger)
                if (synth_target_id is not None
                        and synth_eta is not None
                        and synth_eta <= h):
                    arrival = Arrival(
                        eta=int(synth_eta),
                        owner=int(new_fleet.owner),
                        ships=int(new_fleet.ships),
                        fleet_id=int(new_fleet.id),
                    )
                    existing = new_ledger.get(synth_target_id, ())
                    new_ledger[synth_target_id] = tuple(sorted(
                        existing + (arrival,),
                        key=lambda a: (a.eta, a.fleet_id),
                    ))
                inherited_ledger[h] = new_ledger
        inherited_timeline = {
            k: v for k, v in self._timeline_cache.items()
            if k[0] not in affected
        }
        inherited_combat = {
            k: v for k, v in self._combat_log_cache.items()
            if k[0] not in affected
        }

        return World(
            step=self.step,
            my_id=self.my_id,
            omega=self.omega,
            episode_seed=self.episode_seed,
            cfg=self.cfg,
            planets=new_planets,
            fleets=new_fleets,
            comet_paths=self.comet_paths,
            _outgoing_launches=new_outgoing,
            _planet_by_id=new_planet_by_id,
            _fleet_by_id=new_fleet_by_id,
            _comet_by_planet_id=dict(self._comet_by_planet_id),
            _ledger_cache=inherited_ledger,
            _timeline_cache=inherited_timeline,
            _combat_log_cache=inherited_combat,
        )

    def with_candidates(self, specs: Iterable["LaunchSpec"]) -> "World":
        """Apply multiple candidate launches in sequence. Each
        successive `with_candidate` re-uses the previous overlay's
        state (so the per-source ship deductions accumulate)."""
        w = self
        for s in specs:
            w = w.with_candidate(s)
        return w

    # ---------------------------------------------------------------
    # Phase 8 — multi-turn lookahead primitive
    # ---------------------------------------------------------------

    def snapshot_at(self, t: int,
                    *,
                    horizon: int = DEFAULT_LEDGER_HORIZON,
                    ) -> "World":
        """Re-anchor: return a new World as if `t` turns have elapsed.

        The trajectory layer already SCORES a bundle at a future
        horizon (`BundleEvaluator`). `snapshot_at` is the complement —
        produce a fresh World whose turn 0 is the parent's turn `t`,
        so the caller can chain a new BundleSearch from that rolled-
        forward state (multi-turn lookahead, idle-baseline differencing,
        "replay from here" workflows).
        Invariant pinned by tests:
            `snapshot_at(t).ownership_at(p, 0) ==
             self.ownership_at(p, t)` for every planet p (at the same
            horizon).
        Fleets that have already arrived by `t` are dropped (their
        effect is baked into the new planet states). Fleets still in
        flight at `t` are projected forward. Future-scheduled launches
        (`_outgoing_launches`) and `spawn_turn > t` synthetic fleets
        are carried over with their times shifted by `-t`. Comets
        whose path has exhausted by `t` are dropped.
        Caches are NOT inherited — the new World's timelines re-build
        from the rolled-forward anchor.
        """
        if t == 0:
            return self
        if t < 0:
            raise ValueError(f"snapshot_at requires t >= 0 (got {t})")

        # 1. Rebuild planets at turn t using closed-form position +
        # timeline-driven ownership/ships.
        new_planets: list[PlanetView] = []
        for p in self.planets:
            owner_t, ships_t = self.ownership_at(p.id, t, horizon=horizon)
            pos_t = self.planet_position(p.id, t)
            if pos_t is None:
                # Expired comet (path exhausted); drop the planet so
                # the new World has no stale comet reference.
                continue
            new_planets.append(replace(
                p,
                owner=int(owner_t),
                ships=float(ships_t),
                current_x=float(pos_t[0]),
                current_y=float(pos_t[1]),
            ))

        # 2. Rebuild fleets: drop those that arrived by t; project
        # those still in flight; shift future-scheduled (spawn_turn>t)
        # carriers forward by -t. Reuse the parent's ledger horizon
        # for the per-fleet ray-cast — fleets that don't hit a planet
        # within horizon turns from self.step are dropped (they were
        # already dropped by the parent's ledger).
        max_steps = min(int(horizon), DEFAULT_RAYCAST_STEPS)
        new_fleets: list[FleetView] = []
        for f in self.fleets:
            if f.ships <= 0:
                continue
            if f.spawn_turn > t:
                # Hasn't spawned yet at turn t; shift spawn_turn.
                # current_x/y is the position at f.spawn_turn (the
                # spawn position), so no position update needed.
                new_fleets.append(replace(f,
                                          spawn_turn=f.spawn_turn - t))
                continue
            # Already spawned at or before t. Check arrival.
            target_id, arrival_turn = _fleet_target_planet(
                self, f, max_steps=max_steps,
            )
            if arrival_turn is None:
                # Sun-killed, OOB, or no-planet-hit-within-horizon —
                # the fleet contributes nothing to any planet's
                # timeline (see `_build_full_ledger`'s same drop).
                # Conservative behaviour for snapshot_at: drop it.
                # The fleet may technically still be alive at t for
                # sub-horizon snapshots, but it's destined to die
                # without changing ownership state, so omitting it
                # from the new World preserves ownership invariants.
                continue
            if arrival_turn <= t:
                # Arrived; effect is baked into ownership_at(target, t).
                continue
            # Still in flight. Project position to turn t and
            # re-anchor spawn_turn to 0.
            pos = f.position_at(t)
            if pos is None:
                continue
            new_fleets.append(replace(f,
                                      current_x=float(pos[0]),
                                      current_y=float(pos[1]),
                                      spawn_turn=0))

        # 3. Future-scheduled launches: drop those whose launch_turn
        # has already passed (their effect is in ownership_at(src, t));
        # shift the rest.
        new_outgoing = tuple(
            (int(src), int(lt - t), int(ships))
            for (src, lt, ships) in self._outgoing_launches
            if lt > t
        )

        # 4. Comets: advance path_index; drop exhausted paths.
        new_comet_paths: list[CometPathView] = []
        for cp in self.comet_paths:
            new_idx = cp.path_index + t
            if new_idx < 0 or new_idx >= len(cp.path):
                continue
            new_comet_paths.append(replace(cp, path_index=new_idx))

        # 5. Rebuild indices (fresh dicts — never mutate parent's).
        planet_by_id = {p.id: p for p in new_planets}
        fleet_by_id = {f.id: f for f in new_fleets}
        comet_by_pid = {c.planet_id: c for c in new_comet_paths}

        return World(
            step=int(self.step + t),
            my_id=self.my_id,
            omega=self.omega,
            episode_seed=self.episode_seed,
            cfg=self.cfg,
            planets=tuple(new_planets),
            fleets=tuple(new_fleets),
            comet_paths=tuple(new_comet_paths),
            _outgoing_launches=new_outgoing,
            _planet_by_id=planet_by_id,
            _fleet_by_id=fleet_by_id,
            _comet_by_planet_id=comet_by_pid,
            _ledger_cache={},
            _timeline_cache={},
            _combat_log_cache={},
        )


# ---------------------------------------------------------------------------
# LaunchSpec — Phase 3 input type for `World.with_candidate`
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LaunchSpec:
    """One hypothetical fleet launch — primitive of a Bundle.

    `launch_turn` is the relative turn (from snapshot's obs.step) at
    which the launch happens. `0` = this turn (immediate); positive
    values represent FUTURE-TURN commitments — the planner's "save
    up production then strike" pattern. The trajectory-native chooser
    composes Bundles (sequences of LaunchSpecs at varying launch_turns)
    and scores the resulting trajectory.

    `owner` is who is launching — usually `world.my_id` for our own
    candidates; can be set to an opponent's id when overlaying their
    hypothetical action under a learned opp model.
    """
    src_id: int
    aim_angle: float
    ships: int
    owner: int
    launch_turn: int = 0


# ---------------------------------------------------------------------------
# PHASE 4 — SunFilter (closed-form, no false negatives)
# ---------------------------------------------------------------------------


class SunVerdict(enum.Enum):
    """Result of a SunFilter check.

    `SAFE` — the spec's fleet provably does NOT pass through the sun
    on its full infinite forward ray (zero false negatives).
    `HITS_SUN` — the fleet's ray intersects the sun's safety disc and
    the intersection lies ahead of the spawn.
    `UNCERTAIN` — reserved for future extensions where the closed-form
    can't decide (e.g. interaction with a moving target inside the sun
    zone). Phase 4 never returns this; callers should treat it as
    "not SAFE" via `SunFilter.is_safe`.
    """
    SAFE = 0
    HITS_SUN = 1
    UNCERTAIN = 2


@dataclass(frozen=True)
class SunFilter:
    """Hard O(1) sun-safety predicate.

    Closed-form ray-vs-disc geometry: a launch from `spawn` along
    `direction` traces an infinite half-line; the sun is a disc of
    radius `SUN_RADIUS + safety_margin` centered at `(CENTER, CENTER)`.
    The fleet's ray crosses the disc iff the projection of
    `(sun_center - spawn)` onto `direction` is non-negative AND the
    perpendicular distance from `sun_center` to the line is less than
    `SUN_RADIUS + safety_margin`.

    This covers the full overshoot tail — the existing
    `lib/geometry.path_clears_sun` only checks the segment from
    source to the lead-aim target, missing the case where the fleet
    over-shoots through the sun BEYOND the target. That gap is the
    documented load-bearing source of the 6%+ sun-clip rate.

    Invariant (enforced by Hypothesis fuzz): if `is_safe(spec) is
    True`, the fleet provably does NOT die in the sun on its
    forward ray. Zero false negatives.

    `safety_margin` defaults to 0.5 to match
    `lib/trajectory.SUN_SAFETY` (the cushion that absorbs float drift
    on tangent paths). Tuning upward (e.g. 1.0) increases conservatism
    at the cost of rejecting more borderline-safe launches.
    """
    world: "World"
    safety_margin: float = SUN_SAFETY

    def check(self,
              spec: LaunchSpec,
              *,
              arrival_xy: Optional[tuple[float, float]] = None,
              ) -> SunVerdict:
        """Verdict for `spec`. Pure function of (spec, world).

        `arrival_xy` is accepted for API compatibility with callers
        that pass the lead-aim target endpoint (e.g. mechanism layer);
        Phase 4's implementation ignores it because the closed-form
        on the infinite ray is strictly safer (covers overshoot).
        Phase 5+ may use it to bound the check window.
        """
        src = self.world.planet_by_id(spec.src_id)
        if src is None:
            # Unknown source — can't predict; bias toward UNCERTAIN
            # (callers treat as not-safe via is_safe).
            return SunVerdict.UNCERTAIN
        return _ray_sun_verdict(
            src_x=src.current_x,
            src_y=src.current_y,
            src_radius=src.radius,
            aim_angle=spec.aim_angle,
            safety_margin=self.safety_margin,
        )

    def is_safe(self, spec: LaunchSpec) -> bool:
        """Convenience: True iff `check(spec) == SAFE`. Treats
        UNCERTAIN as not-safe — the load-bearing invariant is that
        TRUE means provably-safe."""
        return self.check(spec) == SunVerdict.SAFE


def _ray_sun_verdict(*,
                     src_x: float,
                     src_y: float,
                     src_radius: float,
                     aim_angle: float,
                     safety_margin: float = SUN_SAFETY,
                     ) -> SunVerdict:
    """Pure-geometry ray-vs-sun check. Spawn position = source center
    + (radius + 0.1) * direction (mirrors env's process_moves).

    Returns SAFE iff (proj < 0) OR (perp_dist² >= threshold²).
    Returns HITS_SUN otherwise.
    """
    cos_a = math.cos(aim_angle)
    sin_a = math.sin(aim_angle)
    spawn_x = src_x + cos_a * (src_radius + 0.1)
    spawn_y = src_y + sin_a * (src_radius + 0.1)

    # Vector from spawn to sun center.
    dx = CENTER - spawn_x
    dy = CENTER - spawn_y

    # Projection of (sun - spawn) onto direction = signed forward
    # distance to the closest-approach point.
    proj = dx * cos_a + dy * sin_a
    if proj < 0.0:
        # Sun is behind the fleet's heading — the forward ray never
        # gets any closer than the spawn distance, which is > 0.
        return SunVerdict.SAFE

    # Perpendicular distance² from sun center to the fleet's line.
    perp_sq = dx * dx + dy * dy - proj * proj
    threshold = SUN_RADIUS + safety_margin
    if perp_sq >= threshold * threshold:
        return SunVerdict.SAFE
    return SunVerdict.HITS_SUN


__all__ = [
    "BOARD_SIZE", "CENTER", "SUN_RADIUS", "ROTATION_RADIUS_LIMIT",
    "SUN_SAFETY", "DEFAULT_LEDGER_HORIZON", "DEFAULT_RAYCAST_STEPS",
    "GameConfig",
    "PlanetView", "FleetView", "CometPathView", "Arrival", "LaunchSpec",
    "SunVerdict", "SunFilter",
    "CombatOutcome",
    "Bundle", "BundleScore", "BundleEvaluator", "BundleSearch",
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
        # Future-launch fleets (spawn_turn > 0) don't exist until then.
        # Phase 1 in-flight fleets have spawn_turn=0 so this is a no-op
        # for the legacy path.
        if f_old is None or f_new is None:
            continue

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
            # Source-planet skip on the spawn step: mirrors the env's
            # explicit `if pid == src_id and step == 0: continue`
            # (lib/trajectory.py:133-135). For fresh launches (Phase 3
            # synthetic fleets), the fleet starts just outside its
            # source — without this skip the swept-pair check would
            # spuriously hit the source itself. For already-in-flight
            # fleets the check is a no-op (the fleet has moved away
            # from `from_planet_id`).
            if pid == fleet.from_planet_id and t == fleet.spawn_turn + 1:
                continue
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
    outgoing: Iterable[tuple[int, int]] = (),
) -> dict:
    """Per-planet ownership/garrison timeline under integer-tick semantics.

    Mirrors `lib/world_model.simulate_planet_timeline` exactly, plus
    Phase 7's outgoing-launch handling for future-turn launches:
      1. At t=0, record current (owner, ships).
      2. For t in [1, horizon]:
         a. Process outgoing launches at turn t (env's `process_moves`):
            if still owned by the launch owner with enough ships,
            deduct. Otherwise the launch is silently dropped (invalid
            commitment — bundle validator should have caught it).
         b. If currently owned (owner != -1), garrison += production.
         c. Resolve same-step arrivals via `lib.combat.resolve_arrivals`.
         d. Record (owner_at[t], ships_at[t]).

    `outgoing` is `[(turn, ships), ...]` of THIS planet's future
    launches recorded in `World._outgoing_launches`. Phase 1-6 callers
    pass `()` to preserve the no-outgoing-launch semantics.

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

    outgoing_by_turn: dict[int, int] = defaultdict(int)
    for out_turn, out_ships in outgoing:
        if out_ships <= 0:
            continue
        ot = max(1, int(out_turn))
        if ot > h:
            continue
        outgoing_by_turn[ot] += int(out_ships)

    owner = planet.owner
    garrison = float(planet.ships)
    owner_at = {0: owner}
    ships_at = {0: max(0.0, garrison)}

    for t in range(1, h + 1):
        # 1. Process outgoing launches (process_moves): only if still
        # owned (a launch from a captured planet is invalidated).
        out_ships = outgoing_by_turn.get(t, 0)
        if out_ships > 0 and owner != -1:
            garrison = max(0.0, garrison - out_ships)
        # 2. Production.
        if owner != -1:
            garrison += planet.production
        # 3. Combat.
        group = by_turn.get(t, [])
        if group:
            owner, garrison = resolve_arrivals(owner, garrison, group)
        owner_at[t] = owner
        ships_at[t] = max(0.0, garrison)

    return {"owner_at": owner_at, "ships_at": ships_at, "horizon": h}


# ---------------------------------------------------------------------------
# PHASE 5 — Per-planet combat outcomes (lazy)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CombatOutcome:
    """Detailed record of a single per-planet combat event.

    Returned by `World.combat_at(pid, t)` when there is at least one
    arrival at turn `t`. Reports both the resolution (`winner_owner`,
    `surviving_ships`) and the participants (the runner-up attacker
    and the full attacker list) so the shot validator (Step 2) and
    the value head (Step 3) can label their data.

    Definitions:
    - `turn` — the relative turn the combat occurred at (matches the
      `t` passed to `combat_at`).
    - `pre_garrison_owner` / `pre_garrison_ships` — state BEFORE the
      arrival group resolves (after production accrual for that turn).
    - `winner_owner` / `surviving_ships` — state AFTER combat.
    - `attackers` — `(owner, ships)` per attacking owner-group, sorted
      descending by ship count. `lib.combat.resolve_arrivals` groups
      same-owner arrivals into one entry per owner.
    - `runner_up_owner` / `runner_up_ships` — the second-largest
      attacker. `-1 / 0` if there's only one attacker.
    - `is_tie` — True iff top_ships == second_ships, in which case all
      attackers are destroyed (rule 4 in `lib/combat.py`).
    """
    turn: int
    pre_garrison_owner: int
    pre_garrison_ships: float
    winner_owner: int
    surviving_ships: float
    attackers: tuple[tuple[int, int], ...]
    runner_up_owner: int
    runner_up_ships: int
    is_tie: bool


def _simulate_timeline_with_combat_log(
    planet: "PlanetView",
    arrivals: Iterable[Arrival],
    horizon: int,
    outgoing: Iterable[tuple[int, int]] = (),
) -> tuple[dict, dict[int, CombatOutcome]]:
    """Variant of `_simulate_planet_timeline` that ALSO emits a
    per-turn combat log. Used by `World.combat_at`.

    Identical state-evolution semantics to `_simulate_planet_timeline`
    — the combat log is a side-channel. Asserted equivalent in
    `tests/test_trajectory_layer_combat.py`.
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

    outgoing_by_turn: dict[int, int] = defaultdict(int)
    for out_turn, out_ships in outgoing:
        if out_ships <= 0:
            continue
        ot = max(1, int(out_turn))
        if ot > h:
            continue
        outgoing_by_turn[ot] += int(out_ships)

    owner = planet.owner
    garrison = float(planet.ships)
    owner_at = {0: owner}
    ships_at = {0: max(0.0, garrison)}
    combat_log: dict[int, CombatOutcome] = {}

    for t in range(1, h + 1):
        # 1. Outgoing launches (env's process_moves).
        out_ships = outgoing_by_turn.get(t, 0)
        if out_ships > 0 and owner != -1:
            garrison = max(0.0, garrison - out_ships)
        # 2. Production.
        if owner != -1:
            garrison += planet.production
        # 3. Combat.
        group = by_turn.get(t, [])
        if group:
            # Aggregate same-owner arrivals to mirror resolve_arrivals'
            # internal grouping, so the `attackers` tuple matches the
            # combat-detail.
            by_attacker: dict[int, int] = {}
            for arr_owner, ships in group:
                if ships <= 0:
                    continue
                by_attacker[arr_owner] = (by_attacker.get(arr_owner, 0)
                                            + int(ships))
            ranked = sorted(by_attacker.items(),
                             key=lambda kv: kv[1], reverse=True)
            attackers_tuple = tuple(ranked)
            runner_owner = ranked[1][0] if len(ranked) > 1 else -1
            runner_ships = ranked[1][1] if len(ranked) > 1 else 0
            is_tie = (len(ranked) > 1
                      and ranked[0][1] == ranked[1][1])

            pre_owner = owner
            pre_ships = garrison
            owner, garrison = resolve_arrivals(owner, garrison, group)

            combat_log[t] = CombatOutcome(
                turn=t,
                pre_garrison_owner=pre_owner,
                pre_garrison_ships=max(0.0, pre_ships),
                winner_owner=owner,
                surviving_ships=max(0.0, garrison),
                attackers=attackers_tuple,
                runner_up_owner=runner_owner,
                runner_up_ships=int(runner_ships),
                is_tie=is_tie,
            )
        owner_at[t] = owner
        ships_at[t] = max(0.0, garrison)

    timeline = {"owner_at": owner_at, "ships_at": ships_at, "horizon": h}
    return timeline, combat_log


# ---------------------------------------------------------------------------
# PHASE 7b — Bundle + BundleEvaluator (trajectory-native value function)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bundle:
    """A planner's commitment: a sequence of launches across future
    turns. The trajectory-native chooser composes Bundles and scores
    the world trajectory each produces; the best-scoring bundle's
    earliest launches get emitted as actions this turn.

    Empty bundle == no-op (the agent does nothing this turn AND has
    no future commitments). Apply via `Bundle.apply(world)` to get
    the overlay World.
    """
    launches: tuple[LaunchSpec, ...] = ()

    def apply(self, world: "World") -> "World":
        """Return the World with all this bundle's launches overlaid."""
        return world.with_candidates(self.launches)

    @property
    def is_empty(self) -> bool:
        return len(self.launches) == 0

    @property
    def first_launch_turn(self) -> Optional[int]:
        """Earliest launch_turn across the bundle's specs, or None if
        the bundle is empty. Useful for "what action does this bundle
        emit THIS turn?" — the launches at turn 0."""
        if not self.launches:
            return None
        return min(s.launch_turn for s in self.launches)

    def specs_at_turn(self, turn: int) -> tuple[LaunchSpec, ...]:
        """All specs scheduled at the given launch_turn. The agent
        loop calls `specs_at_turn(0)` each turn to get the actions to
        emit; advances all other commitments forward by one turn."""
        return tuple(s for s in self.launches if s.launch_turn == turn)

    def shift_forward(self, steps: int) -> "Bundle":
        """Advance every launch's `launch_turn` by `-steps`. Specs
        whose `launch_turn` would become negative are DROPPED (those
        already fired in prior turns and persist in the env, not in
        the bundle).

        Per-turn agent loop usage: at the end of turn T, store the
        chosen bundle; at the start of turn T+1, call
        `stored.shift_forward(1)` to get the carry-over plan to pass
        as `seed_bundle` to BundleSearch.
        """
        if steps < 0:
            raise ValueError(f"steps must be >= 0 (got {steps})")
        if steps == 0:
            return self
        kept = tuple(
            replace(s, launch_turn=s.launch_turn - steps)
            for s in self.launches
            if s.launch_turn - steps >= 0
        )
        return Bundle(launches=kept)


@dataclass(frozen=True)
class BundleScore:
    """Decomposed bundle value, for diagnosis + tuning + ladder
    calibration (each component is its own metric in the pre-
    registration discipline).

    Field semantics — terminal vs path-integrated:
    - `ship_delta`: TERMINAL (us − sum of others) at turn K. Ships
      are transit-state mid-rollout, so the path integral over ships
      has no clean interpretation; we keep this terminal.
    - `planet_delta`: PATH-INTEGRATED. Sum over t in [1..K] of
      (my_planets_t − opp_planets_t). Units = planet-turns. A planet
      we hold for all K turns contributes +K; a planet captured at
      t=5 and held to K contributes (K-5); a planet captured then
      recaptured contributes only the held window. Magnitude scales
      with K — interpret accordingly, NOT as a planet count diff.
    - `production_delta`: PATH-INTEGRATED. Same shape as
      planet_delta but weighted by per-planet `production` rate.
      Units = production-turns.
    - `eliminations`: TERMINAL count of opponents who started with
      planets and own none at turn K.
    - `total`: weighted sum (`ship_delta + planet_weight ·
      planet_delta + production_weight · production_delta +
      elimination_bonus · eliminations`). This is the scalar the
      chooser argmax's over.
    """
    ship_delta: float
    planet_delta: float
    production_delta: float
    eliminations: int
    total: float


@dataclass(frozen=True)
class BundleEvaluator:
    """Score a bundle by reading the trajectory layer's K-turn rollout.

    Hybrid terminal-vs-path-integrated leaf:
    - `ship_delta` is the TERMINAL (turn K) ship-count diff.
    - `planet_delta` and `production_delta` are PATH-INTEGRATED —
      summed over t in [1..K] of the per-turn diff. Earlier captures
      accumulate more credit (more held turns); recaptured planets
      contribute only the held window. This is the H3 fix to the
      pre-2026-05-18 terminal-only scoring, which was blind to mid-
      rollout bleed and gave equal credit to captures-at-t=5 vs
      captures-at-t=25.
    - `eliminations` is a terminal count; the elimination_bonus
      rewards a full opp seat wipe at K.

    `horizon` is the look-ahead in turns. Default 30 — long enough to
    capture cross-board strikes (board-diagonal ETA at speed=1.66 is
    ~85 steps, but most strategic launches are <30 turns), short
    enough that opponent uncertainty doesn't dominate the score.

    Default weights are hand-tuned starting points; Step 3's learned
    value head replaces this function with a trained network later.
    """
    horizon: int = 30
    planet_weight: float = 5.0
    # Coefficient on the path-integrated production_delta (the sum
    # over t in [1..horizon] of `my_prod_t - opp_prod_t`). Default
    # 1.0 → one ship-equivalent per held production-turn. Combined
    # with planet_weight=5, a +2-prod planet captured at turn 5 and
    # held through K=30 yields ~25 turns × (1·2 + 5·1) = 175
    # ship-equivalents over the rollout window — comparable to the
    # ~50-ship launch cost. Adjust per A/B; the magnitude is
    # PROPORTIONAL to horizon, so a horizon change rebalances this
    # weight against ship_delta (which stays terminal).
    production_weight: float = 1.0
    elimination_bonus: float = 200.0
    # Phase B me-followup mode. "off" (default) preserves Phase 7c-8b
    # scoring exactly; "lite" applies a predicted ME reactive bundle
    # (event-driven lite_greedy from my seat) on top of `bundle`
    # before reading the path integral. Bug #14 me-half target: makes
    # the rollout self-consistent on both seats so the score function
    # stops treating my sources as drained-forever after each launch.
    my_followup_mode: str = "off"
    # Phase E Phase 1 (2026-05-18): coordinated-joint-capture bonus.
    # When `joint_bonus > 0`, score() detects bundles with 2+ launches
    # hitting the same enemy/neutral target where (sum_delivered >
    # defenders) AND (no single launch's ships > defenders). Such joints
    # would NOT capture in solo enumeration; they require coordination.
    # The bonus is `joint_bonus * (production * remaining_horizon +
    # planet_weight)` per detected joint, ADDITIVE to the path-integral
    # capture credit. Default 0.0 preserves prior behavior; live
    # config: 0.5. Phase 0 diagnostic showed 21.3% of bundle's ships
    # bounce off enemy planets — the joint bonus + Phase 1a frontier
    # seeding lets the search find pair-cooperative captures the
    # current beam misses.
    joint_bonus: float = 0.0
    # Phase E Phase 2 (2026-05-18): bounce-penalty for failed captures.
    # When `bounce_weight > 0`, score() subtracts `bounce_weight * ships`
    # from total for each launch where (delivered_ships <= defenders at
    # arrival). Ported from `lib/value_heads.composite_capture_value`
    # (the live champion's waste penalty) with the same default 0.5.
    # Phase 0 diagnostic + Phase 1 pick-rate post-mortem showed bundle
    # launches 368 fleets across 16 games at over-defended enemy
    # planets that all bounce; current scorer treats these as silent
    # zero (planet stays opp-owned, no penalty). The penalty pushes the
    # chooser away from these bounces toward either empty bundle or
    # correctly-sized solos elsewhere. Default 0.0 preserves prior
    # behavior.
    bounce_weight: float = 0.0

    def score(self, world: "World", bundle: Bundle,
              *, my_id: Optional[int] = None,
              opp_overlays: Optional[Mapping[int, Bundle]] = None,
              ) -> BundleScore:
        """Compute the bundle's value. `my_id` defaults to
        `world.my_id` — pass an explicit value to score from a
        different seat's perspective (e.g. an opponent-bundle search
        used by the learned opp model).

        `opp_overlays` (Phase 8): per-opponent predicted bundles to
        apply ON TOP of `bundle` before reading ownership at horizon.
        Maps opp_id → Bundle. Each opp bundle's launches carry
        `owner=opp_id`, so World.with_candidate's ownership check
        gates them correctly. An opp bundle that becomes infeasible
        against the composed overlay (e.g. its source got captured
        by `bundle` before the opp's launch_turn) is dropped silently
        — the score then reflects a partial-counterplay world, which
        is the worst case for that opp.
        """
        if my_id is None:
            my_id = world.my_id
        overlay = bundle.apply(world)
        # Phase B me-followup: predict my reactive launches across the
        # rollout and apply them BEFORE opp overlays. Guarded by
        # `my_id == world.my_id` so the inner opp-search (mirror mode,
        # which calls score with my_id=opp_id but unchanged world)
        # does NOT recursively run me-followup from the opp's seat.
        if self.my_followup_mode == "lite" and my_id == world.my_id:
            followup = predict_my_followup_via_event_driven_lite_greedy(
                overlay, my_id=my_id, horizon=self.horizon,
            )
            if not followup.is_empty:
                try:
                    overlay = followup.apply(overlay)
                except ValueError:
                    pass
        if opp_overlays:
            for opp_id, opp_bundle in opp_overlays.items():
                if opp_bundle.is_empty:
                    continue
                try:
                    overlay = opp_bundle.apply(overlay)
                except ValueError:
                    continue

        # Path-integrated production: for each non-comet planet, sum
        # `+production` per turn it's ours and `-production/num_opps`
        # per turn it's owned by an opp, across [1..horizon]. The
        # integral naturally rewards EARLY captures (more turns of
        # accumulated credit) and PENALISES bleed (a planet held
        # turns 5-20 then lost contributes only 16 turns of credit,
        # not the full horizon). Terminal-only scoring is blind to
        # both effects; that blindness was the H3 root cause behind
        # bundle 0/32 vs v7_0.
        # Cost: per planet, the timeline is already built+cached on
        # first `ownership_at` call; the per-turn dict reads are
        # O(1). For 24 planets × 30 horizon = 720 dict reads per
        # score, ~50 us.
        my_planets_path = 0.0
        opp_planets_path = 0.0
        my_prod_path = 0.0
        opp_prod_path = 0.0
        for p in overlay.planets:
            if p.is_comet:
                continue
            timeline = overlay._timeline_for(p.id, self.horizon)
            if timeline is None:
                continue
            owner_at = timeline["owner_at"]
            h = timeline["horizon"]
            for t in range(1, h + 1):
                owner_t = owner_at[t]
                if owner_t == my_id:
                    my_planets_path += 1
                    my_prod_path += p.production
                elif owner_t != -1:
                    opp_planets_path += 1
                    opp_prod_path += p.production

        # Terminal-state ship counts (ships are transit-state mid-
        # rollout; the path integral over ships doesn't have a clean
        # interpretation, so we keep ship_delta as terminal).
        ships_by_owner_K: dict[int, float] = defaultdict(float)
        planets_by_owner_K: dict[int, int] = defaultdict(int)
        for p in overlay.planets:
            if p.is_comet:
                continue
            owner, ships = overlay.ownership_at(p.id, self.horizon)
            if owner != -1:
                ships_by_owner_K[owner] += ships
                planets_by_owner_K[owner] += 1

        my_ships = ships_by_owner_K.get(my_id, 0.0)
        other_ships = sum(v for k, v in ships_by_owner_K.items() if k != my_id)
        ship_delta = my_ships - other_ships

        # Path-integrated planet & production deltas — these are the
        # values the total uses (and the values the chooser ranks
        # against). Surfaced via BundleScore for diagnostic stability.
        planet_delta_path = my_planets_path - opp_planets_path
        production_delta_path = my_prod_path - opp_prod_path

        # Count opponents who started with planets but have none at K.
        initial_opp_owners: set[int] = set()
        for p in world.planets:
            if p.is_comet:
                continue
            if p.owner != -1 and p.owner != my_id:
                initial_opp_owners.add(p.owner)
        eliminations = sum(1 for o in initial_opp_owners
                           if planets_by_owner_K.get(o, 0) == 0)

        # Phase E Phase 1 (2026-05-18): joint coordination bonus. Detects
        # bundles whose 2+ launches at the same enemy/neutral target
        # collectively succeed where no single launch would. ADDITIVE
        # to the path-integral credit — the existing planet_delta_path
        # already credits the resulting capture; this bonus rewards the
        # coordination act itself so the search prefers joints over
        # equally-large solos at less defended (lower-strategic-value)
        # targets. Cheap: O(launches) calls to predict_fleet_fate (~1-2ms
        # at typical bundle size 2-4).
        joint_bonus_total = 0.0
        if self.joint_bonus > 0.0 and len(bundle.launches) >= 2:
            for tgt_id, arr_turn, _sum_s, _max_s, tgt in (
                self._detect_joint_captures(world, bundle, my_id)
            ):
                remaining = max(0, self.horizon - arr_turn)
                joint_value = tgt.production * remaining + self.planet_weight
                joint_bonus_total += self.joint_bonus * joint_value

        # Phase E Phase 2 (2026-05-18): bounce penalty for solo launches
        # that would fail to capture. Mirrors composite_capture_value's
        # `0.5 × ships` waste penalty (the live champion's existing
        # mechanism). SUBTRACTIVE from total — pushes the chooser away
        # from launches whose ship count is insufficient to overcome
        # the predicted defenders at arrival. Joint-aware: launches
        # that contribute to a detected joint capture are exempted
        # (the joint succeeds, so individual under-cap is not a bounce).
        bounce_penalty_total = 0.0
        if self.bounce_weight > 0.0 and bundle.launches:
            bounce_penalty_total = self._compute_bounce_penalty(
                world, bundle, my_id,
            )

        # All deltas summed over [1..K] (path integral) except
        # ship_delta (terminal — see comment above) and eliminations
        # (terminal — opp is eliminated or not).
        total = (ship_delta
                 + self.planet_weight * planet_delta_path
                 + self.production_weight * production_delta_path
                 + self.elimination_bonus * eliminations
                 + joint_bonus_total
                 - bounce_penalty_total)

        return BundleScore(
            ship_delta=ship_delta,
            planet_delta=planet_delta_path,
            production_delta=production_delta_path,
            eliminations=eliminations,
            total=total,
        )

    def _detect_joint_captures(
        self, world: "World", bundle: Bundle, my_id: int,
    ) -> list[tuple[int, int, int, int, Any]]:
        """Group bundle launches by predicted hit-planet. Yield detected
        joint captures: tuples of (target_id, arrival_turn, sum_ships,
        max_indiv_ships, target_planet). A "joint capture" is a target
        where (sum_delivered > defenders) AND (no individual launch's
        ships > defenders) — i.e. coordination is REQUIRED to capture.

        Targets currently owned by `my_id` are excluded (joint
        reinforcement is not the same coordination property and is
        already path-integrated). Neutral and enemy targets both
        qualify; the existing path-integral credits the capture once,
        the joint bonus rewards the coordination on top.

        Uses a per-step raycast (matches predict_fleet_fate's geometry
        for static planets, approximates omega>0 by using planets'
        `current_*` positions as static). For Phase 1 v1, omega>0
        accuracy is best-effort — under-counting joints on orbital
        targets is preferable to mis-detecting them. Scope-limited to
        `launch_turn=0` launches.
        """
        if len(bundle.launches) < 2:
            return []
        omega = float(getattr(world, "omega", 0.0) or 0.0)
        current_step = int(getattr(world, "step", 0) or 0)
        rot_offset = 1 if current_step == 0 else 0
        planets = [p for p in world.planets if not p.is_comet]

        # by_target stores (arrival_turn, ships, src_id) per launch so we
        # can require >=2 DISTINCT sources before tagging a joint. Same-
        # source pseudo-joints (e.g. two 3-ship launches from src=0 at
        # identical angles, emerging from the search's ADD iterations
        # rather than from `_enumerate_joint_seeds`) are NOT real
        # coordination — splitting one launch into two smaller ones
        # provides no strategic benefit and is strictly worse on the
        # fleet_speed curve. Origin: Phase E Phase 1 single-state diag,
        # seed=42 turn=5 (audit/2026-05-18-phase-e-phase1-diag.md).
        by_target: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        for launch in bundle.launches:
            if launch.owner != my_id:
                continue
            if int(launch.launch_turn) != 0:
                continue  # Phase 1: same-turn launches only
            src = None
            for p in planets:
                if p.id == int(launch.src_id):
                    src = p
                    break
            if src is None:
                continue
            hit_pid, arrival_step = _raycast_first_planet_hit(
                src, float(launch.aim_angle), int(launch.ships),
                planets, omega, rot_offset,
            )
            if hit_pid is None or arrival_step is None:
                continue
            if int(hit_pid) == int(launch.src_id):
                continue
            arrival_turn = int(launch.launch_turn) + int(arrival_step)
            by_target[int(hit_pid)].append(
                (arrival_turn, int(launch.ships), int(launch.src_id))
            )

        joints: list[tuple[int, int, int, int, Any]] = []
        for tgt_id, arrivals in by_target.items():
            if len(arrivals) < 2:
                continue
            # Require >=2 DISTINCT source planets — joint = multi-source
            # coordination, not multi-fleet from one source.
            distinct_sources = {src for _, _, src in arrivals}
            if len(distinct_sources) < 2:
                continue
            tgt = None
            for p in planets:
                if p.id == tgt_id:
                    tgt = p
                    break
            if tgt is None:
                continue
            if tgt.owner == my_id:
                continue  # don't bonus joint reinforcement of own planet
            defenders = int(tgt.ships)
            sum_ships = sum(s for _, s, _src in arrivals)
            max_ships = max(s for _, s, _src in arrivals)
            if sum_ships > defenders and max_ships <= defenders:
                min_arrival = min(t for t, _, _src in arrivals)
                joints.append(
                    (tgt_id, min_arrival, sum_ships, max_ships, tgt)
                )
        return joints

    def _compute_bounce_penalty(
        self, world: "World", bundle: Bundle, my_id: int,
    ) -> float:
        """Sum `bounce_weight * ships` over launches that fail to capture
        their target (delivered ships <= predicted defenders at arrival).

        Joint-aware: launches contributing to a detected distinct-source
        joint capture are exempted (the joint succeeds, so individual
        under-cap is not a true bounce). When `joint_bonus == 0` this
        exemption check still runs but joints rarely emerge from regular
        search; the cost is one extra `_detect_joint_captures` call
        (~1-2ms) per scored bundle when bounce_weight > 0.

        Defender count at arrival: read from `world.ownership_at(target,
        arrival_turn)`. This counts the pre-bundle defender state
        (background production + any opp in-flight fleets) but NOT
        our other bundle launches — so each leg is checked
        independently against its own arrival's opp resistance.
        """
        if not bundle.launches:
            return 0.0
        omega = float(getattr(world, "omega", 0.0) or 0.0)
        current_step = int(getattr(world, "step", 0) or 0)
        rot_offset = 1 if current_step == 0 else 0
        planets = [p for p in world.planets if not p.is_comet]
        planet_by_id_local = {p.id: p for p in planets}

        # Identify launches that are part of a successful joint —
        # exempt them from the bounce penalty.
        joint_launch_ids: set[int] = set()
        if len(bundle.launches) >= 2:
            joints = self._detect_joint_captures(world, bundle, my_id)
            for tgt_id, _arr, _sum, _max, _tgt in joints:
                # Re-raycast each launch to find which ones hit this
                # target — those are the joint partners.
                for i, launch in enumerate(bundle.launches):
                    if launch.owner != my_id:
                        continue
                    if int(launch.launch_turn) != 0:
                        continue
                    src = planet_by_id_local.get(int(launch.src_id))
                    if src is None:
                        continue
                    hit_pid, _step = _raycast_first_planet_hit(
                        src, float(launch.aim_angle), int(launch.ships),
                        planets, omega, rot_offset,
                    )
                    if hit_pid is not None and int(hit_pid) == tgt_id:
                        joint_launch_ids.add(i)

        total_penalty = 0.0
        for i, launch in enumerate(bundle.launches):
            if launch.owner != my_id:
                continue
            if int(launch.launch_turn) != 0:
                continue
            if i in joint_launch_ids:
                continue  # part of a successful joint
            src = planet_by_id_local.get(int(launch.src_id))
            if src is None:
                continue
            hit_pid, arrival_step = _raycast_first_planet_hit(
                src, float(launch.aim_angle), int(launch.ships),
                planets, omega, rot_offset,
            )
            if hit_pid is None or arrival_step is None:
                continue
            if int(hit_pid) == int(launch.src_id):
                continue
            tgt = planet_by_id_local.get(int(hit_pid))
            if tgt is None:
                continue
            if tgt.owner == my_id:
                continue  # reinforcing own planet — not a bounce
            # Predicted defenders at arrival via trajectory_layer's
            # native arrival prediction (accounts for opp production
            # + any opp in-flight fleets already in obs).
            try:
                owner_at_arrival, defenders_at_arrival = world.ownership_at(
                    int(hit_pid), int(arrival_step),
                )
            except Exception:
                continue
            # If our other prior launches already captured it,
            # owner_at_arrival could be my_id — but we're reading
            # PRE-bundle world, so this only fires if obs.fleets had
            # an in-flight friendly launch from before.
            if owner_at_arrival == my_id:
                continue
            if int(launch.ships) <= int(defenders_at_arrival):
                total_penalty += self.bounce_weight * int(launch.ships)
        return total_penalty


# ---------------------------------------------------------------------------
# PHASE 7c — BundleSearch (trajectory-native chooser)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BundleSearch:
    """Beam search over candidate bundles. Replaces the drop-one
    chooser as the trajectory-native action-selection primitive.

    Algorithm (v1):
      1. Score the empty bundle as the baseline.
      2. Enumerate candidate single-spec extensions: for each owned
         source, propose launches at the top-N closest non-friendly
         targets. Filter via SunFilter.
      3. Beam-extend the current frontier: for each (score, bundle) in
         the beam, try adding each candidate, score the extended
         bundle. Keep top-`beam_width` extensions globally.
      4. Track the highest-scoring bundle ever seen across all
         iterations (including the baseline).
      5. Repeat up to `max_depth` times.

    Beam search (not greedy) is required because gang-up patterns are
    structurally invisible to greedy: a single 25-ship launch at a
    40-ship target FAILS and scores worse than no-op; but TWO 25-ship
    launches together succeed and score above no-op. Beam width >= 2
    lets the search keep the (otherwise pruned) first launch alive
    long enough to discover the gang-up.

    Phase 7c shipped `launch_turn=0` only. Phase 7d adds the
    `launch_turns` knob — the search now considers DELAYED launches
    timed to arrive coordinated with other ships. This is the
    "idle far planets activate" pattern: a 60-ship planet 40 steps
    from any target has no good single-turn action, but with
    `launch_turns=(0, 5, 10)` the search can find a `launch_turn=10`
    spec that arrives coordinated with a closer source's
    `launch_turn=0` arrival.

    Phase 7e adds seed-and-extend semantics. `search(...,
    seed_bundle=...)` lets the caller carry a previously-chosen
    plan forward across turns: the agent loop stores last turn's
    bundle, calls `.shift_forward(1)` to advance it by one game
    step, and passes the result as `seed_bundle`. The search seeds
    its frontier with BOTH the empty bundle AND the seed, then
    explores TWO neighbor kinds per iteration:
      - ADDS: bundle + new candidate spec (Phase 7c behaviour).
      - DROPS: bundle - one existing launch (lets the search retract
        a seed launch that no longer fits — e.g. its target was
        captured by an ally, or the source got attacked).
    Swap-a-seed-launch (replace target / ship count of one seed
    launch) emerges as drop-then-add over two iterations.

    Knobs:
    - `max_depth`: max bundle size. Each iteration extends the beam,
      so search cost is O(max_depth · beam_width · candidates ·
      score_time).
    - `beam_width`: how many partial-bundles to keep between
      iterations. Must be >= 2 to discover gang-ups.
    - `candidates_per_source`: top-N closest targets per source. The
      target pool is the UNIFORM set of all non-comet planets (own,
      enemy, neutral); top-N is a pure budget throttle, not a
      strategy filter. Defense / staging / attack all emerge from
      the scorer's ΔP(win) ranking — never carve them at enumeration.
    - `launch_turns`: delayed-launch enumeration. Each candidate
      (source, target) pair generates one LaunchSpec per launch_turn
      in this tuple. Default `(0,)` preserves Phase 7c behaviour;
      `(0, 5, 10, 15)` enables coordinated-arrival planning.
    - `min_source_ships`: skip sources too weak to launch from. Also
      acts as the reserve kept at the source after any launch (the
      ship_ratios apply to `avail - min_source_ships`, not raw avail).
    - `ship_ratios`: fractional grid of ships to commit per launch,
      applied to the source's usable budget `(avail -
      min_source_ships)`. Default `(0.5, 1.0)` emits two siblings
      per (src, tgt, launch_turn): half-commit / full-commit. The
      scorer ranks across them via `ownership_at`-driven planet/ship
      deltas — half-commit wins on cheap captures (fewer ships
      wasted if it captures alone), full-commit wins on gang-ups +
      defense reinforcement. A 3-ratio grid `(0.25, 0.5, 1.0)` was
      tried first and dropped: greedy beam favours the smallest
      ratio's better independent score and the full-commit variant
      gets pruned before the next iteration can pair it into a
      gang-up. Two ratios is the sweet spot that keeps the gang-up
      variant in the frontier under beam_width=4.
    """
    evaluator: BundleEvaluator = field(default_factory=BundleEvaluator)
    max_depth: int = 3
    beam_width: int = 4
    candidates_per_source: int = 5
    launch_turns: tuple[int, ...] = (0,)
    min_source_ships: int = 2
    # Reserve N ships at the source AFTER any launch. The ship_ratios
    # apply to `(avail - reserve_ships_at_source)`, not raw avail.
    # Decoupled from `min_source_ships` (which filters sources too
    # weak to launch at all) per Piece 7 root-cause #1: the conflated
    # reserve=2 caused 10-ship sources to top out at 8-ship launches,
    # missing the 9-ship captures the scorer would have rewarded.
    reserve_ships_at_source: int = 1
    ship_ratios: tuple[float, ...] = (0.5, 1.0)
    sun_safety: float = SUN_SAFETY
    # Phase E Phase 1a (2026-05-18): how many explicit joint-pair
    # candidates to pre-seed into the search frontier (in addition to
    # empty + seed_bundle + the regular ADD/DROP iterations). Each
    # joint-pair is a 2-launch Bundle where neither solo would capture
    # the target but the sum does. Default 0 disables (preserves prior
    # behavior). Live config: 10. Coupled with BundleEvaluator's
    # `joint_bonus` — the seeded joints score above empty BECAUSE of
    # the bonus, so they survive beam pruning and can be extended in
    # later iterations.
    joint_seeds: int = 0

    def search(self, world: "World",
               *, my_id: Optional[int] = None,
               seed_bundle: Optional[Bundle] = None,
               opp_overlays: Optional[Mapping[int, Bundle]] = None,
               deadline: Optional[float] = None,
               ) -> Bundle:
        """Return the highest-scoring Bundle for this turn.

        Caller is the agent's per-turn loop. `Bundle.specs_at_turn(0)`
        on the result gives the launches to emit as this turn's
        action; future-turn specs persist in the agent's memory
        across turns (Phase 7d).

        `seed_bundle` (Phase 7e): last turn's plan, typically passed
        in already time-shifted via `Bundle.shift_forward(1)`. The
        frontier is seeded with BOTH the empty bundle AND the seed,
        then per-iteration neighbors include both adds AND drops.
        Drops let the search retract a seed launch that no longer
        fits; the empty-bundle floor guarantees the search never
        returns something worse than no-op. `None` (default) restores
        Phase 7c/d behaviour exactly.

        `opp_overlays` (Phase 8): per-opponent predicted bundles
        passed straight through to BundleEvaluator. Every score this
        search computes applies these overlays after our candidate
        bundle, so the chooser ranks our bundles against realistic
        counterplay instead of a passive world. `None` (default)
        scores against the passive world (Phase 7c-7e behaviour).
        Build the dict with `predict_opp_bundles_via_mirror_search`.

        `deadline` (Phase 8): absolute `time.perf_counter()` wallclock
        bound. If set, the search bails out the moment elapsed time
        crosses it and returns the best bundle seen so far. The
        empty-bundle floor is always preserved (scored before any
        bailable work). `None` (default) runs to depth completion.
        Required for live-env compliance with the 1000ms actTimeout.
        """
        if my_id is None:
            my_id = world.my_id

        # `time.perf_counter` is module-imported below; do this here
        # so the deadline check is a fast attribute lookup.
        if deadline is not None:
            from time import perf_counter as _now
        else:
            _now = None  # type: ignore[assignment]

        empty = Bundle()
        empty_score = self.evaluator.score(
            world, empty, my_id=my_id, opp_overlays=opp_overlays,
        ).total
        best_bundle = empty
        best_score = empty_score

        # Beam frontier: list of (score, bundle). Start with just empty.
        frontier: list[tuple[float, Bundle]] = [(empty_score, empty)]
        # Dedup across iterations: each launches-tuple scored at most
        # once. Bundles share equality via their frozen LaunchSpec
        # tuple. Initialised with empty so drops-to-empty are skipped
        # (re-scoring empty as an "extension" would burn a beam slot).
        seen: set[tuple[LaunchSpec, ...]] = {empty.launches}

        if seed_bundle is not None and not seed_bundle.is_empty:
            try:
                seed_score = self.evaluator.score(
                    world, seed_bundle, my_id=my_id,
                    opp_overlays=opp_overlays,
                ).total
            except ValueError:
                # Seed is infeasible against current world (e.g. a
                # source got captured between turns). Treat as if no
                # seed was passed.
                seed_score = None
            if seed_score is not None:
                frontier.append((seed_score, seed_bundle))
                seen.add(seed_bundle.launches)
                if seed_score > best_score:
                    best_score = seed_score
                    best_bundle = seed_bundle

        # Phase E Phase 1a (2026-05-18): seed the frontier with explicit
        # joint-pair candidates. The chicken-and-egg problem with depth=2
        # discovery is: at iteration 1, a single bouncing launch_a scores
        # WORSE than empty (ship_delta -= committed ships, planet credit
        # = 0); beam_width=3 prunes it before iteration 2 can extend it
        # to launch_a + launch_b. Pre-seeding 2-launch joint candidates
        # bypasses this — the joint already carries its bonus from
        # BundleEvaluator's joint_bonus path, so it survives the beam
        # pruning naturally. Default `joint_seeds=0` skips this work
        # (preserves prior behavior when BUNDLE_JOINT_BONUS unset).
        if self.joint_seeds > 0:
            try:
                joint_pairs = self._enumerate_joint_seeds(
                    world, my_id, max_seeds=self.joint_seeds,
                )
            except Exception:
                joint_pairs = []
            for joint_bundle in joint_pairs:
                if _now is not None and _now() >= deadline:
                    break
                if joint_bundle.launches in seen:
                    continue
                seen.add(joint_bundle.launches)
                try:
                    s = self.evaluator.score(
                        world, joint_bundle, my_id=my_id,
                        opp_overlays=opp_overlays,
                    ).total
                except ValueError:
                    continue
                frontier.append((s, joint_bundle))
                if s > best_score:
                    best_score = s
                    best_bundle = joint_bundle

        # Outer label so deadline-driven `break`s exit the depth loop.
        timed_out = False
        for _ in range(self.max_depth):
            if _now is not None and _now() >= deadline:
                timed_out = True
                break
            extensions: list[tuple[float, Bundle]] = []
            for _, bundle in frontier:
                if _now is not None and _now() >= deadline:
                    timed_out = True
                    break
                current = bundle.apply(world)
                sun = SunFilter(current, safety_margin=self.sun_safety)

                # ADD neighbours: bundle + new candidate.
                for spec in self._enumerate_candidates(current, my_id, sun):
                    if _now is not None and _now() >= deadline:
                        timed_out = True
                        break
                    new_launches = bundle.launches + (spec,)
                    if new_launches in seen:
                        continue
                    seen.add(new_launches)
                    try:
                        extended = Bundle(new_launches)
                        s = self.evaluator.score(
                            world, extended, my_id=my_id,
                            opp_overlays=opp_overlays,
                        ).total
                    except ValueError:
                        continue
                    extensions.append((s, extended))
                    if s > best_score:
                        best_score = s
                        best_bundle = extended

                if timed_out:
                    break

                # DROP neighbours: bundle - one existing launch.
                # No-op when `bundle` is empty (range(0)). For
                # singletons, the drop lands on empty which is in
                # `seen` — skipped without scoring. For larger
                # bundles the drop yields a genuinely new partial
                # plan. Drops are how the search retracts a seed
                # launch that no longer fits + how swap emerges
                # (drop-then-add over two iterations).
                for i in range(len(bundle.launches)):
                    if _now is not None and _now() >= deadline:
                        timed_out = True
                        break
                    new_launches = (bundle.launches[:i]
                                    + bundle.launches[i + 1:])
                    if new_launches in seen:
                        continue
                    seen.add(new_launches)
                    try:
                        dropped = Bundle(new_launches)
                        s = self.evaluator.score(
                            world, dropped, my_id=my_id,
                            opp_overlays=opp_overlays,
                        ).total
                    except ValueError:
                        continue
                    extensions.append((s, dropped))
                    if s > best_score:
                        best_score = s
                        best_bundle = dropped

                if timed_out:
                    break

            if timed_out or not extensions:
                break
            extensions.sort(key=lambda x: x[0], reverse=True)
            frontier = extensions[:self.beam_width]

        return best_bundle

    def _enumerate_candidates(self, world: "World", my_id: int,
                                sun: "SunFilter",
                                ):
        """Yield single-LaunchSpec candidates across the
        (source × top-N target × launch_turn × ship_ratio)
        cross-product. Filtered by per-launch-turn source availability
        + SunFilter.

        The target pool is UNIFORM — every non-comet planet (own,
        enemy, neutral). Defense (own→own to hold a threatened
        planet), staging (own→own to forward-position ships), attack
        (own→enemy / own→neutral), and tempo (large gang-up bundles)
        all emerge from the scorer's ΔP(win) ranking via
        BundleEvaluator.score. Carving the target pool by
        ownership / threat heuristics here biases the chooser away
        from emergent strategy — that's the lesson from the 2026-05-18
        defense-hotfix revert (the framework doc at
        `knowledge-base/concepts/probability-of-winning-framework.md`
        is the authoritative reference).

        Per (src, tgt, launch_turn), the enumerator emits one
        LaunchSpec per `ship_ratios` entry — small / medium / full
        commit. The scorer reads `ownership_at(target, horizon)` for
        each variant and naturally prefers the cheapest commit that
        still captures (or holds, or stages successfully).

        For `launch_turn > 0`, ship availability is queried via
        `world.ownership_at(src.id, launch_turn)` so production
        accrual + prior bundle commitments are correctly accounted
        for. For sources currently owned by us (the only ones we
        enumerate), `ownership_at` returns the future ship count.
        """
        sources = [p for p in world.planets
                   if p.owner == my_id
                   and not p.is_comet
                   and p.ships >= self.min_source_ships]
        all_targets = [p for p in world.planets if not p.is_comet]
        if not sources or not all_targets:
            return

        for src in sources:
            # Top-N closest targets — pure budget throttle across the
            # uniform pool. The scorer decides which target deserves
            # the commitment; we just gate enumeration on distance.
            # FILTER SELF BEFORE the top-N slice — otherwise the
            # source's own entry (distance 0) eats one of the N slots
            # and we enumerate N-1 actual targets. In early game with
            # candidates_per_source=2, that meant 1 target per source
            # and bundle effectively idled.
            targets = [t for t in all_targets if t.id != src.id]
            if not targets:
                continue
            scored = sorted(
                targets,
                key=lambda t: math.hypot(
                    src.current_x - t.current_x,
                    src.current_y - t.current_y,
                ),
            )[:self.candidates_per_source]

            for tgt in scored:
                # Aim at target's current position. Good enough for
                # short-range static targets; orbital lead-aim
                # refinement (lib.aim.search_safe_intercept) is a
                # follow-up — naive aim keeps moving parts minimal
                # for the enumeration-shape A/B.
                dx = tgt.current_x - src.current_x
                dy = tgt.current_y - src.current_y
                if dx == 0 and dy == 0:
                    continue
                angle = math.atan2(dy, dx)

                for launch_turn in self.launch_turns:
                    if launch_turn < 0:
                        continue
                    # Source availability at launch_turn.
                    if launch_turn == 0:
                        avail = int(src.ships)
                    else:
                        owner_at, ships_at = world.ownership_at(
                            src.id, launch_turn,
                        )
                        if owner_at != my_id:
                            # Source captured before launch_turn.
                            continue
                        avail = int(ships_at)
                    # Reserve N ships at the source post-launch; the
                    # rest is the launch budget. `reserve_ships_at_source`
                    # is decoupled from `min_source_ships` (Piece 7
                    # root-cause #1).
                    usable = avail - self.reserve_ships_at_source
                    if usable < 1:
                        continue
                    # Dedup ship counts within the per-(src,tgt,turn)
                    # group (small `usable` values can collapse two
                    # ratios to the same integer). The set keeps the
                    # enumeration tidy and avoids re-scoring identical
                    # specs downstream.
                    emitted_counts: set[int] = set()
                    # Target-aware capture-min variant: `tgt.ships + 1`
                    # is enough to (a) capture an enemy/neutral target
                    # outright (combat: attackers > defenders → owner
                    # flips), or (b) probe a reinforcement of our own
                    # planet at the symmetric size. Uniform across
                    # ownership types — the scorer picks via ΔP(win),
                    # not the enumerator. Piece 7 root-cause #2.
                    capture_min = int(tgt.ships) + 1
                    candidate_counts = [
                        max(1, min(usable, int(round(usable * ratio))))
                        for ratio in self.ship_ratios
                    ]
                    if 1 <= capture_min <= usable:
                        candidate_counts.append(capture_min)
                    for ships in candidate_counts:
                        if ships in emitted_counts:
                            continue
                        emitted_counts.add(ships)
                        spec = LaunchSpec(
                            src_id=src.id, aim_angle=angle,
                            ships=ships, owner=my_id,
                            launch_turn=int(launch_turn),
                        )
                        if not sun.is_safe(spec):
                            continue
                        yield spec

    def _enumerate_joint_seeds(
        self, world: "World", my_id: int, *, max_seeds: int = 10,
    ) -> list["Bundle"]:
        """Generate up to `max_seeds` 2-launch Bundles for joint captures
        that no single source could make alone. Phase E Phase 1a.

        Selection: for each enemy/neutral target T, consider the closest
        pair of our sources. Emit a 2-launch Bundle iff:
          - neither solo would capture (each source's available ships
            < defenders+1 at T)
          - the pair WOULD capture (combined avail > defenders)
        Bounded enumeration: top-5 targets sorted by defender count
        (high-defender targets are the ones that need coordination);
        per target, try the 3 closest source-pairs.

        All launches use `launch_turn=0` (Phase 1 scope: same-turn
        joints). Cross-turn coordination (launch_turn>0 paired with
        launch_turn=0) is deferred — the regular ADD/DROP iterations
        can still discover it from a same-turn-paired seed.
        """
        sources = [p for p in world.planets
                   if p.owner == my_id and not p.is_comet
                   and p.ships >= self.min_source_ships]
        if len(sources) < 2:
            return []
        targets = [p for p in world.planets
                   if not p.is_comet
                   and p.owner != my_id
                   and p.owner != -1]
        if not targets:
            # Fall back to neutral targets if no enemy planets exist
            # (e.g. opening turn before any enemy expansion).
            targets = [p for p in world.planets
                       if not p.is_comet and p.owner == -1]
        if not targets:
            return []
        # Highest-defender targets first — they're the ones that need
        # coordination. Capped at top-5 to bound enumeration cost.
        targets = sorted(targets, key=lambda t: -int(t.ships))[:5]

        seeds: list[Bundle] = []
        for tgt in targets:
            if len(seeds) >= max_seeds:
                break
            defenders = int(tgt.ships)
            if defenders < 1:
                continue
            srcs_by_dist = sorted(
                sources,
                key=lambda s: math.hypot(
                    s.current_x - tgt.current_x,
                    s.current_y - tgt.current_y,
                ),
            )
            # Try the 3 closest sources × pair them with the next 2-3.
            tried_pairs: set[tuple[int, int]] = set()
            for i in range(min(3, len(srcs_by_dist))):
                if len(seeds) >= max_seeds:
                    break
                for j in range(i + 1, min(i + 4, len(srcs_by_dist))):
                    a, b = srcs_by_dist[i], srcs_by_dist[j]
                    key = (min(a.id, b.id), max(a.id, b.id))
                    if key in tried_pairs:
                        continue
                    tried_pairs.add(key)
                    avail_a = int(a.ships) - self.reserve_ships_at_source
                    avail_b = int(b.ships) - self.reserve_ships_at_source
                    if avail_a < 1 or avail_b < 1:
                        continue
                    # Joint condition: neither solo captures, pair does.
                    if avail_a > defenders or avail_b > defenders:
                        continue  # solo can win — no joint needed
                    if avail_a + avail_b <= defenders:
                        continue  # even pair can't win — skip
                    # Aim each source at the target. Atan2 is fine for
                    # the seed (lead-aim refinement reverted in Phase E
                    # Phase 0).
                    angle_a = math.atan2(
                        tgt.current_y - a.current_y,
                        tgt.current_x - a.current_x,
                    )
                    angle_b = math.atan2(
                        tgt.current_y - b.current_y,
                        tgt.current_x - b.current_x,
                    )
                    spec_a = LaunchSpec(
                        src_id=a.id, aim_angle=angle_a,
                        ships=avail_a, owner=my_id, launch_turn=0,
                    )
                    spec_b = LaunchSpec(
                        src_id=b.id, aim_angle=angle_b,
                        ships=avail_b, owner=my_id, launch_turn=0,
                    )
                    # SunFilter check — skip joints whose either leg
                    # would crash the sun.
                    sun = SunFilter(world, safety_margin=self.sun_safety)
                    if not sun.is_safe(spec_a) or not sun.is_safe(spec_b):
                        continue
                    seeds.append(Bundle(launches=(spec_a, spec_b)))
                    if len(seeds) >= max_seeds:
                        break
        return seeds


# ---------------------------------------------------------------------------
# PHASE 8 — Mirror-search opp model
# ---------------------------------------------------------------------------


def predict_opp_bundles_via_mirror_search(
    world: "World",
    *,
    my_id: Optional[int] = None,
    search: Optional[BundleSearch] = None,
    depth: int = 1,
) -> dict[int, Bundle]:
    """For each opponent present in `world`, run BundleSearch from
    their seat and return their predicted best bundle.

    Plugged into a BundleSearch call as `opp_overlays=...` so the
    chooser scores our candidate bundles against realistic
    counterplay instead of a passive world.

    `depth` controls how deep the mirror recursion goes:
      - 0  → returns {} (opponents stay passive).
      - 1  → each opp runs BundleSearch with opp_overlays={} (no inner
             mirror). Default. Adds first-order counterplay; cost is
             ~N_opps extra BundleSearch calls per turn.
      - >1 → each opp's inner search recurses with depth-1. Cost
             scales exponentially in depth; budget carefully.

    Opp ids come from `world.planets`: any non-comet planet with
    owner != my_id and owner != -1. Opps with no surviving sources
    yield an empty bundle (BundleEvaluator skips empties at apply
    time, so they cost nothing downstream).

    `search` defaults to a vanilla BundleSearch(); pass a cheaper one
    (smaller max_depth / beam_width) to bound mirror-search cost
    relative to our own chooser pass.
    """
    if my_id is None:
        my_id = world.my_id
    if depth <= 0:
        return {}
    if search is None:
        search = BundleSearch()

    opp_ids: set[int] = set()
    for p in world.planets:
        if p.is_comet:
            continue
        if p.owner == -1 or p.owner == my_id:
            continue
        opp_ids.add(p.owner)
    if not opp_ids:
        return {}

    out: dict[int, Bundle] = {}
    for opp_id in sorted(opp_ids):
        if depth == 1:
            inner_overlays: dict[int, Bundle] = {}
        else:
            inner_overlays = predict_opp_bundles_via_mirror_search(
                world,
                my_id=opp_id,
                search=search,
                depth=depth - 1,
            )
        bundle = search.search(
            world,
            my_id=opp_id,
            opp_overlays=inner_overlays,
        )
        if not bundle.is_empty:
            out[opp_id] = bundle
    return out


# ---------------------------------------------------------------------------
# PHASE 8b — Event-driven trajectory-native reactive opp model
# ---------------------------------------------------------------------------


def world_to_obs(world: "World", player_id: int) -> dict:
    """Adapter: World → kaggle-style obs dict, from `player_id`'s seat.

    `lite_greedy_policy` consumes a dict-or-namespace with `.player`
    and `.planets` (the latter as `(id, owner, x, y, radius, ships,
    production)` tuples). This adapter materialises that shape from
    a `World` so the reactive opp loop can call lite_greedy on any
    snapshot (turn-0 OR `snapshot_at(t)` output) without re-routing
    through `World.from_obs`.

    `fleets` is included for downstream compatibility but not used
    by `lite_greedy_policy` (verified at `lib/opp_model.py:155-233`).
    """
    return {
        "player": int(player_id),
        "step": int(world.step),
        "planets": [
            (p.id, p.owner, float(p.current_x), float(p.current_y),
             float(p.radius), float(p.ships), float(p.production))
            for p in world.planets
        ],
        "fleets": [
            (f.id, f.owner, float(f.current_x), float(f.current_y),
             float(f.angle), f.from_planet_id, int(f.ships))
            for f in world.fleets if f.spawn_turn == 0
        ],
    }


def predict_opp_via_event_driven_lite_greedy(
    world: "World",
    *,
    my_id: Optional[int] = None,
    horizon: int = DEFAULT_LEDGER_HORIZON,
    max_events: int = 30,
) -> dict[int, Bundle]:
    """Event-driven trajectory-native reactive opp model.

    For each opponent, predict their full plan of launches across the
    rollout horizon by walking the trajectory layer's natural event
    stream — arrival ETAs from the ledger — and calling
    `lite_greedy_policy` at each event timestamp. The result is
    structurally equivalent to what `agents/baseline`'s `_build_opp_
    trajectory` produces via per-step fast_sim, but built WITHOUT
    stepping a simulator: each "step" is a closed-form
    `snapshot_at(t)` reconstruction (~2 ms) followed by a stateless
    lite_greedy call (~1 ms).

    Why event-driven instead of fixed-stride:
    - Fleet arrivals are the ONLY moments state transitions occur
      (production accrual is continuous, fleet motion deterministic,
      ownership only flips at arrivals).
    - The trajectory layer pre-computes every arrival's ETA in the
      ledger — i.e. the event set is FREE.
    - Fixed-stride (e.g. every 5 turns) can miss a cluster of
      transitions between snapshots; event-driven catches every one
      at exactly the right moment.
    - When opp launches at event t, their new fleet's arrival is a
      new event we must process. The event queue grows dynamically.

    Returns dict[opp_id, Bundle], structured for direct use as
    `opp_overlays=...` in `BundleSearch.search` or
    `BundleEvaluator.score`.

    `max_events` caps total event-processing iterations to prevent
    runaway in pathological cases (deep recursive launch chains).
    Default 30 is generous for a 30-turn horizon — most rollouts
    have <15 events.
    """
    if my_id is None:
        my_id = world.my_id
    h = max(0, int(horizon))
    if h <= 0:
        return {}

    opp_ids: list[int] = sorted({
        p.owner for p in world.planets
        if p.owner != -1 and p.owner != my_id and not p.is_comet
    })
    if not opp_ids:
        return {}

    # Initial event set: every arrival ETA in the parent's ledger.
    # The trajectory layer caches this on first query; subsequent
    # calls are free.
    initial_etas: set[int] = set()
    for arrivals in world.ledger_all(h).values():
        for a in arrivals:
            if 0 < a.eta <= h:
                initial_etas.add(int(a.eta))

    event_queue: list[int] = sorted({0} | initial_etas)
    processed: set[int] = set()

    opp_specs_acc: dict[int, list[LaunchSpec]] = {oid: [] for oid in opp_ids}
    overlay = world
    iterations = 0

    while event_queue and iterations < max_events:
        t = event_queue.pop(0)
        if t in processed or t > h:
            continue
        processed.add(t)
        iterations += 1

        # Snapshot to relative turn t. t=0 is a no-op (returns self).
        snap = overlay.snapshot_at(t)

        any_new_launch = False
        for opp_id in opp_ids:
            obs = world_to_obs(snap, opp_id)
            actions = lite_greedy_policy(obs)
            for action in actions:
                src_id, angle, ships = action
                spec = LaunchSpec(
                    src_id=int(src_id),
                    aim_angle=float(angle),
                    ships=int(ships),
                    owner=int(opp_id),
                    launch_turn=int(t),
                )
                # Apply to overlay so future snapshots reflect this
                # launch's source deduction + new fleet arrival. If
                # the launch is infeasible vs the evolving overlay
                # (rare: e.g. source captured by us before t), skip
                # silently — matches BundleEvaluator's drop semantics.
                try:
                    overlay = overlay.with_candidate(spec)
                except ValueError:
                    continue
                opp_specs_acc[opp_id].append(spec)
                any_new_launch = True

        # After applying this event's launches, the overlay's ledger
        # has new arrivals. Add their ETAs as new events. The cache
        # rebuild is O(N_new_fleets) thanks to with_candidate's
        # inherited ledger (Phase 8 perf fix).
        if any_new_launch:
            new_ledger = overlay.ledger_all(h)
            for arrivals in new_ledger.values():
                for a in arrivals:
                    if (a.eta > t and a.eta <= h
                            and a.eta not in processed
                            and a.eta not in event_queue):
                        event_queue.append(int(a.eta))
            event_queue.sort()

    return {
        oid: Bundle(launches=tuple(specs))
        for oid, specs in opp_specs_acc.items()
        if specs
    }


def predict_my_followup_via_event_driven_lite_greedy(
    world: "World",
    *,
    my_id: Optional[int] = None,
    horizon: int = DEFAULT_LEDGER_HORIZON,
    max_events: int = 10,
) -> Bundle:
    """Me-side mirror of `predict_opp_via_event_driven_lite_greedy`.

    Caller passes a `world` that already has my candidate bundle
    applied (i.e. `overlay = my_bundle.apply(base_world)`). This
    function predicts what lite_greedy from MY seat would launch at
    each subsequent arrival event, accumulates those launches into a
    single Bundle, and returns it. The caller composes:

        followup = predict_my_followup_via_event_driven_lite_greedy(overlay)
        overlay = followup.apply(overlay)

    Differs from the opp version:
    - Iterates only [my_id], not all opps. Returns one Bundle, not a
      dict — there is only ever one "me" per call.
    - `max_events=10` (vs opp's 30). Me-followup is invoked per-score-
      call inside `BundleEvaluator.score` (~15-20× per turn at default
      knobs), so its cost compounds where opp's runs once per turn.
      Empirically 1-3 launches show up in a 30-turn rollout, so 10 is
      generous.
    - KEEPS t=0 in the event queue (matches opp). At t=0, the
      overlay's snapshot already reflects my_bundle's ship deductions
      for `launch_turn=0` launches; `lite_greedy_policy` naturally
      filters drained sources via its `src[5] < 10` ship check. The
      value-add at t=0 is launches from sources my_bundle didn't touch
      — those are legitimate reactive launches and capturing them is
      the whole point of making the rollout self-consistent.

    Bug #14 (me-half) target: without this function, `BundleEvaluator.
    score` treats sources as drained-forever after my_bundle's launch,
    so it pessimistically rejects profitable launches even when
    production would refill the source within horizon. With this
    function applied to the overlay before scoring, the path-integrated
    planet/production credit accurately reflects the source's
    refill-and-relaunch trajectory.
    """
    if my_id is None:
        my_id = world.my_id
    h = max(0, int(horizon))
    if h <= 0:
        return Bundle()

    # Need at least one owned planet to launch from; bail early
    # otherwise (cheap path for the eliminated-me / no-my-planets case).
    has_my_planet = any(
        p.owner == my_id and not p.is_comet for p in world.planets
    )
    if not has_my_planet:
        return Bundle()

    initial_etas: set[int] = set()
    for arrivals in world.ledger_all(h).values():
        for a in arrivals:
            if 0 < a.eta <= h:
                initial_etas.add(int(a.eta))

    event_queue: list[int] = sorted({0} | initial_etas)
    processed: set[int] = set()

    my_specs_acc: list[LaunchSpec] = []
    overlay = world
    iterations = 0

    while event_queue and iterations < max_events:
        t = event_queue.pop(0)
        if t in processed or t > h:
            continue
        processed.add(t)
        iterations += 1

        snap = overlay.snapshot_at(t)

        obs = world_to_obs(snap, my_id)
        actions = lite_greedy_policy(obs)
        any_new_launch = False
        for action in actions:
            src_id, angle, ships = action
            spec = LaunchSpec(
                src_id=int(src_id),
                aim_angle=float(angle),
                ships=int(ships),
                owner=int(my_id),
                launch_turn=int(t),
            )
            try:
                overlay = overlay.with_candidate(spec)
            except ValueError:
                continue
            my_specs_acc.append(spec)
            any_new_launch = True

        if any_new_launch:
            new_ledger = overlay.ledger_all(h)
            for arrivals in new_ledger.values():
                for a in arrivals:
                    if (a.eta > t and a.eta <= h
                            and a.eta not in processed
                            and a.eta not in event_queue):
                        event_queue.append(int(a.eta))
            event_queue.sort()

    return Bundle(launches=tuple(my_specs_acc))


