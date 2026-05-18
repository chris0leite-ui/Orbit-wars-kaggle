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
    """
    # End-state deltas at horizon (us vs sum of others, alive planets only).
    ship_delta: float
    planet_delta: float
    production_delta: float
    # Number of opponents eliminated (started with planets, owns none at K).
    eliminations: int
    # Sum: the scalar the chooser optimises.
    total: float


@dataclass(frozen=True)
class BundleEvaluator:
    """Score a bundle by reading the trajectory layer's K-turn outcome.

    The score is a weighted sum of (ship_delta, planet_delta,
    production_delta, eliminations) at the horizon, measured AFTER
    applying the bundle to the world. The default weights are hand-
    tuned starting points; Step 3's learned value head replaces this
    function with a trained network later.

    `horizon` is the look-ahead in turns. Default 30 — long enough to
    capture cross-board strikes (board-diagonal ETA at speed=1.66 is
    ~85 steps, but most strategic launches are <30 turns), short
    enough that opponent uncertainty doesn't dominate the score.
    """
    horizon: int = 30
    planet_weight: float = 5.0
    production_weight: float = 10.0
    elimination_bonus: float = 200.0

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
        if opp_overlays:
            for opp_id, opp_bundle in opp_overlays.items():
                if opp_bundle.is_empty:
                    continue
                try:
                    overlay = opp_bundle.apply(overlay)
                except ValueError:
                    continue

        ships_by_owner: dict[int, float] = defaultdict(float)
        planets_by_owner: dict[int, int] = defaultdict(int)
        prod_by_owner: dict[int, float] = defaultdict(float)
        for p in overlay.planets:
            if p.is_comet:
                continue
            owner, ships = overlay.ownership_at(p.id, self.horizon)
            if owner != -1:
                ships_by_owner[owner] += ships
                planets_by_owner[owner] += 1
                prod_by_owner[owner] += p.production

        my_ships = ships_by_owner.get(my_id, 0.0)
        my_planets = planets_by_owner.get(my_id, 0)
        my_prod = prod_by_owner.get(my_id, 0.0)
        other_ships = sum(v for k, v in ships_by_owner.items() if k != my_id)
        other_planets = sum(v for k, v in planets_by_owner.items() if k != my_id)
        other_prod = sum(v for k, v in prod_by_owner.items() if k != my_id)

        ship_delta = my_ships - other_ships
        planet_delta = float(my_planets - other_planets)
        production_delta = my_prod - other_prod

        # Count opponents who started with planets but have none at K.
        initial_opp_owners: set[int] = set()
        for p in world.planets:
            if p.is_comet:
                continue
            if p.owner != -1 and p.owner != my_id:
                initial_opp_owners.add(p.owner)
        eliminations = sum(1 for o in initial_opp_owners
                           if planets_by_owner.get(o, 0) == 0)

        total = (ship_delta
                 + self.planet_weight * planet_delta
                 + self.production_weight * production_delta
                 + self.elimination_bonus * eliminations)

        return BundleScore(
            ship_delta=ship_delta,
            planet_delta=planet_delta,
            production_delta=production_delta,
            eliminations=eliminations,
            total=total,
        )


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
    - `candidates_per_source`: top-N closest targets per source.
    - `launch_turns`: delayed-launch enumeration. Each candidate
      (source, target) pair generates one LaunchSpec per launch_turn
      in this tuple. Default `(0,)` preserves Phase 7c behaviour;
      `(0, 5, 10, 15)` enables coordinated-arrival planning.
    - `min_source_ships`: skip sources too weak to launch from.
    - `ship_count_multiplier`: ships to commit per launch as a
      multiple of (target.ships + 1). 1.0 = "just enough"; >1 = "buffer."
    """
    evaluator: BundleEvaluator = field(default_factory=BundleEvaluator)
    max_depth: int = 3
    beam_width: int = 4
    candidates_per_source: int = 3
    launch_turns: tuple[int, ...] = (0,)
    min_source_ships: int = 2
    ship_count_multiplier: float = 1.0
    sun_safety: float = SUN_SAFETY

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
        (source × top-N target × launch_turn) cross-product. Filtered
        by per-launch-turn source availability + SunFilter.

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
        targets = [p for p in world.planets
                   if p.owner != my_id and not p.is_comet]
        if not sources or not targets:
            return

        for src in sources:
            # Top-N closest targets — heuristic that the chooser
            # cares most about reachable targets.
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
                # refinement comes later. For a future-launch
                # candidate, ideally we'd aim at target's predicted
                # position at (launch_turn + eta); for static targets
                # that's the same as current position.
                dx = tgt.current_x - src.current_x
                dy = tgt.current_y - src.current_y
                if dx == 0 and dy == 0:
                    continue
                angle = math.atan2(dy, dx)

                # Per-target required ship count (constant across
                # launch_turns — for enemy targets, production over a
                # longer flight makes this an underestimate; the
                # multiplier knob compensates).
                required = max(1, int(tgt.ships) + 1)
                base_commit = int(required * self.ship_count_multiplier)
                if base_commit < 1:
                    continue

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
                    commit = min(base_commit, avail - 1)  # keep 1
                    if commit < 1:
                        continue
                    spec = LaunchSpec(
                        src_id=src.id, aim_angle=angle,
                        ships=commit, owner=my_id,
                        launch_turn=int(launch_turn),
                    )
                    if not sun.is_safe(spec):
                        continue
                    yield spec


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

