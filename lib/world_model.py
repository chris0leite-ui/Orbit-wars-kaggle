"""WorldModel — arrival ledger + per-planet timeline simulator.

The v2 substrate, adapted from Roman 1224 / Pilkwang structured-baseline
patterns (audit/2026-05-10-public-kernel-teardown.md), re-implemented
against our `lib/` interfaces.

Use cases:
- `arrival_ledger` mechanism (v2): drop intents whose target will already
  be ours at our fleet's arrival step (don't double-commit).
- Mission scoring (v3): mission builders read predicted owner-at-arrival
  + predicted garrison-at-arrival to choose target-side mission class.
- Intercept missions (v3): defend planets the timeline says will fall
  to an enemy fleet.

Performance:
Per-planet timeline simulation is O(horizon) per planet. Building the
ledger from in-flight fleets is O(fleets * planets) for the ray-cast
target attribution. Whole-snapshot construction is ~O(planets * horizon
+ fleets * planets); on a typical board (40 planets, <50 in-flight
fleets, horizon=110) this is well under 5 ms.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from lib.combat import resolve_arrivals
from lib.fleet import speed as fleet_speed
from lib.orbit import is_orbiting, predict_relative, predict_relative_smart

# Raised 110 → 250 (2026-05-11): reinforce class was firing 0.2
# candidates/turn because long-runway threats were invisible past
# step 110 + eta. Matches the EPISODE_STEPS/2 framing in the score
# formula (`time_to_hold = 500 - step - eta`); timeline-build cost
# scales linearly so per-turn p95 should remain well under the 1s
# actTimeout. See audit/2026-05-11-v3-snipe-critical-review.md §P2.
DEFAULT_HORIZON = 250

# Bug #12 fix (2026-05-18): width of the in-flight-enemy summation
# window used when computing combined threat against a single planet.
# A staggered multi-wave attack (e.g. f1 at eta=2 + f2 at eta=4)
# should be accounted for as one coordinated threat; pre-fix the
# window was `enemy_eta + 1` of the EARLIEST inbound, which silently
# excluded later waves and zeroed the shortfall. Anchored on the
# asdf-game (76947663) step 37 trace. Promoted to lib so both the
# proposer (`agents/baseline/proposer.py`) and the in-rollout
# defensive policy (`lib/opp_model.me_defensive_action`) import it
# from one location. The principled v2 of this fix is a full
# timeline simulation to find the max shortfall over time; this
# constant is the cheap version.
WAVE_LOOKAHEAD = 12


def _position_at(planet, omega: float, lead_turns: int) -> tuple[float, float]:
    """Return predicted `(x, y)` of `planet` `lead_turns` from now.

    Current position for static planets, when `omega == 0`, or when
    `lead_turns <= 0`. Otherwise routes through
    `lib.orbit.predict_relative` after the `is_orbiting` gate.

    Used by `time_to_enemy_threat` and sibling orbital-safety call sites
    to keep the predict-position-at-arrival pattern in one tested place.
    """
    if lead_turns <= 0 or omega == 0.0:
        return float(planet.x), float(planet.y)
    tup = [planet.id, planet.owner, planet.x, planet.y,
           planet.radius, planet.ships, planet.production]
    if not is_orbiting(tup):
        return float(planet.x), float(planet.y)
    return predict_relative_smart(tup, omega, lead_turns)


def fleet_target_planet(fleet, planets, omega: float = 0.0,
                        max_horizon: int = DEFAULT_HORIZON,
                        comet_paths: dict | None = None):
    """Trace `fleet` along its angle, find first planet it'd hit.

    Returns `(target_planet, eta_turns)` or `(None, None)` if no planet
    intersects the fleet's trajectory within `max_horizon` steps.

    For STATIC (non-orbiting) planets: straight-line ray-cast (cheap;
    closed-form). For ORBITING planets: per-tick collision check using
    `lib.orbit.predict_relative` to predict the planet's position at
    each tick, then test fleet-vs-planet point-in-circle. For COMETS
    (id in `comet_paths`): per-tick check using the comet's pre-
    computed path (env semantics — comets do NOT rotate around the
    sun, they follow polynomial paths).

    `comet_paths` is `{pid: (path, path_index)}` — typically produced
    by `_comet_paths_by_id(world)`. When None / empty, comets fall
    through to the orbital ray-cast path with rotation math, which is
    physically wrong but matches pre-2026-05-23 behaviour.

    The `omega` argument is the environment's angular velocity from the
    obs. When `omega == 0.0`, behaviour matches the previous static-only
    ray-cast (orbiting check is short-circuited since rotation is zero).

    Used to build the arrival ledger from in-flight fleets — the env
    doesn't expose a fleet's intended target, only its angle.

    Bug fix 2026-05-18 (#11): pre-fix the static ray-cast missed
    orbiting targets that rotate INTO the fleet's path mid-flight.
    Asdf game (76947663) step 37: 65-ship fleet aimed at orbiting P15
    returned target=None until P15 had rotated into the straight line
    at step 40 — by then too late to defend.

    Bug fix 2026-05-23: pre-fix, comets passed `is_orbiting` (they sit
    inside ROTATION_RADIUS_LIMIT) and got rotated as if orbital, while
    the env advances them along their path. KT-ON path used
    `lookup_relative` (path-indexed, correct); KT-OFF path used
    `predict_relative` (rotation, wrong). The two paths diverged by
    several board units per comet per tick; cascaded into wrong fleet
    attribution and divergent opponent-model decisions in
    `me_defensive_action`. See /tmp/trace_turn_273_p1.py: planet 24 at
    turn 273 returned (29.31, 65.20) on KT-OFF vs (29.63, 69.89) on
    KT-ON. Fix: partition comets to a separate per-tick scan using
    `path[idx + t]` directly.
    """
    dir_x = math.cos(fleet.angle)
    dir_y = math.sin(fleet.angle)
    spd = fleet_speed(fleet.ships)
    if spd <= 0:
        return None, None

    # Partition planets: comets (path-indexed) → static (closed-form
    # fast path) → orbiting (per-tick scan). Comets MUST be checked
    # first because they sit inside ROTATION_RADIUS_LIMIT and would
    # otherwise route to the orbital branch with rotation math.
    static_planets = []
    orbiting_planets = []
    comet_planets = []  # list of (planet, path, path_index)
    cps = comet_paths or {}
    for p in planets:
        pid = int(p.id)
        if pid in cps:
            path, path_idx = cps[pid]
            comet_planets.append((p, path, int(path_idx)))
            continue
        # Build minimal tuple for is_orbiting (only x, y, radius used)
        p_tuple = (pid, int(p.owner), float(p.x), float(p.y),
                   float(p.radius), 0, 0)
        if omega != 0.0 and is_orbiting(p_tuple):
            orbiting_planets.append((p, p_tuple))
        else:
            static_planets.append(p)

    best_planet = None
    best_turns = None

    # Fast path: static planets — straight-line ray-cast (unchanged
    # math from pre-fix behaviour).
    for p in static_planets:
        dx = p.x - fleet.x
        dy = p.y - fleet.y
        proj = dx * dir_x + dy * dir_y
        if proj < 0:
            continue
        perp_sq = dx * dx + dy * dy - proj * proj
        r_sq = p.radius * p.radius
        if perp_sq >= r_sq:
            continue
        hit_d = max(0.0, proj - math.sqrt(max(0.0, r_sq - perp_sq)))
        turns = hit_d / spd
        if turns <= max_horizon and (best_turns is None or turns < best_turns):
            best_turns = turns
            best_planet = p

    # Orbital + comet path: per-tick collision scan.
    if orbiting_planets or comet_planets:
        for t in range(1, int(max_horizon) + 1):
            if best_turns is not None and t > best_turns:
                break
            fx = fleet.x + dir_x * spd * t
            fy = fleet.y + dir_y * spd * t
            for p, p_tuple in orbiting_planets:
                px, py = predict_relative_smart(p_tuple, omega, t)
                if math.hypot(fx - px, fy - py) <= float(p.radius):
                    if best_turns is None or t < best_turns:
                        best_turns = t
                        best_planet = p
                    break
            for p, path, path_idx in comet_planets:
                pt_idx = path_idx + t
                if pt_idx >= len(path):
                    continue  # comet exits before tick t
                pt = path[pt_idx]
                if math.hypot(fx - float(pt[0]), fy - float(pt[1])) <= float(p.radius):
                    if best_turns is None or t < best_turns:
                        best_turns = t
                        best_planet = p
                    break

    if best_planet is None:
        return None, None
    return best_planet, int(math.ceil(best_turns))


def build_arrival_ledger(fleets, planets, omega: float = 0.0,
                         horizon: int = DEFAULT_HORIZON,
                         comet_paths: dict | None = None):
    """{planet_id: [(eta, owner, ships), ...]} for in-flight fleets.

    Fleets that won't hit any planet within `horizon` are dropped (they
    will exit the board or die in sun/non-target collision — out of
    scope for the timeline).

    `omega` is the env's angular velocity; passed through to
    `fleet_target_planet` for correct orbiting-target attribution.
    Defaults to 0 for backward compatibility (callers that don't pass
    it get the previous static-only behaviour).

    `comet_paths` is `{pid: (path, path_index)}` from
    `_comet_paths_by_id(world)`. When provided, comet targets are
    attributed via path-indexed positions instead of rotated orbital
    math. See `fleet_target_planet` docstring for the bug fix details.
    """
    ledger: dict[int, list[tuple[int, int, int]]] = {p.id: [] for p in planets}
    for fleet in fleets:
        target, eta = fleet_target_planet(
            fleet, planets, omega, horizon, comet_paths=comet_paths,
        )
        if target is None:
            continue
        ledger[target.id].append((eta, int(fleet.owner), int(fleet.ships)))
    return ledger


def simulate_planet_timeline(planet, arrivals, horizon: int = DEFAULT_HORIZON):
    """Per-planet step-by-step ownership/garrison simulation.

    `arrivals` is a list of `(eta, owner, ships)`. For each step `t` in
    `[1, horizon]`:
    1. If currently owned (not neutral), produce `production` ships.
    2. Resolve same-step arrivals via `resolve_arrivals`.
    3. Record `owner_at[t]`, `ships_at[t]`.

    Returns a dict with `owner_at` (dict[int, int]), `ships_at`
    (dict[int, float]), and `horizon` (int).
    """
    horizon = max(0, int(math.ceil(horizon)))
    by_turn: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for eta, owner, ships in arrivals:
        if ships <= 0:
            continue
        bucket = max(1, int(math.ceil(eta)))
        if bucket > horizon:
            continue
        by_turn[bucket].append((owner, int(ships)))

    owner = planet.owner
    garrison = float(planet.ships)
    owner_at = {0: owner}
    ships_at = {0: max(0.0, garrison)}

    for t in range(1, horizon + 1):
        if owner != -1:
            garrison += planet.production
        group = by_turn.get(t, [])
        if group:
            owner, garrison = resolve_arrivals(owner, garrison, group)
        owner_at[t] = owner
        ships_at[t] = max(0.0, garrison)

    return {"owner_at": owner_at, "ships_at": ships_at, "horizon": horizon}


def state_at_timeline(timeline, arrival_turn):
    """Read `(owner, ships)` from a timeline at a given turn.

    Clamps `arrival_turn` to `[0, timeline['horizon']]`. Reads from the
    `owner_at` / `ships_at` dicts.
    """
    t = min(max(0, int(math.ceil(arrival_turn))), timeline["horizon"])
    return timeline["owner_at"][t], timeline["ships_at"][t]


def predict_garrison_at(planet, eta: int,
                        arrivals: list[tuple[int, int, int]],
                        ) -> tuple[int, float]:
    """Single-tick combat prediction: `(owner, garrison)` at exactly `eta`
    ticks from now. O(eta) walk, O(arrivals) total work.

    Cheaper alternative to `simulate_planet_timeline` when callers only
    need state at one specific tick (e.g. a candidate's arrival). Same
    combat rules (production tick → resolve_arrivals per step), just
    doesn't build the full dict timeline.

    `arrivals` matches the per-planet entry in `build_arrival_ledger`:
    list of `(eta_arrival, owner, ships)`.

    Origin: trajectory-first chooser (2026-05-17). The chooser scores
    each candidate by predicting the arrival outcome at exactly the
    candidate's eta; building a 40-step timeline per planet per call
    was the dominant cost of the K-step rollout we're replacing.
    """
    eta = max(0, int(math.ceil(eta)))
    if eta == 0:
        return planet.owner, max(0.0, float(planet.ships))

    # Bucket arrivals by tick.
    by_turn: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for arrival_eta, arrival_owner, arrival_ships in arrivals:
        if arrival_ships <= 0:
            continue
        bucket = max(1, int(math.ceil(arrival_eta)))
        if bucket > eta:
            continue
        by_turn[bucket].append((arrival_owner, int(arrival_ships)))

    owner = planet.owner
    garrison = float(planet.ships)
    for t in range(1, eta + 1):
        if owner != -1:
            garrison += planet.production
        group = by_turn.get(t, [])
        if group:
            owner, garrison = resolve_arrivals(owner, garrison, group)
    return owner, max(0.0, garrison)


@dataclass
class WorldModel:
    """Per-turn arrival-ledger snapshot. Built once at the top of an
    agent's turn; consumed by the `arrival_ledger` mechanism today and
    by mission scoring tomorrow."""

    ledger: dict
    timelines: dict
    horizon: int = DEFAULT_HORIZON

    @classmethod
    def from_world(cls, world, horizon: int = DEFAULT_HORIZON):
        """Build from `lib.intent.World`'s obs_raw. Reads in-flight fleets
        directly from the raw obs because `World` doesn't materialise them.

        Threads the env's `angular_velocity` through to the ledger build
        so inbound fleets aimed at orbiting planets are correctly
        attributed (bug #11 fix, 2026-05-18).
        """
        raw = world.obs_raw
        if isinstance(raw, dict):
            fleets_raw = raw.get("fleets", [])
            omega = float(raw.get("angular_velocity", 0.0) or 0.0)
        else:
            fleets_raw = getattr(raw, "fleets", [])
            omega = float(getattr(raw, "angular_velocity", 0.0) or 0.0)

        from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet  # local import — keeps lib/ env-free
        fleets = [Fleet(*f) for f in fleets_raw]
        planets = list(world.planets_by_id.values())
        comet_paths = _comet_paths_by_id(world)
        ledger = build_arrival_ledger(
            fleets, planets, omega, horizon, comet_paths=comet_paths,
        )
        timelines = {
            p.id: simulate_planet_timeline(p, ledger[p.id], horizon) for p in planets
        }
        return cls(ledger=ledger, timelines=timelines, horizon=horizon)

    def owner_at(self, planet_id: int, step) -> int | None:
        """Predicted owner of `planet_id` at `step` from now (None if unknown)."""
        tl = self.timelines.get(planet_id)
        if tl is None:
            return None
        return state_at_timeline(tl, step)[0]

    def ships_at(self, planet_id: int, step) -> float | None:
        """Predicted garrison of `planet_id` at `step` from now (None if unknown)."""
        tl = self.timelines.get(planet_id)
        if tl is None:
            return None
        return state_at_timeline(tl, step)[1]

    def incoming_enemy_eta(self, planet_id: int, my_id: int) -> int | None:
        """Min ETA among in-flight fleets owned by a non-`my_id` player
        currently targeting `planet_id`. None if no enemy fleet is
        inbound within the horizon.

        Used by the source-drain mission to gate "is this planet safe
        to empty"; safe iff `incoming_enemy_eta is None or eta >
        our_attack_eta + buffer`."""
        arrivals = self.ledger.get(planet_id)
        if not arrivals:
            return None
        enemy_etas = [eta for (eta, owner, ships) in arrivals if owner != my_id and ships > 0]
        if not enemy_etas:
            return None
        return min(enemy_etas)

    def incoming_enemy_eta_after(self, planet_id: int, my_id: int,
                                  after: int) -> int | None:
        """Min ETA among in-flight enemy fleets arriving STRICTLY AFTER
        `after`. None if no qualifying fleet exists.

        Used by `time_to_enemy_threat` with `arrival_eta > 0`: pre-arrival
        and same-step inbound fleets are resolved by `owner_at` /
        `ships_at` combat at our arrival, so they must NOT double-count
        as future threats; AND, the earliest inbound fleet may itself be
        pre-arrival while a LATER wave is a real threat — `incoming_enemy_eta`
        would silently drop the later wave because it only returns the
        minimum. This method surfaces the earliest post-`after` wave.
        Origin: B6 fix (2026-05-22 audit of f1774a7 orbital-safety patch).
        """
        arrivals = self.ledger.get(planet_id)
        if not arrivals:
            return None
        candidates = [
            eta for (eta, owner, ships) in arrivals
            if owner != my_id and ships > 0 and eta > after
        ]
        if not candidates:
            return None
        return min(candidates)

    def time_to_enemy_threat(self, planet_id: int, my_id: int, world,
                              arrival_eta: int = 0) -> int | None:
        """Earliest turn at which an enemy could have a fleet at
        `planet_id`. Considers BOTH (a) in-flight enemy fleets
        currently inbound, and (b) potential launches from every
        currently-stationary enemy-owned planet at its present
        garrison.

        Returns `None` if no enemy can plausibly threaten the planet
        (caller should treat as "saturate at game horizon").

        H22 helper for Hold-Aware Value scoring. See plan file
        2026-05-14 HAV section.

        `arrival_eta` (PI 2026-05-21 bug fix, completed 2026-05-22) —
        when > 0, the target and enemy planet positions are predicted
        at that future turn via `predict_relative`. This fixes a silent
        scoring bug where an orbiting target that rotates INTO enemy
        territory by our arrival was scored as safe (long expected_hold)
        because the threat ETA was computed from the CURRENT target
        position. Default 0 preserves the original "current position"
        semantics for source-safety callers (drain checks etc).

        Coverage notes (B5/B6/B7, completed in this audit pass):
        - B5: in-flight fleets that arrive at-or-before our arrival are
          resolved by combat at our arrival; only fleets arriving
          STRICTLY AFTER `arrival_eta` count as future threats.
        - B6: `incoming_enemy_eta` returns only the earliest inbound;
          when that earliest is pre-arrival, a later wave can be the
          real threat. Use `incoming_enemy_eta_after` to find it.
        - B7: enemy fleet aims at target-at-our-arrival, but target
          keeps rotating during enemy travel. Iterate a 5-step
          fixed-point on `enemy_eta_travel` for orbiting targets;
          fall through to the seed estimate on non-convergence.
        """
        target = world.planets_by_id.get(planet_id)
        if target is None:
            return None

        omega = float(getattr(world, "omega", 0.0))
        target_is_orbital = (
            arrival_eta > 0 and omega != 0.0
            and is_orbiting([target.id, target.owner, target.x, target.y,
                             target.radius, target.ships, target.production])
        )

        # Target position at our arrival. Use comet-aware dispatcher
        # so comet targets (which sit inside ROTATION_RADIUS_LIMIT and
        # would otherwise be rotated) get their real path position.
        tx, ty = planet_position_at(target, world, arrival_eta)

        best: int | None = None

        # (a) in-flight enemy fleets — B5 + B6 fix. Filter strictly to
        # fleets arriving AFTER our arrival; the earliest qualifying
        # fleet (not just the earliest overall) becomes the in-flight
        # threat ETA.
        if arrival_eta > 0:
            inbound = self.incoming_enemy_eta_after(planet_id, my_id,
                                                     arrival_eta)
        else:
            inbound = self.incoming_enemy_eta(planet_id, my_id)
        if inbound is not None:
            best = inbound

        # (b) potential launches from each enemy planet. When
        # arrival_eta > 0, predict the enemy's position at our arrival
        # too (assumes enemy launches immediately upon our capture).
        for p in world.planets_by_id.values():
            if p.id == planet_id:
                continue
            if p.owner == my_id or p.owner == -1:
                continue
            if p.ships <= 0:
                continue
            px, py = planet_position_at(p, world, arrival_eta)
            dx = tx - px
            dy = ty - py
            dist = (dx * dx + dy * dy) ** 0.5
            v = fleet_speed(int(p.ships))
            if v <= 0:
                continue
            eta_travel = int(-(-dist // v))  # math.ceil without import

            # B7 — 5-iteration fixed-point on `eta_travel` for orbiting
            # targets. The enemy fleet aims at target-at-our-arrival, but
            # during its travel the target keeps rotating; the actual
            # rendezvous point shifts. Iterate target_pos_at(arrival +
            # eta_travel) → recompute dist → recompute eta_travel until
            # |Δ| ≤ 1 (mirror of lib/aim.py:aim_orbiting). The enemy
            # planet's position at arrival_eta stays fixed (the assumed
            # launch moment).
            if target_is_orbital and eta_travel > 0:
                for _ in range(5):
                    tx_k, ty_k = planet_position_at(
                        target, world, arrival_eta + eta_travel,
                    )
                    dist_k = ((tx_k - px) ** 2 + (ty_k - py) ** 2) ** 0.5
                    new_eta = int(-(-dist_k // v))
                    if abs(new_eta - eta_travel) <= 1:
                        eta_travel = new_eta
                        break
                    eta_travel = new_eta

            threat_arrival = arrival_eta + eta_travel
            if best is None or threat_arrival < best:
                best = threat_arrival

        return best


# ---------------------------------------------------------------------------
# Comet lifetime — public helper used by ROI scoring sites
# ---------------------------------------------------------------------------


def _comet_paths_by_id(world) -> dict[int, tuple[list, int]]:
    """{planet_id: (path, path_index)} for every comet in `world.obs_raw`.

    `obs["comets"]` is a list of groups, each `{planet_ids, paths,
    path_index}`. `paths[i]` is the trajectory of `planet_ids[i]` — a
    list of `[x, y]` pairs. `path_index` is shared across the group.

    Mirrors `lib/mechanism._comet_path_lookup` but promoted to a public
    helper because ROI scoring now needs it as well.
    """
    raw = world.obs_raw
    if raw is None:
        return {}
    if isinstance(raw, dict):
        comets = raw.get("comets", [])
    else:
        comets = getattr(raw, "comets", [])
    out: dict[int, tuple[list, int]] = {}
    for group in comets or []:
        if hasattr(group, "keys"):
            planet_ids = list(group["planet_ids"])
            paths = list(group["paths"])
            path_index = int(group["path_index"])
        else:
            planet_ids = list(group.planet_ids)
            paths = list(group.paths)
            path_index = int(group.path_index)
        for idx, pid in enumerate(planet_ids):
            out[int(pid)] = (paths[idx], path_index)
    return out


def comet_remaining_lifetime(planet_id: int, world) -> int | None:
    """Steps until `planet_id` leaves the board.

    Returns `len(path) - path_index` for comets, or `None` for non-comet
    planets (which have no finite lifetime in this sense — the static /
    orbiting planets stay until end-of-game).

    Used by ROI scoring sites (`lib/missions/snipe.py`,
    `agents/simple/roi.py`, `agents/v2/main.py`) to cap `time_to_hold`
    on comet targets: sending a fleet to a comet that leaves before we
    arrive is wasted ships.
    """
    paths_by_id = _comet_paths_by_id(world)
    entry = paths_by_id.get(int(planet_id))
    if entry is None:
        return None
    path, path_index = entry
    return max(0, len(path) - path_index)


def comet_position_at(planet_id: int, world, lead_turns: int) -> tuple[float, float] | None:
    """Position of comet `planet_id` at `lead_turns` from now.

    Returns `(x, y)` from the comet's pre-computed path at index
    `path_index + lead_turns`, or `None` if the comet has exited the
    board by then (index past the end of the path) or if `planet_id`
    isn't a comet.

    Comets travel along polynomial paths at `cometSpeed=4` board
    units/turn (env: `orbit_wars.py::generate_comet_paths`), NOT around
    the central sun like orbital planets. So `lib.orbit.predict_relative`
    is wrong for comets — use this instead.
    """
    paths_by_id = _comet_paths_by_id(world)
    entry = paths_by_id.get(int(planet_id))
    if entry is None:
        return None
    path, path_index = entry
    idx = int(path_index) + int(lead_turns)
    if idx < 0 or idx >= len(path):
        return None
    point = path[idx]
    return float(point[0]), float(point[1])


def planet_position_at(planet, world, lead_turns: int) -> tuple[float, float]:
    """Predict `(x, y)` of `planet` at `lead_turns` from now — comet-aware.

    Dispatcher: comets route to `comet_position_at` (path lookup);
    orbital / static planets route to `lib.orbit.predict_relative`
    (which honors `is_orbiting`). Prefer this over raw `predict_relative`
    at call sites that have a `world` handle — raw `predict_relative` is
    physically wrong for comets (rotates them around the sun instead of
    advancing their path).

    `planet` may be a `Planet` namedtuple, a Plain Python list/tuple in
    the `[id, owner, x, y, radius, ships, prod]` shape used throughout
    `lib/`, or a `WorldModel.PlanetSnapshot`. Pid extraction is
    duck-typed: `.id` attribute first, else index `[0]`.

    Comets whose path has expired return the `(-1e6, -1e6)` OFF_BOARD
    sentinel, matching `lib.kinematic_table.lookup_relative` semantics.
    """
    pid = int(getattr(planet, "id", None) if hasattr(planet, "id") else planet[0])
    if pid in world.comet_ids:
        pos = comet_position_at(pid, world, lead_turns)
        if pos is not None:
            return pos
        return (-1e6, -1e6)
    return predict_relative(planet, world.omega, lead_turns)
