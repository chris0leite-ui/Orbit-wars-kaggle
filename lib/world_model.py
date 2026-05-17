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

# Raised 110 → 250 (2026-05-11): reinforce class was firing 0.2
# candidates/turn because long-runway threats were invisible past
# step 110 + eta. Matches the EPISODE_STEPS/2 framing in the score
# formula (`time_to_hold = 500 - step - eta`); timeline-build cost
# scales linearly so per-turn p95 should remain well under the 1s
# actTimeout. See audit/2026-05-11-v3-snipe-critical-review.md §P2.
DEFAULT_HORIZON = 250


class _MiniWorld:
    """Minimal duck-typed `World` for delegating to
    `lib.trajectory.predict_fleet_fate`. Carries the three fields the
    target reads: `omega`, `planets_by_id`, and `step` (the env's
    observed step number — needed so `predict_fleet_fate` can apply
    its `obs.step == 0` stationary-first-tick parity shift)."""

    __slots__ = ("omega", "planets_by_id", "step")

    def __init__(self, omega: float, planets, step: int = 1):
        self.omega = omega
        self.planets_by_id = {p.id: p for p in planets}
        # Default 1 = "assume obs.step >= 1 behaviour" (no parity
        # shift). Callers that have a real obs.step value (notably
        # `WorldModel.from_world`) should pass it explicitly.
        self.step = int(step)


def fleet_target_planet(
    fleet,
    planets,
    max_horizon: int = DEFAULT_HORIZON,
    omega: float = 0.0,
    obs_step: int = 1,
):
    """Trace `fleet` along its angle, find first planet it'd hit.

    Returns `(target_planet, eta_turns)` or `(None, None)` if no
    planet intersects the fleet's trajectory within `max_horizon`
    steps.

    Used to build the arrival ledger from in-flight fleets — the env
    doesn't expose a fleet's intended target, only its angle.

    Two paths:

    * `omega == 0.0` (default; non-rotating games) — fast static
      raycast (the pre-2026-05-15 behaviour). O(planets) per fleet.
    * `omega != 0.0` — orbit-aware step-by-step walk via
      `lib.trajectory.predict_fleet_fate`, which mirrors the env's
      collision semantics (orbital chord per step, swept-pair hit
      against every planet). Closes the documented inner-orbiting-
      target gap: for long-range fleets aimed at an inner planet, the
      planet has rotated away by arrival, and the static raycast
      mistakenly attributes the planet as the target.

    The orbital path is O(planets × min(max_horizon, board_diagonal /
    speed)) per fleet with early termination on first collision /
    OOB. Typical cost is ~1-2 ms per fleet on a 24-planet board.
    """
    spd = fleet_speed(fleet.ships)
    if spd <= 0:
        return None, None

    if omega == 0.0:
        return _static_first_hit(fleet, planets, max_horizon, spd)
    return _orbital_first_hit(fleet, planets, max_horizon, omega, obs_step)


def _static_first_hit(fleet, planets, max_horizon: int, spd: float):
    """Original static raycast — preserved verbatim for the
    no-rotation fast path. Does NOT account for orbital drift; only
    valid when no planet rotates (`omega == 0.0`).
    """
    dir_x = math.cos(fleet.angle)
    dir_y = math.sin(fleet.angle)
    best_planet = None
    best_turns = None
    for p in planets:
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
    if best_planet is None:
        return None, None
    return best_planet, int(math.ceil(best_turns))


def _orbital_first_hit(fleet, planets, max_horizon: int, omega: float,
                        obs_step: int = 1):
    """Orbit-aware first-hit via `predict_fleet_fate`. Synthesises a
    zero-radius source at the fleet's current position so the
    function's spawn-offset computation yields the fleet's actual
    (x, y); reads `hit_planet_id` regardless of whether
    `predict_fleet_fate` labels the hit `"target"` or `"planet"` (the
    `target=` arg is irrelevant for our which-planet-does-it-hit
    question).
    """
    if not planets:
        return None, None

    # Local import to avoid a circular if lib.trajectory ever imports
    # lib.world_model in the future.
    from kaggle_environments.envs.orbit_wars.orbit_wars import Planet as _Planet
    from lib.trajectory import predict_fleet_fate

    cos_a = math.cos(fleet.angle)
    sin_a = math.sin(fleet.angle)
    # Position the synthetic source so `spawn = src + (r+0.1)*direction`
    # in predict_fleet_fate lands at the fleet's actual (x, y).
    fake_src = _Planet(
        id=-1,
        owner=fleet.owner,
        x=fleet.x - cos_a * 0.1,
        y=fleet.y - sin_a * 0.1,
        radius=0.0,
        ships=0,
        production=0,
    )
    mini_world = _MiniWorld(omega=omega, planets=planets, step=obs_step)
    fate = predict_fleet_fate(
        src=fake_src,
        target=planets[0],  # dummy; we only read hit_planet_id
        aim_angle=fleet.angle,
        ships=fleet.ships,
        world=mini_world,
        max_steps=max_horizon,
    )
    if fate.outcome in ("target", "planet") and fate.hit_planet_id is not None:
        return mini_world.planets_by_id[fate.hit_planet_id], int(fate.step)
    return None, None


def build_arrival_ledger(
    fleets,
    planets,
    horizon: int = DEFAULT_HORIZON,
    omega: float = 0.0,
    obs_step: int = 1,
):
    """`{planet_id: [(eta, owner, ships), ...]}` for in-flight fleets.

    Fleets that won't hit any planet within `horizon` are dropped
    (they will exit the board or die in sun/non-target collision —
    out of scope for the timeline).

    `omega` is forwarded to `fleet_target_planet` so inner orbiting
    planets are attributed correctly. Pass `world.omega` from the
    caller (`WorldModel.from_world` does this); leaving it at the
    default `0.0` preserves the pre-2026-05-15 static raycast for
    callers in non-rotating games.
    """
    ledger: dict[int, list[tuple[int, int, int]]] = {p.id: [] for p in planets}
    for fleet in fleets:
        target, eta = fleet_target_planet(
            fleet, planets, horizon, omega=omega, obs_step=obs_step,
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
        directly from the raw obs because `World` doesn't materialise them."""
        raw = world.obs_raw
        if isinstance(raw, dict):
            fleets_raw = raw.get("fleets", [])
        else:
            fleets_raw = getattr(raw, "fleets", [])

        from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet  # local import — keeps lib/ env-free
        fleets = [Fleet(*f) for f in fleets_raw]
        planets = list(world.planets_by_id.values())
        omega = float(getattr(world, "omega", 0.0))
        obs_step = int(getattr(world, "step", 1))
        ledger = build_arrival_ledger(
            fleets, planets, horizon, omega=omega, obs_step=obs_step,
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

    def time_to_enemy_threat(self, planet_id: int, my_id: int, world) -> int | None:
        """Earliest turn at which an enemy could have a fleet at
        `planet_id`. Considers BOTH (a) in-flight enemy fleets
        currently inbound, and (b) potential launches from every
        currently-stationary enemy-owned planet at its present
        garrison.

        Returns `None` if no enemy can plausibly threaten the planet
        (caller should treat as "saturate at game horizon").

        H22 helper for Hold-Aware Value scoring. See plan file
        2026-05-14 HAV section. The "potential launch" leg uses
        `lib.scoring.eta_proxy(enemy_planet, target_planet)` — that
        helper already estimates ETA from `ceil(dist / fleet_speed(
        target.ships+1))`. We override its target argument so the
        ship-count proxy is the LAUNCHING planet's garrison, not the
        target's.
        """
        target = world.planets_by_id.get(planet_id)
        if target is None:
            return None

        best: int | None = None

        # (a) in-flight enemy fleets — reuse existing helper.
        inbound = self.incoming_enemy_eta(planet_id, my_id)
        if inbound is not None:
            best = inbound

        # (b) potential launches from each enemy planet at its current
        #     garrison.
        for p in world.planets_by_id.values():
            if p.id == planet_id:
                continue
            if p.owner == my_id or p.owner == -1:
                continue
            if p.ships <= 0:
                continue
            dx = target.x - p.x
            dy = target.y - p.y
            dist = (dx * dx + dy * dy) ** 0.5
            v = fleet_speed(int(p.ships))
            if v <= 0:
                continue
            eta = int(-(-dist // v))  # math.ceil without import
            if best is None or eta < best:
                best = eta

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
