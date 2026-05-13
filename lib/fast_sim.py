"""Fast steppable game-state Snapshot for deep-lookahead inner loops.

Why this exists: `env.clone() + env.step()` is bit-exact but ~5.6 ms
per simulated step (audit/2026-05-11-lookahead-phase2-forward-sim.md),
mostly Environment-level overhead — action-schema validation, structify,
state-history append, full-state deepcopy on clone. The actual game
physics inside `orbit_wars.interpreter()` is a small fraction of that.

This module bypasses the Environment by calling `interpreter()` directly
on a minimal Snapshot. Expected speedup: 20-50x on the per-step path.
Faithful semantics: same interpreter, same RNG path (seeded comet
spawns at steps 50/150/250/350/450), parity-tested in
`tests/test_fast_sim_parity.py`.

API surface (mirrors `lib/lookahead.py::score_action`'s inner loop but
on Snapshots, not Environments):

    snap = from_obs(obs, configuration, episode_seed, num_seats=2)
    snap = step(snap, [action_p0, action_p1])         # tick once
    snap = rollout(snap, K=30, policies=[pol_p0, pol_p1])
    totals = ship_totals(snap)                         # scoring head
    snap2 = clone(snap)                                # branch the tree

Caveats:
- The `episode_seed` is not exposed in the live ladder's
  `configuration` (scrubbed at env init for fairness). Without it,
  future comet spawns at steps 50/150/250/350/450 will use seed=0 and
  diverge from the real game. For sims that don't cross a spawn
  boundary (≤ ~50 steps from a current step that isn't 49/149/...)
  the rebuild is bit-exact regardless. Phase 2 probe accepted this
  trade-off (audit:14-19).
- `lib.lookahead.env_from_obs` does the same with an `Environment`;
  this is the same idea reduced to its physics-only core.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from kaggle_environments.utils import Struct

from lib.game.interpreter import interpreter as _orbit_wars_interpreter
from lib.game.interpreter import BOARD_SIZE as _GAME_BOARD_SIZE
from lib.game.interpreter import CENTER as _GAME_CENTER
from lib.game.interpreter import ROTATION_RADIUS_LIMIT as _GAME_RRL


# Default configuration matching kaggle_environments.envs.orbit_wars's
# `orbit_wars.json` defaults; kept here so fast_sim doesn't depend on
# any I/O at import time.
DEFAULT_CONFIG: dict[str, Any] = {
    "episodeSteps": 500,
    "shipSpeed": 6.0,
    "sunRadius": 10.0,
    "boardSize": 100.0,
    "cometSpeed": 4.0,
    "actTimeout": 1.0,
    "agentTimeout": 60.0,
    "runTimeout": 1200.0,
}


class _FakeEnv:
    """Minimal env shim that satisfies what `interpreter()` reads.

    The interpreter touches `env.configuration`, `env.info`, `env.done`
    only (orbit_wars.py:335, 348, 363, 406, 438, 570, 686). We provide
    just those three attributes. `info["seed"]` is read once per comet
    spawn (orbit_wars.py:438-440) — passing it through keeps comet RNG
    deterministic.

    `comet_path_cache` is a dict {(episode_seed, spawn_step):
    (comet_paths_or_None, comet_ships_or_None)} populated lazily by the
    interpreter. It's SHARED across clones so all branches of a
    lookahead inherit the same cache and amortise the ~100 ms
    generate_comet_paths cost across rollouts.

    `planet_position_cache` is a dict {planet_id: list[(x, y)]} keyed
    by initial-planet id, indexed by absolute step. Pre-computed once
    in `from_obs()` for every rotating planet; the interpreter uses it
    instead of recomputing `atan2/cos/sin` each step. SHARED across
    clones too. ~240 KB at 30 rotating planets × 500 steps.
    """
    __slots__ = (
        "configuration", "info", "done",
        "comet_path_cache", "planet_position_cache",
    )

    def __init__(self, configuration: Struct, episode_seed: int) -> None:
        self.configuration = configuration
        self.info = {"seed": episode_seed}
        self.done = False
        self.comet_path_cache = {}
        self.planet_position_cache = {}


@dataclass
class Snapshot:
    """Forward-simulator state. `state` is a list of `Struct` per seat,
    each with `observation` (also a Struct), `action`, `status`,
    `reward`. This matches what the env's interpreter expects to mutate.

    Don't mutate directly; use `step()` / `clone()`. `from_obs()` is the
    only sanctioned constructor for production code.
    """
    state: list[Struct]
    fake_env: _FakeEnv
    episode_seed: int

    @property
    def obs(self) -> Struct:
        """Primary observation (seat 0). The interpreter mutates this in
        place; the other seats hold the same list/dict references."""
        return self.state[0].observation

    @property
    def step_idx(self) -> int:
        return int(self.obs.get("step", 0))

    @property
    def done(self) -> bool:
        return self.fake_env.done

    @property
    def num_seats(self) -> int:
        return len(self.state)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


# The mutable fields on `state[i].observation` that fast_sim cares about.
# Anything not in this tuple is either immutable across the episode
# (e.g. `player`) or not read by the interpreter.
_OBS_KEYS = (
    "planets", "fleets", "comets", "comet_planet_ids",
    "initial_planets", "angular_velocity", "step", "next_fleet_id",
)


def _configuration_struct(configuration) -> Struct:
    """Coerce a configuration (dict / SimpleNamespace / Struct) into a
    Struct with the defaults filled in.

    The env's interpreter reads `configuration.shipSpeed`,
    `configuration.cometSpeed`, `configuration.episodeSteps` — Struct's
    dual attr+dict access satisfies both styles.
    """
    cfg = dict(DEFAULT_CONFIG)
    if configuration is not None:
        if isinstance(configuration, dict):
            cfg.update(configuration)
        else:
            for k in DEFAULT_CONFIG:
                v = getattr(configuration, k, None)
                if v is not None:
                    cfg[k] = v
    return Struct(**cfg)


def _read_obs_field(obs: Any, key: str, default: Any = None) -> Any:
    """Dual dict-or-attr read, matching the env's `get()` helper."""
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def from_obs(
    obs: Any,
    configuration: Any = None,
    *,
    episode_seed: int = 0,
    num_seats: int = 2,
) -> Snapshot:
    """Build a Snapshot from an agent-visible observation.

    `obs` accepts both dict-style (live ladder) and Struct-style (env
    in-memory). Pulls each mutable field via `_read_obs_field` and
    deep-copies it so the Snapshot is independent of the source obs.

    `configuration` is folded into the env defaults; pass the agent's
    `configuration` argument unchanged.

    `episode_seed` is the env's `info["seed"]` (or any deterministic
    int). Required for bit-exact comet spawns; if unknown, pass 0 and
    accept divergence past a spawn boundary (see module docstring).

    `num_seats` defaults to 2P; pass 4 for FFA. Single-seat (`num_seats
    = 1`) is allowed for tests but isn't a real game configuration.
    """
    config_struct = _configuration_struct(configuration)

    obs_data: dict[str, Any] = {}
    for k in _OBS_KEYS:
        v = _read_obs_field(obs, k)
        if v is not None:
            obs_data[k] = copy.deepcopy(v)
    # Make sure required scalar fields are present (interpreter reads
    # `step` and `next_fleet_id` with `get(..., default)`; the defaults
    # are safe but having them explicit avoids attribute fallbacks).
    obs_data.setdefault("step", 0)
    obs_data.setdefault("next_fleet_id", 0)
    obs_data.setdefault("fleets", [])
    obs_data.setdefault("comets", [])
    obs_data.setdefault("comet_planet_ids", [])
    obs_data.setdefault("initial_planets", [list(p) for p in obs_data.get("planets", [])])
    obs_data.setdefault("angular_velocity", 0.0)

    state: list[Struct] = []
    obs0 = Struct(**obs_data)
    obs0.player = 0
    state.append(Struct(
        observation=obs0,
        action=None,
        status="ACTIVE",
        reward=0,
        info={},
    ))
    # Other seats share the same mutable references — same aliasing
    # behaviour the env's `interpreter()` sets up on init
    # (orbit_wars.py:393-402) and re-applies after each step
    # (orbit_wars.py:676-682).
    for i in range(1, num_seats):
        obs_i = Struct(**{k: getattr(obs0, k) for k in obs_data})
        obs_i.player = i
        state.append(Struct(
            observation=obs_i,
            action=None,
            status="ACTIVE",
            reward=0,
            info={},
        ))

    fake_env = _FakeEnv(config_struct, episode_seed)
    _populate_planet_position_cache(fake_env, obs0)
    return Snapshot(state=state, fake_env=fake_env, episode_seed=episode_seed)


def _populate_planet_position_cache(fake_env, obs0) -> None:
    """Pre-compute orbital positions for every rotating planet at every
    step of the episode. Eliminates per-step atan2/cos/sin for the planet
    path computation. Storage ~240 KB at 30 planets × 500 steps.
    """
    import math as _math
    cache = fake_env.planet_position_cache
    angular_velocity = float(obs0.angular_velocity)
    episode_steps = int(fake_env.configuration.episodeSteps)
    initial_planets = obs0.initial_planets
    comet_pid_set = set(obs0.comet_planet_ids)
    sqrt = _math.sqrt
    cos = _math.cos
    sin = _math.sin
    atan2 = _math.atan2
    for ip in initial_planets:
        pid = ip[0]
        if pid in comet_pid_set:
            continue
        dx = ip[2] - _GAME_CENTER
        dy = ip[3] - _GAME_CENTER
        r = sqrt(dx * dx + dy * dy)
        if r + ip[4] >= _GAME_RRL:
            # Non-rotating; no cache entry needed (interpreter keeps the
            # static position).
            continue
        initial_angle = atan2(dy, dx)
        # Index 0 is the position AT step 0 (which equals initial position).
        positions = []
        for s in range(episode_steps + 1):
            theta = initial_angle + angular_velocity * s
            positions.append((_GAME_CENTER + r * cos(theta), _GAME_CENTER + r * sin(theta)))
        cache[pid] = positions


# ---------------------------------------------------------------------------
# Cloning
# ---------------------------------------------------------------------------


def clone(snap: Snapshot) -> Snapshot:
    """Targeted deep-copy of mutating fields. Faster than `copy.deepcopy`.

    What gets copied (the interpreter mutates these):
    - `planets`: list + each inner list (planet ships incremented,
      positions overwritten — `orbit_wars.py:514, 615`).
    - `fleets`: list + each inner list (positions advanced —
      `orbit_wars.py:580-581`).
    - `initial_planets`: list (filtered on comet expiration), inner
      lists are read-only after init so shared safely.
    - `comets`: list + per-group dict + `planet_ids` list. `paths` is
      built once and only read; we share it.
    - `comet_planet_ids`: list (filtered on expiration / appended on
      spawn).
    - Scalars: `step`, `next_fleet_id`, `angular_velocity`, `player`.

    What gets shared (immutable across the episode):
    - `fake_env.configuration` (Struct).
    - Comet `paths`.
    """
    src0 = snap.state[0].observation

    obs0 = Struct(
        planets=[list(p) for p in src0.planets],
        fleets=[list(f) for f in src0.fleets],
        initial_planets=[list(p) for p in src0.initial_planets],
        comet_planet_ids=list(src0.comet_planet_ids),
        comets=[
            {
                "planet_ids": list(g["planet_ids"]),
                "paths": g["paths"],
                "path_index": g["path_index"],
            }
            for g in src0.comets
        ],
        angular_velocity=src0.angular_velocity,
        step=int(src0.get("step", 0)),
        next_fleet_id=int(src0.next_fleet_id),
        player=0,
    )
    new_state: list[Struct] = [Struct(
        observation=obs0,
        action=None,
        status=snap.state[0].status,
        reward=snap.state[0].reward,
        info={},
    )]
    for i in range(1, snap.num_seats):
        obs_i = Struct(
            planets=obs0.planets,
            fleets=obs0.fleets,
            initial_planets=obs0.initial_planets,
            comet_planet_ids=obs0.comet_planet_ids,
            comets=obs0.comets,
            angular_velocity=obs0.angular_velocity,
            step=obs0.step,
            next_fleet_id=obs0.next_fleet_id,
            player=i,
        )
        new_state.append(Struct(
            observation=obs_i,
            action=None,
            status=snap.state[i].status,
            reward=snap.state[i].reward,
            info={},
        ))

    fake_env = _FakeEnv(snap.fake_env.configuration, snap.episode_seed)
    fake_env.done = snap.fake_env.done
    # Share both caches with the parent so all lookahead branches benefit
    # from work done by any one of them (comet generation, planet orbits).
    fake_env.comet_path_cache = snap.fake_env.comet_path_cache
    fake_env.planet_position_cache = snap.fake_env.planet_position_cache
    return Snapshot(state=new_state, fake_env=fake_env, episode_seed=snap.episode_seed)


# ---------------------------------------------------------------------------
# Stepping
# ---------------------------------------------------------------------------


def step(
    snap: Snapshot,
    actions_per_seat: Sequence[list],
    *,
    in_place: bool = False,
) -> Snapshot:
    """Advance the snapshot by one tick.

    `actions_per_seat` is a list of `[[src_id, angle, ships], ...]` per
    seat (matches the env action format). Pass `[]` for seats that
    don't launch this turn.

    Returns a new Snapshot unless `in_place=True`, in which case `snap`
    is mutated and returned.

    The actual game logic comes from
    `kaggle_environments.envs.orbit_wars.orbit_wars.interpreter` — same
    physics, RNG, combat resolution the real env uses. The fast path
    skips: action-schema validation (`core.py:262`), structify wrapping
    (`core.py:600`), state-history append (`core.py:277`), and the
    redundant per-step state deepcopy that Environment.clone() pays.
    """
    if snap.fake_env.done:
        # Mirror Environment.step()'s "cannot step a done env"
        # convention without raising (rollout loops check `snap.done`
        # to exit; raising would surprise them).
        if not in_place:
            snap = clone(snap)
        return snap

    if not in_place:
        snap = clone(snap)

    # Wire per-seat actions.
    for i, action in enumerate(actions_per_seat):
        snap.state[i].action = action
    # Any seat we didn't get an action for gets [] (no-op).
    for i in range(len(actions_per_seat), snap.num_seats):
        snap.state[i].action = []

    # Call the env's interpreter directly. Mutates state in place.
    _orbit_wars_interpreter(snap.state, snap.fake_env)

    # Post-interpreter bookkeeping that core.py handles for us:
    #   1. Increment observation.step (core.py:602).
    #   2. Update fake_env.done if any seat went DONE.
    obs0 = snap.state[0].observation
    obs0.step = int(obs0.get("step", 0)) + 1
    for i in range(1, snap.num_seats):
        snap.state[i].observation.step = obs0.step

    if any(s.status == "DONE" for s in snap.state):
        snap.fake_env.done = True

    return snap


# ---------------------------------------------------------------------------
# Scoring head
# ---------------------------------------------------------------------------


def ship_totals(snap: Snapshot) -> dict[int, float]:
    """Sum ships on owned planets + in-flight fleets, per owner.

    Same scoring head as `lib/lookahead.py::_ship_total_by_owner` and
    the Phase 2 probe (audit:69-76); kept here so future consumers
    don't import from `lookahead`.
    """
    totals: dict[int, float] = {}
    obs0 = snap.state[0].observation
    for p in obs0.planets:
        owner = int(p[1])
        if owner >= 0:
            totals[owner] = totals.get(owner, 0.0) + float(p[5])
    for f in obs0.fleets:
        owner = int(f[1])
        if owner >= 0:
            totals[owner] = totals.get(owner, 0.0) + float(f[6])
    return totals


def delta_us_minus_them(snap: Snapshot, my_id: int) -> float:
    """`(our ships) - (sum of other seats' ships)`. The Phase 2 scoring
    scalar — the value that the AUC = 0.952 probe at K=50 measured.
    """
    t = ship_totals(snap)
    ours = t.get(my_id, 0.0)
    theirs = sum(v for k, v in t.items() if k != my_id)
    return ours - theirs


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


Policy = Callable[[Any], list]


def rollout(
    snap: Snapshot,
    K: int,
    policies: Sequence[Policy],
    *,
    in_place: bool = False,
) -> Snapshot:
    """Roll forward up to `K` ticks under per-seat policies.

    `policies[i]` is `Callable(obs) -> action`. Each tick:
    1. For each seat, call `policies[i](snap.state[i].observation)` to
       get the action.
    2. `step(snap, actions, in_place=True)`.
    3. If `snap.done`, exit early.

    Returns the terminal Snapshot. Equivalent to
    `lib/lookahead.py::score_action`'s inner loop but operating on a
    Snapshot, with the env-overhead stripped.
    """
    if not in_place:
        snap = clone(snap)
    if len(policies) != snap.num_seats:
        raise ValueError(
            f"need {snap.num_seats} policies, got {len(policies)}"
        )
    for _ in range(K):
        if snap.fake_env.done:
            break
        actions = [policies[i](snap.state[i].observation) for i in range(snap.num_seats)]
        snap = step(snap, actions, in_place=True)
    return snap
