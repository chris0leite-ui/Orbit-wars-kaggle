"""Phase 5 tests for lib.trajectory_layer — per-planet combat outcomes.

`World.combat_at(pid, t)` returns a verbose `CombatOutcome` for every
turn at which the planet has at least one fleet arrival. The
state-evolution of the verbose path MUST match the plain path
(`World.ownership_at`); the verbose path adds attacker/runner-up
detail.

Tests cover the 4 combat rules from `data/README.md`:
1. Same-step arrivals grouped by owner; ships summed.
2. Largest attacker fights second-largest; difference survives.
3a. Survivor.owner == garrison.owner → reinforce.
3b. Survivor.owner != garrison.owner → fights garrison.
4. Two-way tie among attackers → all destroyed.

Plus production semantics:
- Neutral (owner=-1) doesn't produce.
- Just-captured planet doesn't produce on capture turn (it produces
  the following turn).
"""

from __future__ import annotations

import math
import random
from typing import Any

import pytest

pytestmark = pytest.mark.slow

from kaggle_environments import make

from lib.combat import resolve_arrivals
from lib.fast_sim import clone as fs_clone
from lib.fast_sim import from_obs as fs_from_obs
from lib.fast_sim import step as fs_step
from lib.trajectory_layer import (
    Arrival,
    CombatOutcome,
    World,
    _simulate_planet_timeline,
    _simulate_timeline_with_combat_log,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _toy_world(planets: list, fleets: list,
               *, step: int = 0, angular_velocity: float = 0.0,
               my_id: int = 0,
               ) -> World:
    """Build a synthetic World from raw planet/fleet rows. Each `planet`
    is `[id, owner, x, y, radius, ships, production]`; each `fleet` is
    `[id, owner, x, y, angle, from_pid, ships]`."""
    obs = {
        "step": step,
        "player": my_id,
        "angular_velocity": angular_velocity,
        "planets": planets,
        "initial_planets": planets,
        "fleets": fleets,
        "comet_planet_ids": [],
        "comets": [],
        "next_fleet_id": max((f[0] for f in fleets), default=-1) + 1,
    }
    return World.from_obs(obs)


# ---------------------------------------------------------------------------
# Rule-by-rule hand-built scenarios
# ---------------------------------------------------------------------------


def test_combat_at_returns_none_for_no_arrival_turn():
    """A planet with no inbound fleet at turn `t` → None (no combat).

    `ownership_at` may still return a non-trivial value (production
    accrues) — but `combat_at` only fires on arrival-driven events.
    """
    world = _toy_world(
        planets=[[0, 0, 30.0, 50.0, 1.0, 10, 1]],
        fleets=[],
    )
    assert world.combat_at(0, 1) is None
    assert world.combat_at(0, 5) is None
    assert world.combat_at(0, 50) is None


def test_combat_at_single_attacker_captures_neutral():
    """Rule 2 + 3b: one attacker (5 ships) vs neutral garrison
    (3 ships) → attacker wins, owns the planet, garrison = 2.

    Geometry chosen to avoid the sun: source at (10, 80), target
    at (80, 80) — horizontal line at y=80, well clear of the sun
    at (50, 50) which has safety radius 10.5.
    """
    world = _toy_world(
        planets=[
            [0, -1, 80.0, 80.0, 1.0, 3, 0],   # target, neutral, no prod
            [1, 0, 10.0, 80.0, 2.0, 50, 1],   # our source
        ],
        fleets=[
            # 5-ship fleet just east of source, flying east toward target.
            [0, 0, 12.0, 80.0, 0.0, 1, 5],
        ],
    )
    log = world._combat_log_for(0, horizon=100)
    assert len(log) == 1, f"expected exactly 1 combat event, got {log}"
    eta = next(iter(log.keys()))
    outcome = log[eta]
    assert outcome.winner_owner == 0
    assert outcome.surviving_ships == 2.0
    assert outcome.pre_garrison_owner == -1
    assert outcome.pre_garrison_ships == 3.0
    assert outcome.attackers == ((0, 5),)
    assert outcome.runner_up_owner == -1
    assert outcome.runner_up_ships == 0
    assert outcome.is_tie is False


def test_combat_at_reinforce_same_owner():
    """Rule 3a: a 5-ship fleet of owner 0 arriving at owner-0 planet
    with 8 ships → garrison = 13. Same sun-clear geometry."""
    world = _toy_world(
        planets=[
            [0, 0, 80.0, 80.0, 1.0, 8, 0],
            [1, 0, 10.0, 80.0, 2.0, 50, 1],
        ],
        fleets=[
            [0, 0, 12.0, 80.0, 0.0, 1, 5],
        ],
    )
    log = world._combat_log_for(0, horizon=100)
    assert len(log) == 1
    outcome = next(iter(log.values()))
    assert outcome.winner_owner == 0
    assert outcome.surviving_ships == 13.0
    assert outcome.pre_garrison_ships == 8.0
    assert outcome.attackers == ((0, 5),)


def test_combat_at_tie_destroys_all_attackers():
    """Rule 4: two equal attackers (5 vs 5) → all destroyed,
    garrison unchanged."""
    # Hand-build the timeline directly with a synthetic arrival group;
    # easier than constructing fleets that arrive on the same turn
    # from different sources via ray-cast.
    planet_view_planets = [[0, -1, 40.0, 50.0, 1.0, 10, 0]]
    world = _toy_world(planets=planet_view_planets, fleets=[])
    planet = world.planet_by_id(0)
    arrivals = [
        Arrival(eta=5, owner=0, ships=5, fleet_id=100),
        Arrival(eta=5, owner=1, ships=5, fleet_id=101),
    ]
    _timeline, log = _simulate_timeline_with_combat_log(
        planet, arrivals, horizon=10,
    )
    assert 5 in log
    outcome = log[5]
    # Two-way tie → all attackers destroyed; garrison stays.
    assert outcome.winner_owner == -1
    assert outcome.surviving_ships == 10.0
    assert outcome.pre_garrison_ships == 10.0
    assert outcome.is_tie is True
    # Attackers tuple ordered descending by ships (or by owner on tie).
    assert set(outcome.attackers) == {(0, 5), (1, 5)}
    assert outcome.runner_up_ships == 5


def test_combat_at_largest_minus_second_largest():
    """Rule 2: three attackers (7, 4, 2) → 7-4 = 3 survives, then
    fights neutral garrison of 1 → owner 0, garrison = 2."""
    planet_view_planets = [[0, -1, 40.0, 50.0, 1.0, 1, 0]]
    world = _toy_world(planets=planet_view_planets, fleets=[])
    planet = world.planet_by_id(0)
    arrivals = [
        Arrival(eta=3, owner=0, ships=7, fleet_id=100),
        Arrival(eta=3, owner=1, ships=4, fleet_id=101),
        Arrival(eta=3, owner=2, ships=2, fleet_id=102),
    ]
    _timeline, log = _simulate_timeline_with_combat_log(
        planet, arrivals, horizon=10,
    )
    outcome = log[3]
    assert outcome.winner_owner == 0
    assert outcome.surviving_ships == 2.0   # 3 (survivor) - 1 (garrison)
    assert outcome.attackers == ((0, 7), (1, 4), (2, 2))
    assert outcome.runner_up_owner == 1
    assert outcome.runner_up_ships == 4
    assert outcome.is_tie is False


def test_combat_at_neutral_doesnt_produce():
    """Neutral planets don't produce. A neutral with 5 ships at t=0
    still has 5 ships at t=10 (no arrivals, no production)."""
    world = _toy_world(
        planets=[[0, -1, 40.0, 50.0, 1.0, 5, 3]],  # production=3 IGNORED for neutral
        fleets=[],
    )
    owner, ships = world.ownership_at(0, 10)
    assert owner == -1
    assert ships == 5.0


def test_combat_at_captured_planet_produces_following_turn():
    """A planet captured at turn t starts producing the NEXT turn
    (env semantics: production accrues BEFORE arrival resolution).

    Setup: neutral planet (0 ships, 5 production) captured at t=2 by
    a 1-ship attacker → garrison = 1 at t=2.
    At t=3: production accrues → 1 + 5 = 6.
    At t=4: production accrues again → 6 + 5 = 11.
    """
    planet_view_planets = [[0, -1, 40.0, 50.0, 1.0, 0, 5]]
    world = _toy_world(planets=planet_view_planets, fleets=[])
    planet = world.planet_by_id(0)
    arrivals = [Arrival(eta=2, owner=0, ships=1, fleet_id=100)]
    timeline, log = _simulate_timeline_with_combat_log(
        planet, arrivals, horizon=4,
    )
    # t=1 — still neutral, no production (rule), no combat.
    assert timeline["owner_at"][1] == -1
    assert timeline["ships_at"][1] == 0.0
    # t=2 — captured. Garrison = 1 - 0 = 1.
    assert timeline["owner_at"][2] == 0
    assert timeline["ships_at"][2] == 1.0
    # t=3 — production fires (now owned).
    assert timeline["owner_at"][3] == 0
    assert timeline["ships_at"][3] == 6.0
    # t=4 — keeps producing.
    assert timeline["ships_at"][4] == 11.0


# ---------------------------------------------------------------------------
# Random-arrival fuzzy comparison vs `lib.combat.resolve_arrivals`
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trial", range(100))
def test_combat_at_random_scenarios_match_resolve_arrivals(trial: int):
    """For 100 random arrival-group scenarios, the combat outcome
    matches a direct `resolve_arrivals` call."""
    rng = random.Random(trial * 7919 + 3)
    garrison_owner = rng.choice([-1, 0, 1, 2, 3])
    garrison_ships = float(rng.randint(0, 50))
    production = rng.randint(0, 5)

    # 1-4 random attackers.
    n_att = rng.randint(1, 4)
    attackers: list[tuple[int, int]] = []
    for _ in range(n_att):
        attackers.append((rng.randint(-1, 3), rng.randint(1, 30)))

    # Build a synthetic planet + arrivals.
    planet_view_planets = [[0, garrison_owner, 40.0, 50.0, 1.0,
                             garrison_ships, production]]
    world = _toy_world(planets=planet_view_planets, fleets=[])
    planet = world.planet_by_id(0)
    arrivals = [Arrival(eta=1, owner=o, ships=s, fleet_id=100 + i)
                for i, (o, s) in enumerate(attackers)]

    _timeline, log = _simulate_timeline_with_combat_log(
        planet, arrivals, horizon=2,
    )
    outcome = log[1]

    # Compute expected via direct resolve_arrivals (with production
    # accrued first if garrison_owner != -1).
    g_ships = garrison_ships
    if garrison_owner != -1:
        g_ships += production
    expected_owner, expected_ships = resolve_arrivals(
        garrison_owner, g_ships, attackers,
    )
    assert outcome.winner_owner == expected_owner
    assert math.isclose(outcome.surviving_ships,
                         max(0.0, expected_ships), abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Equivalence with the plain timeline
# ---------------------------------------------------------------------------


def test_verbose_and_plain_timeline_state_evolution_identical():
    """The verbose `_simulate_timeline_with_combat_log` and the plain
    `_simulate_planet_timeline` produce identical `owner_at` /
    `ships_at` series. The combat log is strictly additive."""
    planet_view_planets = [[0, -1, 40.0, 50.0, 1.0, 10, 2]]
    world = _toy_world(planets=planet_view_planets, fleets=[])
    planet = world.planet_by_id(0)
    arrivals = [
        Arrival(eta=3, owner=0, ships=4, fleet_id=100),
        Arrival(eta=5, owner=1, ships=8, fleet_id=101),
        Arrival(eta=7, owner=0, ships=15, fleet_id=102),
    ]
    plain = _simulate_planet_timeline(planet, arrivals, horizon=10)
    verbose, _log = _simulate_timeline_with_combat_log(
        planet, arrivals, horizon=10,
    )
    assert plain["owner_at"] == verbose["owner_at"]
    assert plain["ships_at"] == verbose["ships_at"]
    assert plain["horizon"] == verbose["horizon"]


# ---------------------------------------------------------------------------
# Cache sharing — combat_at and ownership_at populate the same timeline
# ---------------------------------------------------------------------------


def test_combat_at_populates_timeline_cache():
    """Calling `combat_at(pid, t)` should also fill the
    `_timeline_cache` for that (pid, horizon) so a subsequent
    `ownership_at(pid, t)` reuses the work."""
    world = _toy_world(
        planets=[[0, -1, 40.0, 50.0, 1.0, 5, 1],
                 [1, 0, 10.0, 50.0, 2.0, 50, 1]],
        fleets=[[0, 0, 12.0, 50.0, 0.0, 1, 5]],
    )
    assert (0, 250) not in world._timeline_cache
    world.combat_at(0, 5)
    assert (0, 250) in world._timeline_cache
    # ownership_at should now NOT recompute.
    n_before = len(world._timeline_cache)
    world.ownership_at(0, 10)
    assert len(world._timeline_cache) == n_before


def test_combat_at_lazy_per_planet():
    """`combat_at(pid_A, t)` materialises ONE combat log; pid_B is
    untouched until queried separately."""
    world = _toy_world(
        planets=[[0, -1, 40.0, 50.0, 1.0, 5, 1],
                 [1, -1, 60.0, 50.0, 1.0, 5, 1]],
        fleets=[],
    )
    assert len(world._combat_log_cache) == 0
    world.combat_at(0, 5)
    keys = list(world._combat_log_cache.keys())
    assert keys == [(0, 250)]
    world.combat_at(1, 5)
    keys = sorted(world._combat_log_cache.keys())
    assert keys == [(0, 250), (1, 250)]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_combat_at_unknown_planet():
    world = _toy_world(planets=[], fleets=[])
    assert world.combat_at(999, 5) is None


def test_combat_at_non_positive_t():
    world = _toy_world(
        planets=[[0, -1, 40.0, 50.0, 1.0, 5, 1]],
        fleets=[],
    )
    assert world.combat_at(0, 0) is None
    assert world.combat_at(0, -3) is None


# ---------------------------------------------------------------------------
# Real-game parity: combat predictions land in fast_sim ground truth
# ---------------------------------------------------------------------------


def _step_env_to_obs(seed: int, warmup: int, num_seats: int,
                     ) -> tuple[Any, int]:
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=num_seats)
    rng = random.Random(seed * 11 + 3)
    for _ in range(warmup):
        obs0 = env.state[0].observation
        planets = (obs0["planets"] if isinstance(obs0, dict)
                   else obs0.planets)
        actions: list[list] = [[] for _ in range(num_seats)]
        for p in planets:
            owner = p[1]
            if 0 <= owner < num_seats and p[5] > 5 and rng.random() < 0.3:
                actions[owner].append([p[0], rng.uniform(0.0, 6.283),
                                       int(p[5] // 2)])
        env.step(actions)
    return env.state[0].observation, int(env.info.get("seed", seed))


@pytest.mark.parametrize("seed", [7, 42, 100])
def test_combat_at_matches_fast_sim_winner(seed: int):
    """For real game states, every predicted combat winner matches
    the env's actual ownership change at that turn."""
    num_seats = 2
    obs, ep_seed = _step_env_to_obs(seed, warmup=30, num_seats=num_seats)
    world = World.from_obs(obs, episode_seed=ep_seed)
    horizon = 50

    snap_S = fs_from_obs(obs, episode_seed=ep_seed, num_seats=num_seats)
    # Track per-planet owner history under empty actions.
    snap = fs_clone(snap_S)
    owner_at_step: dict[int, dict[int, int]] = {}  # pid -> {t -> owner}
    for t in range(horizon + 1):
        planets = (snap.obs["planets"] if isinstance(snap.obs, dict)
                   else snap.obs.planets)
        for p in planets:
            pid = int(p[0])
            owner_at_step.setdefault(pid, {})[t] = int(p[1])
        if t < horizon:
            snap = fs_step(snap, [[] for _ in range(num_seats)],
                            in_place=True)

    # For every owner CHANGE in the env history, find the
    # corresponding predicted combat and assert winner matches.
    for pid, history in owner_at_step.items():
        for t in range(1, horizon + 1):
            if t not in history or (t - 1) not in history:
                continue
            if history[t] == history[t - 1]:
                continue
            outcome = world.combat_at(pid, t, horizon=horizon)
            if outcome is None:
                # The env may surface an owner change from a multi-step
                # sequence we didn't enumerate (e.g. comet despawn).
                # Skip these — the parity claim is "OUR predictions
                # match the env", not "we predict all events".
                continue
            assert outcome.winner_owner == history[t], (
                f"seed={seed} pid={pid} t={t}: "
                f"predicted winner={outcome.winner_owner} "
                f"actual={history[t]}"
            )
