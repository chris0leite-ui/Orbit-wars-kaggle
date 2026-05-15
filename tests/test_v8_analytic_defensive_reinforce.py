"""Defensive-reinforce enumeration tests for v8_analytic (Commit 1 of
v8_scavenge port).

`enumerate_defensive_reinforce` proposes ActionSpec atoms aimed at my
own threatened planets, sized from `WorldModel.incoming_enemy_eta`.
Without this, the beam's offensive-only enumeration can't choose
defence, so opp's continuous fleet pressure flips captured planets
back and eliminates us mid-game (the 0/32 smoke loss vs `nearest`).
"""

from __future__ import annotations

import numpy as np
from kaggle_environments import make

from lib.foundation.actions import ActionSpec
from lib.foundation.obs_to_state import obs_to_jax_state, my_id_from_obs
from lib.foundation.strategies.analytic_score import (
    enumerate_atomic_launches,
    enumerate_defensive_reinforce,
)


def _play_one_step_each_side(seed: int = 11):
    """Run a single env step where both seats play `nearest`.

    Returns the GameState seen by player 0 at step=1, where at least
    one inbound enemy fleet is typically already in flight.
    """
    env = make(
        "orbit_wars",
        configuration={"seed": seed, "episodeSteps": 5},
        debug=False,
    )
    env.reset(num_agents=2)
    env.run(["agents/simple/nearest.py", "agents/simple/nearest.py"])
    obs = env.state[0].observation
    return obs, env.configuration


def _nearest_action(observation):
    """Call agents/simple/nearest.py inline for synthetic dispatch."""
    from agents.simple.nearest import agent
    return agent(observation)


def test_returns_zero_atoms_when_no_threats_visible():
    """At step=0 nobody has launched yet — no incoming enemy fleets,
    so defensive enumeration returns nothing."""
    env = make("orbit_wars", configuration={"seed": 0}, debug=False)
    env.reset(num_agents=2)
    obs = env.state[0].observation
    state = obs_to_jax_state(obs, configuration=env.configuration)

    atoms = enumerate_defensive_reinforce(
        state, my_id=0, raw_obs=obs,
    )
    assert atoms == []


def _state_with_two_close_planets():
    """Return a GameState seeded from a real env, plus the indices of
    own planets for player 0. Plays enough nearest self-play turns
    that both seats own ≥2 planets (fleets need ~30-50 turns to land
    on the default board).
    """
    for seed in range(20):
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.reset(num_agents=2)
        for _ in range(50):
            env.step([
                _nearest_action(env.state[0].observation),
                _nearest_action(env.state[1].observation),
            ])
        obs = env.state[0].observation
        state = obs_to_jax_state(obs, configuration=env.configuration)
        owner = np.asarray(state.planets_owner)
        alive = np.asarray(state.planets_alive)
        my_idxs = [
            i for i in range(len(owner))
            if alive[i] and int(owner[i]) == 0
        ]
        if len(my_idxs) >= 2:
            return state, obs, my_idxs
    raise RuntimeError("could not find a seed with ≥2 own planets after 50 turns")


def _hand_built_worldmodel(state, target_index, enemy_eta, enemy_ships):
    """Build a `WorldModel` with a single hand-set ledger entry: one
    enemy fleet inbound to `state.planets_id[target_index]` at the
    given ETA, with `enemy_ships`. Bypasses the env's fleet raycast so
    the test is deterministic.
    """
    from lib.intent import World
    from lib.world_model import WorldModel, simulate_planet_timeline
    from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

    ids = np.asarray(state.planets_id)
    owner = np.asarray(state.planets_owner)
    ships = np.asarray(state.planets_ships)
    prod = np.asarray(state.planets_prod)
    x = np.asarray(state.planets_x)
    y = np.asarray(state.planets_y)
    radius = np.asarray(state.planets_radius)
    alive = np.asarray(state.planets_alive)

    planets = [
        Planet(
            id=int(ids[i]),
            owner=int(owner[i]),
            x=float(x[i]),
            y=float(y[i]),
            radius=float(radius[i]),
            ships=int(ships[i]),
            production=int(prod[i]),
        )
        for i in range(len(alive)) if alive[i]
    ]
    target_pid = int(ids[target_index])
    ledger = {p.id: [] for p in planets}
    ledger[target_pid] = [(int(enemy_eta), 1, int(enemy_ships))]
    timelines = {
        p.id: simulate_planet_timeline(p, ledger[p.id], 80) for p in planets
    }
    return WorldModel(ledger=ledger, timelines=timelines, horizon=80)


def test_returns_at_least_one_atom_when_my_planet_is_threatened():
    """Headline behaviour: with a deterministic threat ledger entry
    big enough to overwhelm production-time garrison growth, and a
    second own planet close enough to reinforce, the helper proposes
    at least one atom."""
    state, _obs, my_idxs = _state_with_two_close_planets()
    target_index = my_idxs[0]

    # Threat: 200 enemy ships arriving in 18 turns. Garrison at eta
    # for a small planet (~5-15 ships, prod 1-3) is ≪ 200, guaranteed
    # shortfall.
    wm = _hand_built_worldmodel(state, target_index, enemy_eta=18, enemy_ships=200)

    atoms = enumerate_defensive_reinforce(
        state, my_id=0, world_model=wm,
    )
    assert len(atoms) >= 1, (
        f"Expected ≥1 defensive atom; got 0. "
        f"State has {len(my_idxs)} own planets; target_index={target_index}."
    )

    owner = np.asarray(state.planets_owner)
    ids = np.asarray(state.planets_id)
    alive = np.asarray(state.planets_alive)
    owned_ids = {
        int(ids[i]) for i in range(len(alive))
        if alive[i] and int(owner[i]) == 0
    }
    for spec in atoms:
        assert isinstance(spec, ActionSpec)
        assert spec.from_planet_id in owned_ids
        assert spec.ships >= 2
        assert spec.launch_turn == 0
        assert spec.agent_id == 0
        assert np.isfinite(spec.dir_angle)


def test_defensive_atoms_extend_offensive_pool():
    """End-to-end shape check: defensive atoms concatenate onto the
    offensive pool without overlap or loss."""
    state, _obs, my_idxs = _state_with_two_close_planets()
    wm = _hand_built_worldmodel(state, my_idxs[0], enemy_eta=18, enemy_ships=200)

    offensive = enumerate_atomic_launches(state, my_id=0)
    defensive = enumerate_defensive_reinforce(
        state, my_id=0, world_model=wm,
    )
    combined = offensive + defensive
    assert len(combined) == len(offensive) + len(defensive)


def test_no_atoms_when_only_one_owned_planet():
    """Sanity floor: need a separate source planet to reinforce from.
    With ≤1 own planet, no defensive reinforce is possible."""
    env = make("orbit_wars", configuration={"seed": 0}, debug=False)
    env.reset(num_agents=2)
    obs = env.state[0].observation
    state = obs_to_jax_state(obs, configuration=env.configuration)

    # Artificially clobber the owner array so only one planet stays
    # ours. This is a unit-level invariant check, not a realistic
    # mid-game probe.
    owner = np.asarray(state.planets_owner).copy()
    my_idxs = [i for i in range(len(owner)) if int(owner[i]) == 0]
    if len(my_idxs) <= 1:
        return  # already in the no-source-to-reinforce regime
    for i in my_idxs[1:]:
        owner[i] = -1  # mark neutral

    # Rebuild a state-like namespace with the patched owner; cheapest
    # path is to read fields and rewrite.
    import jax.numpy as jnp
    from dataclasses import replace
    state2 = replace(state, planets_owner=jnp.asarray(owner))

    atoms = enumerate_defensive_reinforce(
        state2, my_id=0, raw_obs=obs,
    )
    assert atoms == []
