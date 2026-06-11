"""Parity gate for the oracle agent's engine core.

Same contract as tests/test_ledger_forecast.py, applied to the ported
agents/oracle/engine.py: absent new launches, the World ledger must predict
every planet's (owner, ships) exactly — orbit rotation, comet paths and
expiry, fleet flight with swept-disk collision, production order, and
combat resolution included.
"""

import importlib.util
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load_engine():
    spec = importlib.util.spec_from_file_location(
        "oracle_engine_under_test", REPO / "agents" / "oracle" / "engine.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _scripted(player_seed):
    """Agent that fires a few deterministic launches early, then goes quiet."""
    import random

    def agent(obs, configuration=None):
        step = obs["step"] if isinstance(obs, dict) else obs.step
        me = obs["player"] if isinstance(obs, dict) else obs.player
        planets = obs["planets"] if isinstance(obs, dict) else obs.planets
        if step not in (2, 5, 9, 13):
            return []
        rng = random.Random(player_seed * 1000 + step)
        moves = []
        mine = [p for p in planets if p[1] == me and p[5] > 3]
        targets = [p for p in planets if p[1] != me]
        for p in mine:
            t = rng.choice(targets)
            angle = math.atan2(t[3] - p[3], t[2] - p[2])
            moves.append([p[0], angle + rng.uniform(-0.2, 0.2),
                          max(1, p[5] // 2)])
        return moves

    return agent


@pytest.mark.parametrize("seed", [11, 42])
def test_forecast_matches_engine(seed):
    from kaggle_environments import make

    mod = _load_engine()
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([_scripted(0), _scripted(1)])
    steps = env.steps

    # windows: (obs step, max forecast depth) — stop before comet spawns
    windows = [(16, 30), (25, 22), (60, 35), (100, 40)]
    checked = 0
    for s, depth in windows:
        if s + depth >= len(steps):
            continue
        obs = steps[s][0].observation
        world = mod.World(obs)
        world.build_ledger()
        for dt in range(1, min(depth, world.horizon) + 1):
            actual = {p[0]: (p[1], p[5])
                      for p in steps[s + dt][0].observation["planets"]}
            for i in range(world.n_planets):
                pid = world.pid[i]
                pred_owner = world.post_owner[i][dt]
                pred_ships = world.post_ships[i][dt]
                if pred_owner == -2:
                    assert pid not in actual, (
                        f"seed {seed} s{s} dt{dt}: planet {pid} predicted "
                        f"gone but engine still has it")
                    continue
                assert pid in actual, (
                    f"seed {seed} s{s} dt{dt}: planet {pid} predicted alive "
                    f"but engine removed it")
                ao, ash = actual[pid]
                assert (pred_owner, pred_ships) == (ao, ash), (
                    f"seed {seed} s{s} dt{dt} planet {pid}: predicted "
                    f"owner={pred_owner} ships={pred_ships}, engine has "
                    f"owner={ao} ships={ash}")
                checked += 1
    assert checked > 500, f"too few comparisons ran ({checked})"
